import logging
import os
import pickle
import random
import shutil
import subprocess
import SharedArray

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(angle.shape[0])
    ones = angle.new_ones(angle.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


def sa_create(name, var):
    x = SharedArray.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reverse_sparse_trilinear_interpolate_torch(feat, b, zyx, normalize=False):
    """
    Args:
        im: (Z, H, W, C) [z, y, x]
        x: (N)
        y: (N)
        z: (N)

    Returns:

    """
    im, spatial_shape = feat.dense(), feat.spatial_shape
    x, y, z = zyx[..., 2], zyx[..., 1], zyx[..., 0]
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    z0 = torch.floor(z).long()
    z1 = z0 + 1

    if normalize:
        z0_mask = 1
        z1_mask = 1
        y0_mask = 1
        y1_mask = 1
        x0_mask = 1
        x1_mask = 1
    else:
        z0_mask = ((z0 >= 0) & (z0 < spatial_shape[0])).unsqueeze(-1)
        z1_mask = ((z1 >= 0) & (z1 < spatial_shape[0])).unsqueeze(-1)
        y0_mask = ((y0 >= 0) & (y0 < spatial_shape[1])).unsqueeze(-1)
        y1_mask = ((y1 >= 0) & (y1 < spatial_shape[1])).unsqueeze(-1)
        x0_mask = ((x0 >= 0) & (x0 < spatial_shape[2])).unsqueeze(-1)
        x1_mask = ((x1 >= 0) & (x1 < spatial_shape[2])).unsqueeze(-1)

    w000 = torch.abs((z1.type_as(z) - z) * (y1.type_as(y) - y) * (x1.type_as(x) - x))
    w010 = torch.abs(-(z1.type_as(z) - z) * (y0.type_as(y) - y) * (x1.type_as(x) - x))
    w001 = torch.abs(-(z1.type_as(z) - z) * (y1.type_as(y) - y) * (x0.type_as(x) - x))
    w011 = torch.abs((z1.type_as(z) - z) * (y0.type_as(y) - y) * (x0.type_as(x) - x))
    w100 = torch.abs(-(z0.type_as(z) - z) * (y1.type_as(y) - y) * (x1.type_as(x) - x))
    w110 = torch.abs((z0.type_as(z) - z) * (y0.type_as(y) - y) * (x1.type_as(x) - x))
    w101 = torch.abs((z0.type_as(z) - z) * (y1.type_as(y) - y) * (x0.type_as(x) - x))
    w111 = torch.abs(-(z0.type_as(z) - z) * (y0.type_as(y) - y) * (x0.type_as(x) - x))

    x0 = torch.clamp(x0, 0, spatial_shape[2] - 1)
    x1 = torch.clamp(x1, 0, spatial_shape[2] - 1)
    y0 = torch.clamp(y0, 0, spatial_shape[1] - 1)
    y1 = torch.clamp(y1, 0, spatial_shape[1] - 1)
    z0 = torch.clamp(z0, 0, spatial_shape[0] - 1)
    z1 = torch.clamp(z1, 0, spatial_shape[0] - 1)

    I000 = im[b, :, z0, y0, x0] * z0_mask * y0_mask * x0_mask  # [1, 65536, 352]
    I010 = im[b, :, z0, y1, x0] * z0_mask * y1_mask * x0_mask
    I001 = im[b, :, z0, y0, x1] * z0_mask * y0_mask * x1_mask
    I011 = im[b, :, z0, y1, x1] * z0_mask * y1_mask * x1_mask
    I100 = im[b, :, z1, y0, x0] * z1_mask * y0_mask * x0_mask
    I110 = im[b, :, z1, y1, x0] * z1_mask * y1_mask * x0_mask
    I101 = im[b, :, z1, y0, x1] * z1_mask * y0_mask * x1_mask
    I111 = im[b, :, z1, y1, x1] * z1_mask * y1_mask * x1_mask

    ans = I000 * w000.unsqueeze(-1) + I010 * w010.unsqueeze(-1) + I001 * w001.unsqueeze(-1) + I011 * w011.unsqueeze(
        -1) + I100 * w100.unsqueeze(-1) + I110 * w110.unsqueeze(-1) + I101 * w101.unsqueeze(-1) + I111 * w111.unsqueeze(
        -1)
    return ans
