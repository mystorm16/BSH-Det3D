import numpy as np
import scipy
import torch

perm_dict = {
    "xyz": [0, 1, 2, 0, 2],
    "zyx": [2, 1, 0, 1, 3],
}


def coords3inds(coords, nz, ny, nx):
    coords = coords.to(torch.int32)
    gperm = torch.tensor(nx * ny * nz, dtype=torch.int32, device="cuda")
    gperm1 = torch.tensor(nx * ny, dtype=torch.int32, device="cuda")
    gperm2 = torch.tensor(nx, dtype=torch.int32, device="cuda")

    bdim = coords[:, 0] * gperm
    zdim = coords[:, 1] * gperm1
    ydim = coords[:, 2] * gperm2
    xdim = coords[:, 3]
    inds = bdim + zdim + ydim + xdim
    return inds.to(torch.int32)


def inds3coords(inds, nz, ny, nx):
    gperm = torch.tensor(nx * ny * nz, dtype=torch.int32, device="cuda")
    gperm1 = torch.tensor(nx * ny, dtype=torch.int32, device="cuda")
    gperm2 = torch.tensor(nx, dtype=torch.int32, device="cuda")

    b_dim = torch.div(inds, gperm)
    inds_leftover = inds - b_dim * gperm
    z_dim = torch.div(inds_leftover, gperm1)
    inds_leftover = inds_leftover - z_dim * gperm1
    y_dim = torch.div(inds_leftover, gperm2)
    x_dim = inds_leftover - y_dim * gperm2
    coords = torch.stack((b_dim, z_dim, y_dim, x_dim), axis=-1)
    return coords


def coords6inds(coords, nz, ny, nx, sz, sy, sx):
    gperm0 = nz * ny * nx * sz * sy * sx
    gperm1 = ny * nx * sz * sy * sx
    gperm2 = nx * sz * sy * sx
    gperm3 = sz * sy * sx
    gperm4 = sy * sx
    gperm5 = sx

    gperm = torch.tensor([gperm0, gperm1, gperm2, gperm3, gperm4, gperm5, 1], dtype=torch.int32, device="cuda").view(1,
                                                                                                                     -1)
    coords = coords.to(torch.int32) * gperm
    return torch.sum(coords, axis=1)


def inds6coords(inds, nz, ny, nx, sz, sy, sx):
    # coords -> n * bnznynxszsysx
    inds = inds.to(torch.int32)
    gperm0 = nz * ny * nx * sz * sy * sx
    gperm1 = ny * nx * sz * sy * sx
    gperm2 = nx * sz * sy * sx
    gperm3 = sz * sy * sx
    gperm4 = sy * sx
    gperm5 = sx

    gperm = torch.tensor([gperm0, gperm1, gperm2, gperm3, gperm4, gperm5, 1], dtype=torch.int32, device="cuda").view(1,
                                                                                                                     -1)
    mods = torch.tensor([9000000, nz, ny, nx, sz, sy, sx], dtype=torch.int32, device="cuda").view(1, -1)
    return torch.fmod(torch.div(inds.view(-1, 1), gperm), mods)


def coords4inds(coords, nz, ny, nx, nl):
    # coords -> n * bnznynxszsysx
    gperm0 = nz * ny * nx * nl
    gperm1 = ny * nx * nl
    gperm2 = nx * nl
    gperm3 = nl

    gperm = torch.tensor([gperm0, gperm1, gperm2, gperm3, 1], dtype=torch.int32, device="cuda").view(1, -1)
    coords = coords.to(torch.int32) * gperm
    return torch.sum(coords, axis=1)


def inds4coords(inds, nz, ny, nx, nl):
    # coords -> n * bnznynxszsysx
    inds = inds.to(torch.int32)
    gperm0 = nz * ny * nx * nl
    gperm1 = ny * nx * nl
    gperm2 = nx * nl
    gperm3 = nl

    gperm = torch.tensor([gperm0, gperm1, gperm2, gperm3, 1], dtype=torch.int32, device="cuda").view(1, -1)
    mods = torch.tensor([90000, nz, ny, nx, nl], dtype=torch.int32, device="cuda").view(1, -1)
    return torch.fmod(torch.div(inds.view(-1, 1), gperm), mods)


def voxel_perm(mask, bs, nz, ny, nx, sz, sy, sx, perm=[0, 3, 5, 2, 4, 6, 1], f_dims=1):
    perm_mask = mask.view(bs, f_dims, nz * sz, ny, sy, nx, sx)
    perm_mask = perm_mask.permute(perm)
    return perm_mask.reshape(bs, ny, nx, nz * sz * sy * sx, f_dims)


def voxelize_pad(voxel_num_points, inds, inverse_indices):
    cluster_num = voxel_num_points.size()[0]
    max_points = torch.max(voxel_num_points).data.cpu().numpy()
    P = inds.shape[0]
    inverse_indices = inverse_indices[inds]
    range_indices = torch.arange(0, P, device="cuda")
    voxel_num_points_addaxis = torch.cumsum(
        torch.cat([torch.zeros([1], dtype=torch.int64, device='cuda'), voxel_num_points[:-1]], dim=0), dim=0)
    indices_voxel = range_indices - voxel_num_points_addaxis[inverse_indices]
    return torch.stack([inverse_indices, indices_voxel], axis=-1), inds, [cluster_num, max_points]


def unq_subvoxel_pid_padding(valid_coords_szsysx, valid_coords_bnznynx, valid_point_label, sz, sy, sx, num_class):
    valid_coords_bnznynx, cluster_inds = torch.unique(valid_coords_bnznynx, return_inverse=True, dim=0)
    valid_coords_cszsysxl = torch.cat(
        [torch.unsqueeze(cluster_inds, -1), valid_coords_szsysx, torch.unsqueeze(valid_point_label, -1)],
        axis=-1)
    voxel_cszsysxl, inverse_indices, voxel_num_points = torch.unique(valid_coords_cszsysxl, dim=0, sorted=False,
                                                                     return_inverse=True, return_counts=True)
    _, inds = torch.sort(inverse_indices)

    subvoxel_inds, point_inds, vp_size = voxelize_pad(voxel_num_points, inds, inverse_indices)
    voxel_coords_bnznynx = valid_coords_bnznynx[voxel_cszsysxl[:, 0]]
    voxel_coords_szsysxl = voxel_cszsysxl[:, 1:]
    voxel_bnynxnzszsysxl = [voxel_coords_bnznynx[:, 0], voxel_coords_bnznynx[:, 2], voxel_coords_bnznynx[:, 3],
                            voxel_coords_bnznynx[:, 1], voxel_coords_szsysxl[:, 0], voxel_coords_szsysxl[:, 1],
                            voxel_coords_szsysxl[:, 2], voxel_coords_szsysxl[:, 3]]
    voxel_bnysynxsx = [voxel_coords_bnznynx[:, 0], voxel_coords_bnznynx[:, 2], voxel_coords_szsysxl[:, 1],
                       voxel_coords_bnznynx[:, 3], voxel_coords_szsysxl[:, 2]]
    return valid_coords_bnznynx, valid_coords_cszsysxl, voxel_coords_bnznynx, voxel_coords_szsysxl, voxel_bnynxnzszsysxl, voxel_bnysynxsx, subvoxel_inds, point_inds, vp_size, voxel_num_points


def unq_subvoxel(valid_coords_cszsysxl, valid_coords_bnznynx, valid_inds, sz, sy, sx):
    valid_coords_cszsysx = valid_coords_cszsysxl[..., :-1]  # C, 4
    voxel_cszsysx, inverse_indices, voxel_num_points = torch.unique(valid_coords_cszsysx, dim=0, sorted=False,
                                                                    return_inverse=True, return_counts=True)
    _, inds = torch.sort(inverse_indices)
    subvoxel_inds, point_inds, vp_size = voxelize_pad(voxel_num_points, inds, inverse_indices)
    ind_vox = torch.zeros([vp_size[0], vp_size[1], 2], dtype=torch.int64, device="cuda")
    ind_vox[subvoxel_inds[:, 0], subvoxel_inds[:, 1]] = valid_inds[point_inds, :] + 1
    voxel_coords_bnznynx = valid_coords_bnznynx[voxel_cszsysx[:, 0]]
    voxel_coords_szsysx = voxel_cszsysx[:, 1:]
    voxel_bnynxnzszsysx = [voxel_coords_bnznynx[:, 0], voxel_coords_bnznynx[:, 2], voxel_coords_bnznynx[:, 3],
                           voxel_coords_bnznynx[:, 1], voxel_coords_szsysx[:, 0], voxel_coords_szsysx[:, 1],
                           voxel_coords_szsysx[:, 2]]
    return ind_vox, voxel_bnynxnzszsysx, vp_size[0]


def get_all_voxel_centers_xyz(bs, finer_grids_num, finer_grid_origin, finer_voxel_size):
    nx, ny, nz = finer_grids_num[0], finer_grids_num[1], finer_grids_num[2]
    x_ind = torch.arange(nx, device="cuda")
    y_ind = torch.arange(ny, device="cuda")
    z_ind = torch.arange(nz, device="cuda")
    x, y, z = torch.meshgrid(x_ind, y_ind, z_ind)
    xyz = torch.stack([x, y, z], axis=-1)
    voxel_centers = (0.5 + xyz.to(torch.float32)) * finer_voxel_size.view(1, 1, 1, 1, 3) + finer_grid_origin.view(1, 1,
                                                                                                                  1, 1,
                                                                                                                  3)
    voxel_centers = voxel_centers.repeat(bs, 1, 1, 1, 1)
    return voxel_centers


def get_all_voxel_centers_zyx(bs, grids_num, grid_origin, voxel_size):
    voxel_size = torch.tensor([voxel_size[2], voxel_size[1], voxel_size[0]], device="cuda")  # oz, oy, ox
    grid_origin = torch.tensor([grid_origin[2], grid_origin[1], grid_origin[0]], device="cuda")  # oz, oy, ox
    nx, ny, nz = grids_num[0], grids_num[1], grids_num[2]
    x_ind = torch.arange(nx, device="cuda")
    y_ind = torch.arange(ny, device="cuda")
    z_ind = torch.arange(nz, device="cuda")
    z, y, x = torch.meshgrid(z_ind, y_ind, x_ind)
    zyx = torch.stack([z, y, x], axis=0)
    voxel_centers = (0.5 + zyx.to(torch.float32)) * voxel_size.view(3, 1, 1, 1) + grid_origin.view(3, 1, 1, 1)
    voxel_centers = voxel_centers.view(1, 3, nz, ny, nx).repeat(bs, 1, 1, 1, 1)
    return voxel_centers


def creat_grid_coords(pntrange, voxel_size):
    nx = int((pntrange[3] - pntrange[0]) / voxel_size[0])
    ny = int((pntrange[4] - pntrange[1]) / voxel_size[1])
    nz = int((pntrange[5] - pntrange[2]) / voxel_size[2])
    voxel_size = np.asarray(voxel_size)
    origin = np.asarray(pntrange[:3])
    # grids_num = np.array([nx, ny, nz], dtype=np.int)
    x_ind = np.arange(nx)
    y_ind = np.arange(ny)
    z_ind = np.arange(nz)
    x, y, z = np.meshgrid(x_ind, y_ind, z_ind, indexing='ij')
    xyz = np.stack([x, y, z], axis=-1)
    voxel_centers = (0.5 + xyz.astype(np.float32)) * voxel_size.reshape(1, 1, 1, 3) + origin.reshape(1, 1, 1, 3)
    return voxel_centers
