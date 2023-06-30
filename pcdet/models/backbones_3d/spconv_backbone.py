from functools import partial

import torch
import torch.nn as nn
from tools.visual_utils.open3d_vis_utils import draw_scenes

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None, output_padding=0):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'fixspconv':
        conv = fixSparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               bias=False, indice_key=indice_key, defaultvalue=1.0)
        conv.requires_grad_(False)
    elif conv_type == 'spdeconv':
        conv = spconv.SparseConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                            bias=False, indice_key=indice_key, output_padding=output_padding)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    elif conv_type == 'submbias':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=True, indice_key=indice_key)
    elif conv_type == 'maxpool':
        conv = spconv.SparseMaxPool3d(kernel_size, stride=stride, padding=padding)
    else:
        raise NotImplementedError

    if norm_fn is not None:
        m = spconv.SparseSequential(
            conv,
            norm_fn(out_channels),
            nn.ReLU(),
        )
    else:
        m = spconv.SparseSequential(
            conv
        )
    return m


class fixSparseConv3d(spconv.SparseConv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, indice_key=None,
                 defaultvalue=1.0):
        super(fixSparseConv3d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                              bias=bias, indice_key=indice_key)
        self.weight.data.fill_(defaultvalue)


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        self.type = kwargs['vfe_type'] if kwargs.__contains__('type') else None
        self.expand = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }
        self.type = kwargs['type'] if kwargs.__contains__('type') else None

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if self.type == 'center':
            voxel_features, voxel_coords = batch_dict['center_voxel_features'], batch_dict['center_voxel_coords']
        else:
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelBackBoneDeconv(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.y_shift = model_cfg.get("SHIFT", 0)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        self.sparse_shape[1] += self.y_shift * 2

        block = post_act_block
        channels = [16, 32, 64]

        self.conv1 = spconv.SparseSequential(
            block(input_channels, channels[0], 3, norm_fn=norm_fn, padding=1, indice_key='spconv1', conv_type='spconv'),
        )  # 1

        self.conv2 = spconv.SparseSequential(
            block(channels[0], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )  # 2

        self.conv3 = spconv.SparseSequential(
            block(channels[1], channels[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(channels[2], channels[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )  # 2

        self.deconv4 = spconv.SparseSequential(
            block(channels[2], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4',
                  conv_type='spdeconv'),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        )

        self.deconv5 = spconv.SparseSequential(
            block(channels[1], channels[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5',
                  conv_type='spdeconv', output_padding=[0, 3, 3]),
            block(channels[1], channels[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm5')
        )
        self.num_point_features = channels[1]

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['occ_voxel_features'], \
                                       batch_dict['occ_voxel_coords'].int()

        batch_size = batch_dict['batch_size']
        if self.y_shift > 0:
            voxel_features, voxel_coords = self.add_shift(voxel_features, voxel_coords)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x_conv1 = self.conv1(input_sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv2d = self.deconv4(x_conv3)
        x_conv1d = self.deconv5(x_conv2d)

        batch_dict.update({
            'encoded_spconv_tensor': x_conv1d,
            'encoded_spconv_tensor_stride': 1
        })
        return batch_dict

    def remove_shift(self, sparse_feat):
        y_max = self.sparse_shape[1] - 2 * self.y_shift
        sparse_feat.indices[..., 2] -= self.y_shift
        keep_inds = (sparse_feat.indices[..., 2] >= 0) & (sparse_feat.indices[..., 2] < y_max)
        sparse_feat.features = sparse_feat.features[keep_inds, :]
        sparse_feat.indices = sparse_feat.indices[keep_inds, :]
        sparse_feat.spatial_shape[1] -= self.y_shift * 2
        return sparse_feat

    def add_shift(self, voxel_features, voxel_coords):
        y_max = self.sparse_shape[1] - 2 * self.y_shift
        left_ind = voxel_coords[..., 2] < self.y_shift
        right_ind = voxel_coords[..., 2] >= (y_max - self.y_shift)
        left_feat, left_coords, right_feat, right_coords = voxel_features[left_ind, :], voxel_coords[left_ind,
                                                                                        :].clone(), voxel_features[
                                                                                                    right_ind,
                                                                                                    :], voxel_coords[
                                                                                                        right_ind,
                                                                                                        :].clone()
        right_coords[..., 2] -= y_max
        left_coords[..., 2] += y_max
        all_features = torch.cat([right_feat, voxel_features, left_feat], dim=0)
        all_coords = torch.cat([right_coords, voxel_coords, left_coords], dim=0)
        all_coords[..., 2] += self.y_shift
        return all_features, all_coords
