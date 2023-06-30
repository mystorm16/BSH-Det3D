import torch
import torch.nn as nn
from matplotlib import pyplot as plt

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['pillar_voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                64,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['pillar_spatial_features'] = batch_spatial_features

        if self.training == True:
            batch_dict['c_bm_pillar_features'] = torch.ones(
                batch_dict['c_bm_voxels'].shape[0], 1,
                dtype=pillar_features.dtype,
                device=pillar_features.device)
            c_bm_pillar_features, c_bm_coords = batch_dict['c_bm_pillar_features'], batch_dict['c_bm_voxel_coords']
            c_bm_batch_spatial_features = []
            for batch_idx in range(batch_size):
                c_bm_spatial_feature = torch.zeros(
                    1, 1 * 176 * 200,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)

                batch_mask = c_bm_coords[:, 0] == batch_idx
                this_coords = c_bm_coords[batch_mask, :]
                indices = this_coords[:, 1] + this_coords[:, 2] * 176 + this_coords[:, 3]
                indices = indices.type(torch.long)
                c_bm_pillars = c_bm_pillar_features[batch_mask, :]
                c_bm_pillars = c_bm_pillars.t()
                c_bm_spatial_feature[:, indices] = c_bm_pillars
                c_bm_batch_spatial_features.append(c_bm_spatial_feature)

            c_bm_batch_spatial_features = torch.stack(c_bm_batch_spatial_features, 0)
            c_bm_batch_spatial_features = c_bm_batch_spatial_features.view(batch_size, 1 * 1, 200, 176)
            batch_dict['c_bm_spatial_features'] = c_bm_batch_spatial_features
        return batch_dict
