import torch

from .detector3d_template import Detector3DTemplate


class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.bev_shape_module_lists, self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.bev_shape_module_lists:
            batch_dict = cur_module(batch_dict)
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0

        loss_bev_shape, tb_dict = self.bev_shape_modules.dense_head.get_loss_bev_shape()
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn + loss_bev_shape
        
        if hasattr(self.backbone_3d, 'get_loss'):
            loss_backbone3d, tb_dict = self.backbone_3d.get_loss(tb_dict)
            loss += loss_backbone3d

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_bev_shape': loss_bev_shape.item(),
            'loss_rcnn': loss_rcnn.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict
