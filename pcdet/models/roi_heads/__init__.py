from .second_head import SECONDHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'SECONDHead': SECONDHead,
    'VoxelRCNNHead': VoxelRCNNHead,
}
