import os.path as osp

import mmengine.fileio as fileio
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RescueNetDataset(BaseSegDataset):
    """RescueNet dataset for semantic segmentation.

    Args:
        split (str): Split txt file for the dataset (e.g., 'train', 'val').
    """
    from collections import OrderedDict

    color_encoding = OrderedDict([
        ('unlabeled', (0, 0, 0)),
        ('water', (61, 230, 250)),
        ('building-no-damage', (180, 120, 120)),
        ('building-medium-damage', (235, 255, 7)),
        ('building-major-damage', (255, 184, 6)),
        ('building-total-destruction', (255, 0, 0)),
        ('vehicle', (255, 0, 245)),
        ('road-clear', (140, 140, 140)),
        ('road-blocked', (160, 150, 20)),
        ('tree', (4, 250, 7)),
        ('pool', (255, 235, 0))
    ])

    METAINFO = dict(
        classes=tuple(color_encoding.keys()),
        palette=list(color_encoding.values())
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_lab.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

        # Ensure image and annotation paths exist
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args), "Image path not found."
