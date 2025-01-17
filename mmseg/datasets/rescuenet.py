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
    METAINFO = dict(
        classes=(
            'background', 'water', 'building-no-damage', 'building-medium-damage',
            'building-major-damage', 'building-total-destruction', 'vehicle',
            'road-clear', 'road-blocked', 'tree', 'pool'
        ),
        palette=[
            [0, 0, 0], [61, 230, 250], [180, 120, 120], [235, 255, 7],
            [255, 184, 6], [255, 0, 0], [255, 0, 245],
            [140, 140, 140], [160, 150, 20], [4, 250, 7], [255, 235, 0]
        ]
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
