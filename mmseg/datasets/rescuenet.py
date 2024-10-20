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
            'background', 'water', 'building_no_damage', 'building_minor_damage',
            'building_major_damage', 'building_total_destruction', 'road_clear',
            'road_blocked', 'vehicle', 'tree', 'pool'
        ),
        palette=[
            [0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0],
            [255, 165, 0], [255, 0, 0], [0, 128, 0], [128, 0, 0],
            [128, 128, 0], [0, 255, 255], [255, 0, 255]
        ]
    )

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='_lab.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            **kwargs)

        # Ensure image and annotation paths exist
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args), "Image path not found."
        assert osp.isfile(self.ann_file), "Annotation file not found."
