import mmcv
import os.path as osp
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer

# Output name and save directory
out_file = 'rescuenet_sample'
save_dir = './work_dirs/vis_rescuenet'

# Paths to input image and segmentation label
image_path = 'data/rescuenet/img_dir/test/15005.jpg'
label_path = 'data/rescuenet/ann_dir/test/15005_lab.png'

# Load image and label
image = mmcv.imread(image_path, 'color')
# Load segmentation label (not RGB)
sem_seg = mmcv.imread(label_path, 'unchanged')

# Convert to tensor and wrap into PixelData
sem_seg = torch.from_numpy(sem_seg)
gt_sem_seg = PixelData(data=sem_seg)

# Wrap into SegDataSample structure
data_sample = SegDataSample()
data_sample.gt_sem_seg = gt_sem_seg

# Define the RescueNet visualizer
seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=save_dir)

# Apply RescueNet dataset meta (classes and color palette)
seg_local_visualizer.dataset_meta = dict(
    classes=(
        'background', 'water', 'building-no-damage', 'building-medium-damage',
        'building-major-damage', 'building-total-destruction', 'vehicle',
        'road-clear', 'road-blocked', 'tree', 'pool'
    ),
    palette=[
        [0, 0, 0], [61, 230, 250], [180, 120, 120], [235, 255, 7],
        [255, 184, 6], [255, 0, 0], [255, 0, 245],
        [140, 140, 140], [160, 150, 20], [4, 250, 7], [255, 235, 0]
    ])

# Create visualization; set show=True if you want to view it interactively
seg_local_visualizer.add_datasample(
    name=out_file,
    image=image,
    data_sample=data_sample,
    show=False  # Set to True if you want to visualize it on-screen
)


# 15005.jpg
# 15821.jpg
