from mmseg.visualization import SegLocalVisualizer
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import mmcv
import torch
from mmcv.image import imresize

# === Load and resize ===
image = mmcv.imread('data/rescuenet/img_dir/train/10816.jpg', 'color')
gt_mask = mmcv.imread('data/rescuenet/ann_dir/train/10816_lab.png', 'unchanged')
pred_mask = mmcv.imread('work_dirs/pred_dir/10816_pred.png', 'unchanged')

# Resize all to 512x512 for consistent overlay
image = imresize(image, (512, 512))
gt_mask = imresize(gt_mask, (512, 512), interpolation='nearest')
pred_mask = imresize(pred_mask, (512, 512), interpolation='nearest')

# Wrap into data structures
data_sample = SegDataSample()
data_sample.gt_sem_seg = PixelData(data=torch.from_numpy(gt_mask))
data_sample.pred_sem_seg = PixelData(data=torch.from_numpy(pred_mask))

# Setup visualizer
visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir='work_dirs/vis_rescuenet'
)

# Dataset meta
visualizer.dataset_meta = dict(
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

# Visualize GT and prediction overlayed
visualizer.add_datasample(
    name='15005_overlay',
    image=image,
    data_sample=data_sample,
    show=True,
    draw_gt=True,
    draw_pred=True
)

print("✅ Visualization complete. Check the output at: work_dirs/vis_rescuenet")


# === run with nohup ===
# nohup python custom_inference_masking.py > custom_inference_masking.log 2>&1 &
