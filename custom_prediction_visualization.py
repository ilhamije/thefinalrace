import mmcv
import torch
import os.path as osp

from mmseg.apis import init_model, inference_model
from mmengine.structures import PixelData
from mmseg.visualization import SegLocalVisualizer
from mmcv.image import imresize

# === 1. Define paths ===
# 10816.jpg
# 10943.jpg
# 11006.jpg
img_path = 'data/rescuenet/img_dir/train/10816.jpg'
gt_path = 'data/rescuenet/ann_dir/train/10816_lab.png'
img_resize_to = (1500, 1125)  # Based on your config

# === 2. Load & resize image ===
image = mmcv.imread(img_path, 'color')
image = imresize(image, img_resize_to)

# === 3. Load & resize ground truth mask ===
sem_seg = mmcv.imread(gt_path, 'unchanged')  # labelTrainIds format
sem_seg = imresize(sem_seg, img_resize_to, interpolation='nearest')
sem_seg = torch.from_numpy(sem_seg)

# === 4. Initialize model ===
# config_file = 'configs/segnext/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-b_shift3x3.py'
# checkpoint_file = 'work_dirs/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-b_shift3x3/iter_20000.pth'
# config_file = 'configs/segnext/RTX3090_segnext_mscan-sparseshift-t_ablation_v2_var-b_colorjitter.py'
# checkpoint_file = 'work_dirs/RTX3090_segnext_mscan-sparseshift-t_ablation_v2_var-b_colorjitter/iter_20000.pth'
config_file = 'configs/segnext/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-o_mscan_original_baseline.py'
checkpoint_file = 'work_dirs/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-o_mscan_original_baseline/iter_20000.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')

# === 5. Run inference ===
result = inference_model(model, image)

# === 6. Attach resized ground truth to prediction result ===
result.gt_sem_seg = PixelData(data=sem_seg)

# === 7. Set up visualizer ===
basename = osp.basename(img_path)
filename_wo_ext = osp.splitext(basename)[0]

seg_local_visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir='work_dirs/vis_rescuenet'
)
seg_local_visualizer.dataset_meta = dict(
    classes=model.dataset_meta['classes'],
    palette=model.dataset_meta['palette']
)

# === 8. Save prediction, ground truth, and combined overlays ===
seg_local_visualizer.add_datasample(
    name=f'{filename_wo_ext}_pred',
    image=image,
    data_sample=result,
    draw_pred=True,
    draw_gt=False,
    show=False
)

seg_local_visualizer.add_datasample(
    name=f'{filename_wo_ext}_gt',
    image=image,
    data_sample=result,
    draw_pred=False,
    draw_gt=True,
    show=False
)

seg_local_visualizer.add_datasample(
    name=f'{filename_wo_ext}_pred_gt',
    image=image,
    data_sample=result,
    draw_pred=True,
    draw_gt=True,
    show=False
)
