import numpy as np
import mmcv
from mmseg.apis import inference_model, init_model
from mmcv.image import imresize

# CONFIG and CHECKPOINT
config_file = 'configs/segnext/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-o_mscan_original_baseline.py'
checkpoint_file = 'work_dirs/RTX3090_segnext_mscan-groupshift-t_ablation_v2_var-o_mscan_original_baseline/iter_20000.pth'

# Init model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Inference on a resized image
image_path = 'data/rescuenet/img_dir/test/10816.jpg'
original_image = mmcv.imread(image_path, 'color')
image_resized = imresize(original_image, (512, 512))  # Match training size

# Run inference
result = inference_model(model, image_resized)
pred_mask = result.pred_sem_seg.data.squeeze().cpu().numpy().astype('uint8')
print('Unique values in prediction:', np.unique(pred_mask))

# Save resized prediction mask
out_path = './work_dirs/pred_dir/10816_pred_baseline.png'
mmcv.imwrite(pred_mask, out_path)
print(f"Saved predicted mask to {out_path}")


# nohup python custom_inference.py > work_dirs/vis_rescuenet/custom_inference_log.txt 2>&1 &