import os
import numpy as np
import mmcv
from mmcv.image import imresize

BASE_PATH = 'work_dirs/vis_rescuenet/vis_data/vis_image/renamed_files/'
VARIANT_KEY = 'groupshift'  # or 'sparseshift'

# Find all *_lab.png files (ground truths)
all_files = os.listdir(BASE_PATH)
gt_files = [f for f in all_files if f.endswith('_lab.png')]

# Config crop size
CROP_SIZE = (512, 512)  # (height, width)


def center_crop(img, crop_size):
    """Crop the center patch of specified size."""
    h, w = img.shape
    ch, cw = crop_size
    top = max(0, (h - ch) // 2)
    left = max(0, (w - cw) // 2)
    return img[top:top+ch, left:left+cw]


e1 = 0
e2 = 0
matched = 0

for gt_file in gt_files:
    prefix = gt_file.replace('_lab.png', '')

    gt_path = os.path.join(BASE_PATH, gt_file)
    baseline_path = os.path.join(
        BASE_PATH, f'{prefix}_pred_gt_0_original_baseline.png')
    variant_path = os.path.join(
        BASE_PATH, f'{prefix}_pred_gt_0_{VARIANT_KEY}.png')

    if not (os.path.exists(gt_path) and os.path.exists(baseline_path) and os.path.exists(variant_path)):
        print(f"❌ Skipping: Missing files for {prefix}")
        continue

    # Read and convert to grayscale if needed
    gt_mask = mmcv.imread(gt_path, 'unchanged')
    baseline_pred = mmcv.imread(baseline_path, 'unchanged')
    variant_pred = mmcv.imread(variant_path, 'unchanged')

    if baseline_pred.ndim == 3:
        baseline_pred = baseline_pred[:, :, 0]
    if variant_pred.ndim == 3:
        variant_pred = variant_pred[:, :, 0]

    # Resize prediction masks to match GT
    target_shape = gt_mask.shape[::-1]
    if baseline_pred.shape != gt_mask.shape:
        baseline_pred = imresize(
            baseline_pred, target_shape, interpolation='nearest')
    if variant_pred.shape != gt_mask.shape:
        variant_pred = imresize(
            variant_pred, target_shape, interpolation='nearest')

    # Center crop
    gt_mask = center_crop(gt_mask, CROP_SIZE)
    baseline_pred = center_crop(baseline_pred, CROP_SIZE)
    variant_pred = center_crop(variant_pred, CROP_SIZE)

    # Pixel-level correctness
    baseline_correct = (baseline_pred == gt_mask).flatten()
    variant_correct = (variant_pred == gt_mask).flatten()

    e1 += np.logical_and(baseline_correct, ~variant_correct).sum()
    e2 += np.logical_and(~baseline_correct, variant_correct).sum()
    matched += 1

print(f"\n✅ Compared {matched} image sets using variant: {VARIANT_KEY}")
print(f"e1 (baseline correct, variant wrong): {e1}")
print(f"e2 (variant correct, baseline wrong): {e2}")

# McNemar's Chi-Square Test
if e1 + e2 > 0:
    from scipy.stats import chi2
    chi2_stat = ((abs(e1 - e2) - 1) ** 2) / (e1 + e2)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    print(f"\nChi-square: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Result: 🔥 Significant difference")
    else:
        print("Result: ❄️ Not significant")
else:
    print("Not enough disagreement cases to compute significance.")
