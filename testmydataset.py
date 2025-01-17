from mmseg.datasets import RescueNetDataset

# Instantiate the dataset directly
dataset = RescueNetDataset(
    data_prefix={'img_path': 'data/rescuenet/img_dir/train', 
                 'seg_map_path': 'data/rescuenet/ann_dir/train'},
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='Resize', scale=(1024, 768), keep_ratio=True),
        dict(type='RandomCrop', crop_size=(512, 512)),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs'),
    ]
)

# Check the number of samples
print(f"Loaded dataset with {len(dataset)} samples.")

# Test loading a single sample
sample = dataset[0]
print("First sample:", sample)

