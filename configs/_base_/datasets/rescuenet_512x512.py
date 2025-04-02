# dataset settings
dataset_type = 'RescueNetDataset'
data_root = 'data/rescuenet/'

# Adjusted crop size and image scale for memory optimization
crop_size = (512, 512)
img_scale = (1500, 1125)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(
        type='Albu',
        transforms=[
            dict(type='ColorJitter', brightness=0.2,
                 contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            # ✅ Albumentations uses `Rotate` not `RandomRotate`
            dict(type='Rotate', limit=10, p=0.3),
            # ✅ Albumentations expects `blur_limit`, not `sigma_min/max`
            dict(type='GaussianBlur', blur_limit=(3, 7), p=0.3)
        ],
        keymap={
            'img': 'image',
            'gt_semantic_seg': 'mask'
        },
        update_pad_shape=False,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs'),
]

# dict(type='PhotoMetricDistortion'),  ❌ consider removing this # You can keep PhotoMetricDistortion or replace it — but NOT both ColorJitter + PhotometricDistortion

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='Pad', size=(512, 512), pad_val=0),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ]
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))
# test_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/test',
            seg_map_path='ann_dir/test'),
        pipeline=test_pipeline))

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],
    compute_loss=True  # this will log val_loss automatically!
)
test_evaluator = val_evaluator
