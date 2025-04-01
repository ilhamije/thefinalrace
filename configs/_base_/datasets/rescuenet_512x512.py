# dataset settings
dataset_type = 'RescueNetDataset'
data_root = 'data/rescuenet/'

# Adjusted crop size and image scale for memory optimization
crop_size = (512, 512)
img_scale = (1500, 1125)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    # Ensures all images are (1500, 1125)
    dict(type='Resize',
         scale=img_scale,
         keep_ratio=False),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
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
