crop_size = (
    512,
    512,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=32),
    type='SegDataPreProcessor')
data_root = 'data/rescuenet/'
dataset_type = 'RescueNetDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    gpus=2,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
gpu_ids = [
    0,
    1,
]
ham_norm_cfg = dict(num_groups=32, requires_grad=True, type='GN')
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
img_scale = (
    1500,
    1125,
)
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        depths=(
            3,
            4,
            6,
            3,
        ),
        drop_path_rate=0.1,
        embed_dims=(
            64,
            128,
            320,
            512,
        ),
        in_channels=3,
        mlp_ratios=(
            4,
            4,
            4,
            4,
        ),
        type='MSCANSparseShift'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=32),
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            128,
            320,
            512,
        ],
        in_index=[
            1,
            2,
            3,
        ],
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(num_groups=32, requires_grad=True, type='GN'),
        num_classes=11,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=6e-05,
        momentum=0.9,
        type='AdamW',
        weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0), pos_block=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='AdamW', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=20000,
        eta_min=1e-05,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/test', seg_map_path='ann_dir/test'),
        data_root='data/rescuenet/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(pad_val=0, size=(
                512,
                512,
            ), type='Pad'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RescueNetDataset'),
    num_workers=2,
    persistent_workers=False,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    compute_loss=True, iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(pad_val=0, size=(
        512,
        512,
    ), type='Pad'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        data_root='data/rescuenet/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(
                keymap=dict(gt_semantic_seg='mask', img='image'),
                transforms=[
                    dict(
                        brightness=0.2,
                        contrast=0.2,
                        hue=0.1,
                        p=1.0,
                        saturation=0.2,
                        type='ColorJitter'),
                    dict(limit=10, p=0.3, type='Rotate'),
                    dict(blur_limit=(
                        3,
                        7,
                    ), p=0.3, type='GaussianBlur'),
                ],
                type='Albu',
                update_pad_shape=False),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackSegInputs'),
        ],
        type='RescueNetDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(
        keymap=dict(gt_semantic_seg='mask', img='image'),
        transforms=[
            dict(
                brightness=0.2,
                contrast=0.2,
                hue=0.1,
                p=1.0,
                saturation=0.2,
                type='ColorJitter'),
            dict(limit=10, p=0.3, type='Rotate'),
            dict(blur_limit=(
                3,
                7,
            ), p=0.3, type='GaussianBlur'),
        ],
        type='Albu',
        update_pad_shape=False),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        data_root='data/rescuenet/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(pad_val=0, size=(
                512,
                512,
            ), type='Pad'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='RescueNetDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    compute_loss=True, iou_metrics=[
        'mIoU',
        'mFscore',
    ], type='IoUMetric')
vis_backends = []
visualizer = dict(
    name='visualizer', type='SegLocalVisualizer', vis_backends=[])
work_dir = 'work_dirs/RTX3090_segnext_mscan-sparseshift-t_ablation2_exp'
