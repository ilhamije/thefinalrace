_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py',
    '../_base_/datasets/rescuenet_512x512.py'
]
# model settings
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512),
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCANSparseShift',  # ✅ your new backbone
        in_channels=3,
        embed_dims=[64, 128, 320, 512],  # ✅ matches backbone output
        num_blocks=[3, 3, 12, 3],
        shift_groups=8,  # use a value that divides all embed_dims
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 320, 512],  # ✅ MUST match your embed_dims
        in_index=[1, 2, 3, 4],
        channels=256,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True),
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
train_dataloader = dict(batch_size=8, num_workers=8, pin_memory=True)
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = val_dataloader


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),  # ✅ AdamW uses betas instead of momentum
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]
