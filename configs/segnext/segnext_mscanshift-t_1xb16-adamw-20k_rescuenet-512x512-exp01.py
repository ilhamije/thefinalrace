_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py',
    '../_base_/datasets/rescuenet.py'
]
# model settings
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'  # noqa
ham_norm_cfg = dict(type='GN', num_groups=16, requires_grad=True)
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
    backbone=dict(
        type='MSCANShift',
        embed_dims=[32, 64, 160, 256],
        depths=[3, 3, 5, 2],
        mlp_ratios=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        num_stages=4
    ),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=11,
        align_corners=False
    )
)

train_dataloader = dict(batch_size=4, num_workers=4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5),
    accumulative_iter=2  # Gradient accumulation to simulate batch size 8
)


# dataset settings
# train_dataloader = dict(batch_size=16, num_workers=0)
train_dataloader = dict(batch_size=4, num_workers=1, pin_memory=True)

# Optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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
