_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
    '../_base_/datasets/rescuenet_512x512.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segnext/mscan_t_20230227-119e8c9f.pth'

crop_size = (512, 512)
ham_norm_cfg = dict(type='SyncBN', requires_grad=True,
                    eps=1e-5)  # ✅ patch norm cfg here

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32)
)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MSCANSparseShift',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        use_1x1_after_shift=True,  # ✅
        act_cfg=dict(type='GELU'),  # ✅
        norm_cfg=dict(type='SyncBN', requires_grad=True,
                      eps=1e-5)  # ✅ force all eps=1e-5
    ),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True),
        ham_kwargs=dict(
            MD_S=1,
            MD_R=16,
            train_steps=6,
            eval_steps=7,
            inv_t=100,
            rand_init=True)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

train_dataloader = dict(batch_size=8, num_workers=4, pin_memory=True)
val_dataloader = dict(batch_size=4, num_workers=2)
test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    # loss_scale=64.0,  # ✅ reduced from 128.0
    optimizer=dict(
        type='AdamW', lr=0.00018, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=0.5, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        })
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=5000),  # keep warmup long to stabilize early shifts
    dict(
        type='PolyLR',
        power=1.0,
        begin=5000,
        end=20000,
        eta_min=0.0,
        by_epoch=False,
    )
]


# ABLATION STUDY
# We will conduct an ablation study to compare different variants of the model.
# Variant	SparseShift	1×1 Conv	Activation	Norm	Augment	Notes
# A	✅	✅	GELU	SyncBN	ColorJitter	Baseline
# B	✅	✅	GELU	SyncBN	❌ No Augment
# C	✅	✅	ReLU	SyncBN	ColorJitter
# D	✅	✅	ReLU	SyncBN	❌ No Augment
# E	✅	❌	GELU	SyncBN	ColorJitter	No 1×1 conv
# F	✅	❌	ReLU	SyncBN	❌ No Augment	Minimalist
# G	✅	✅	GELU	GroupNorm	ColorJitter	Switch norm
# H	✅	✅	ReLU	GroupNorm	❌ No Augment	GN + ReLU
# Let’s start with Variant A again — but with:

# ✅ Clamped shift ops

# ✅ Lower loss_scale = 128.0

# ✅ New config structure(cleaned up)
