_base_ = ['./RTX3090_segformer_mit-b0_8xb2-160k_rescuenet-755x769.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa

# model settings
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
