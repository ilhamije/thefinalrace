# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmengine.model.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)

from mmseg.registry import MODELS


class GroupedShiftConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        shift_offsets=None,
        use_1x1_after_shift=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_1x1 = use_1x1_after_shift

        # Default to 3x3 shift if none provided
        if shift_offsets is None:
            shift_offsets = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, -1), (1, -1), (-1, 1)
            ]
        self.shift_offsets = shift_offsets
        self.num_groups = len(shift_offsets)
        assert in_channels % self.num_groups == 0, \
            f"in_channels={in_channels} must be divisible by groups={self.num_groups}"
        self.channels_per_group = in_channels // self.num_groups

        # Optional post-shift 1x1 conv
        if self.use_1x1:
            self.pointwise = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.norm = build_norm_layer(norm_cfg, in_channels)[1]
            self.act = build_activation_layer(act_cfg)
        else:
            self.pointwise = None

    def forward(self, x):
        B, C, H, W = x.shape
        x_split = torch.chunk(x, self.num_groups, dim=1)

        out = []
        for xi, (dy, dx) in zip(x_split, self.shift_offsets):
            shifted = torch.roll(xi, shifts=(dy, dx), dims=(2, 3))
            out.append(shifted)

        x_shifted = torch.cat(out, dim=1)

        if self.use_1x1:
            x_shifted = self.pointwise(x_shifted)
            x_shifted = self.norm(x_shifted)
            x_shifted = self.act(x_shifted)

        return x_shifted



class Mlp(BaseModule):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)  # Fixed layer name
        attn_0 = self.conv0_2(attn_0)  # Fixed layer name

        attn_1 = self.conv1_1(attn)  # Fixed layer name
        attn_1 = self.conv1_2(attn_1)  # Fixed layer name

        attn_2 = self.conv2_1(attn)  # Fixed layer name
        attn_2 = self.conv2_2(attn_2)  # Fixed layer name

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        x = attn * u
        return x


class MSCASpatialAttention(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module
    (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


# Modify MSCABlock to integrate the GroupedShiftConv2d
class MSCABlock(BaseModule):
    def __init__(self,
                 channels,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 num_groups=4,
                 kernel_size=3,
                 dilation=1,
                 use_1x1_after_shift=True,
                 shift_offsets=None):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]

        # Store parameters for transparency/debug
        self.num_groups = num_groups
        self.kernel_size = kernel_size
        self.dilation = dilation

        # Replace original Spatial Attention with GroupedShiftConv2d
        self.attn = GroupedShiftConv2d(
            in_channels=channels,
            shift_offsets=shift_offsets,
            use_1x1_after_shift=use_1x1_after_shift,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)

        # GSC replacing original attention
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))

        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x




class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        patch_size (int): The patch size.
            Defaults: 7.
        stride (int): Stride of the convolutional layer.
            Default: 4.
        in_channels (int): The number of input channels.
            Defaults: 3.
        embed_dims (int): The dimensions of embedding.
            Defaults: 768.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@MODELS.register_module()
class MSCANGroupShift(BaseModule):
    """SegNeXt Multi-Scale Convolutional Attention Network (MCSAN) backbone.

    This backbone is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Args:
        in_channels (int): The number of input channels. Defaults: 3.
        embed_dims (list[int]): Embedding dimension.
            Defaults: [64, 128, 256, 512].
        mlp_ratios (list[int]): Ratio of mlp hidden dim to embedding dim.
            Defaults: [4, 4, 4, 4].
        drop_rate (float): Dropout rate. Defaults: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.
        depths (list[int]): Depths of each Swin Transformer stage.
            Default: [3, 4, 6, 3].
        num_stages (int): MSCAN stages. Default: 4.
        attention_kernel_sizes (list): Size of attention kernel in
            Attention Module (Figure 2(b) of original paper).
            Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): Size of attention paddings
            in Attention Module (Figure 2(b) of original paper).
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        norm_cfg (dict): Config of norm layers.
            Defaults: dict(type='SyncBN', requires_grad=True).
        pretrained (str, optional): model pretrained path.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 num_groups=4,
                 kernel_size=3,
                 dilation=1,
                 init_cfg=None,
                 use_1x1_after_shift=True,
                 shift_offsets=None):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = StemConv(in_channels if i == 0 else embed_dims[i - 1], embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    num_groups=num_groups,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    use_1x1_after_shift=use_1x1_after_shift,
                    shift_offsets=shift_offsets
                ) for j in range(depths[i])
            ])

            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        # Debugging logs
        # print(f"Stage Outputs: {[out.shape for out in outs]}")

        return outs
