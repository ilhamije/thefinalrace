import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, build_activation_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule
from mmseg.registry import MODELS


class SparseShiftConv(nn.Module):
    """Sparse Shift Convolution where only selected channels are shifted."""

    def __init__(self, channels, sparsity=0.5, shift_size=1):
        super(SparseShiftConv, self).__init__()
        self.channels = channels
        self.sparsity = sparsity  # Fraction of channels that get shifted
        self.shift_size = shift_size
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.mask = torch.zeros(channels, dtype=torch.bool)
        selected = torch.randperm(channels)[:int(channels * self.sparsity)]
        self.mask[selected] = True  # Only a subset of channels will be shifted

    def forward(self, x):
        B, C, H, W = x.size()
        x_shift = x.clone()
        x_shift[:, self.mask, :, :] = torch.roll(
            x[:, self.mask, :, :], shifts=self.shift_size, dims=3)
        return self.pw_conv(x_shift)


class MSCABlock(BaseModule):
    """Multi-Scale Convolutional Attention Block with Sparse Shift Convolution."""

    def __init__(self, channels, mlp_ratio=4., drop=0., drop_path=0., act_cfg=dict(type='GELU'), norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        # Using SparseShiftConv instead of standard conv
        self.attn = SparseShiftConv(channels)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        # Ensure activation function is applied
        self.act = build_activation_layer(act_cfg)
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden_channels, kernel_size=1),
            self.act,
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_channels, channels, kernel_size=1),
            nn.Dropout(drop)
        )
        self.layer_scale_1 = nn.Parameter(
            torch.ones(channels) * 1e-2, requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            torch.ones(channels) * 1e-2, requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + \
            self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x



class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding with Sparse Shift Convolution."""

    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=768, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


@MODELS.register_module()
class MSCANSparseShift(BaseModule):
    """SegNeXt Multi-Scale Convolutional Attention Network (MSCAN) with Sparse Shift Convolution."""

    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., depths=[3, 4, 6, 3], num_stages=4, act_cfg=dict(type='GELU'), norm_cfg=dict(type='SyncBN', requires_grad=True), pretrained=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_channels=in_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
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
        return outs
