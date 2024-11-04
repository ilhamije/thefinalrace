import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class MSCANShift(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSCANShift, self).__init__()

        # Define multi-scale depth-wise convolutions (like the original MSCAN)
        self.conv_5x5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv_7x1 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            7, 1), padding=(3, 0), groups=in_channels)
        self.conv_1x11 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            1, 11), padding=(0, 5), groups=in_channels)
        self.conv_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Channel mixing (like in original MSCAN)
        self.channel_mixer = nn.Conv2d(
            in_channels * 4, out_channels, kernel_size=1)

        # Attention mechanism (similar to original MSCAN)
        self.attention = nn.Sigmoid()

    def forward(self, x):
        # Shift operations on input feature map
        shifted_left = self.shift(x, -1, 0)  # Shift left
        shifted_right = self.shift(x, 1, 0)  # Shift right
        shifted_up = self.shift(x, 0, -1)  # Shift up
        shifted_down = self.shift(x, 0, 1)  # Shift down

        # Apply depth-wise convolutions on shifted inputs
        out_5x5 = self.conv_5x5(shifted_left)
        out_7x1 = self.conv_7x1(shifted_right)
        out_1x11 = self.conv_1x11(shifted_up)
        out_1x1 = self.conv_1x1(shifted_down)

        # Concatenate multi-scale features
        multi_scale_features = torch.cat(
            [out_5x5, out_7x1, out_1x11, out_1x1], dim=1)

        # Channel mixing
        mixed_features = self.channel_mixer(multi_scale_features)

        # Apply attention
        attention_weights = self.attention(mixed_features)
        output = attention_weights * mixed_features  # Element-wise multiplication

        return output

    @staticmethod
    def shift(x, dx, dy):
        """Shift tensor x by (dx, dy) pixels."""
        B, C, H, W = x.size()
        # Pad input to handle shifts
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        # Compute slicing indices after shift
        x_start = pad_right
        x_end = x_start + W
        y_start = pad_bottom
        y_end = y_start + H

        # Extract the shifted tensor
        x_shifted = x_padded[:, :, y_start:y_end, x_start:x_end]
        return x_shifted


# Example usage of MSCAN with shift convolution
if __name__ == "__main__":
    x = torch.randn(8, 64, 32, 32)  # Example input tensor
    mscan = MSCANShift(64, 128)
    y = mscan(x)
    print(f"Output shape: {y.shape}")  # Expected: [8, 128, 32, 32]

