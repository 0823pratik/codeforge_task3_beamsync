# ultralytics/nn/modules/attention.py  (CORRECTED VERSION)

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, c1, ratio=16):
        super().__init__()
        if c1 == -1:  # Handle Ultralytics parse_model()
            self.avg_pool = self.max_pool = self.mlp = self.act = nn.Identity()
            return
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        c_ = max(c1 // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        if hasattr(self, 'avg_pool') and self.avg_pool is not nn.Identity():
            avg = self.mlp(self.avg_pool(x))
            mx = self.mlp(self.max_pool(x))
            return self.act(avg + mx) * x
        return x  # identity if c1 was -1 during init

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        a = self.cv1(a)
        return self.act(a) * x

class CBAM(nn.Module):
    def __init__(self, c1, kernel_size=7, ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
