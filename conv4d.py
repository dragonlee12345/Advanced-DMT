r""" Implementation of center-pivot 4D convolution """

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])
        dropout = 0.1
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels < 8:
            num_heads = in_channels
        else:
            num_heads = 8
        self.self_attn = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=num_heads, dropout=dropout
        )
        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, n, = ct.size()
        ct = rearrange(ct, 'bsz ch ha wa n -> bsz (ch ha wa) n')
        ct = F.interpolate(ct, scale_factor=1/self.stride[-1], mode='linear',
                                      align_corners=False)
        ct_pruned = rearrange(ct, 'bsz (ch ha wa) n -> bsz ch ha wa n', ch=ch, ha=ha, wa=wa,)
        return ct_pruned

    def forward(self, x):
        x = x.squeeze(-1).squeeze(-1)

        bsz, inch, ha, wa, n, = x.size()
        out1 = x.permute(0, 4, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, n, outch, o_ha, o_wa).permute(0, 2, 3, 4, 1).contiguous()


        if self.stride[2:][-1] > 1:
            out1 = self.prune(out1)
        bsz, inch, ha, wa, n = out1.size()
        out2 = out1.permute(0, 2, 3, 1, 4).contiguous().view(-1, inch, n).permute(2, 0, 1)
        out2_ = self.norm(out2)
        q = k = out2_
        out2_ = self.self_attn(q, k, out2_)[0]
        out2 = out2 + self.dropout(out2_)
        out2 = out2.permute(1, 2, 0)
        out2 = rearrange(out2, '(bsz ha wa) inch n -> bsz inch ha wa n', bsz=bsz, ha=ha, wa=wa,)

        return out2
