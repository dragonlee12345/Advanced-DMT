
import torch.nn as nn
import torch.nn.functional as F

from conv4d import CenterPivotConv4d as Conv4d
from icecream import ic

class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4
                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)
        outch1, outch2, outch3 = 64, 128, 256

        # Squeezing building blocks
        self.enc_layer = make_building_block(inch, [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])

    def forward(self, hypercorr_pyramid):
        hypercorr_sqz = self.enc_layer(hypercorr_pyramid)
        hypercorr_encoded = hypercorr_sqz.mean(dim=-1)
        return hypercorr_encoded
