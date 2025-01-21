import math
import torch
import torch.nn as nn


class _DASPPConvBranch(nn.Module):
    """
    ConvNet block for building DenseASPP.
    """

    def __init__(self, in_channel, out_channel, inter_channel=None, dilation_rate=1, norm='BN'):
        super().__init__()

        if not inter_channel:
            inter_channel = in_channel // 2

        use_bias = norm == ""
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=inter_channel, kernel_size=1, bias=use_bias),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inter_channel, out_channels=out_channel, kernel_size=3, stride=1,
                      dilation=dilation_rate, padding=dilation_rate, bias=use_bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DASPPBlock(nn.Module):
    def __init__(self, cfg):
        super(DASPPBlock, self).__init__()

        enc_last_channel = cfg.MODEL.ENCODER.CHANNEL[-1]
        adap_channel = cfg.MODEL.DASPP.ADAP_CHANNEL

        self.adap_layer = nn.Sequential(
            nn.Conv2d(in_channels=enc_last_channel, out_channels=adap_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(adap_channel),
            nn.ReLU()
        )

        dilations = cfg.MODEL.DASPP.DILATIONS
        self.convlayers = len(dilations)

        # must be divisible by 32 because of the group norm
        dil_branch_ch = math.ceil(adap_channel/self.convlayers/32)*32

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.DASPP_Conv_Branches = []
        for idx, dilation in enumerate(dilations):
            this_conv_branch = _DASPPConvBranch(
                adap_channel + idx * dil_branch_ch,
                dil_branch_ch,
                inter_channel=adap_channel // 2,
                dilation_rate=dilation,
                norm="BN",
            )
            self.add_module("conv_brach_{}".format(idx + 1), this_conv_branch)
            self.DASPP_Conv_Branches.append(this_conv_branch)

        self.after_daspp = nn.Sequential(
            nn.Conv2d(in_channels=adap_channel*2 + dil_branch_ch*self.convlayers,
                      out_channels=adap_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(adap_channel),
            nn.ReLU()
        )

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, fea):
        fea = self.adap_layer(fea)

        global_pool_fea = self.global_pool(fea).expand_as(fea)

        out_fea = fea
        for idx, layer in enumerate(self.DASPP_Conv_Branches):
            dil_conv_fea = layer(out_fea)
            out_fea = torch.cat([dil_conv_fea, out_fea], dim=1)

        daspp_fea = torch.cat([global_pool_fea, out_fea], dim=1)
        after_daspp_fea = self.after_daspp(daspp_fea)

        return after_daspp_fea
