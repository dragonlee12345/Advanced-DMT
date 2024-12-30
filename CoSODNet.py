import math
import torch
import torch.nn as nn
from torch.nn import functional as F

#from detectron2.layers import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

from CoFormer_Decoder import CoFormer_Decoder

from B2_VGG import B2_VGG

from GroupAttention import GroupAttention


class _DASPPConvBranch(nn.Module):
    """
    ConvNet block for building DenseASPP.
    """

    def __init__(self, in_channel, out_channel, inter_channel=None, dilation_rate=1, norm='BN'):
        super().__init__()

        if not inter_channel:
            inter_channel = in_channel // 2

        use_bias = norm == ""
        # self.conv1 = Conv2d(
        #     in_channel,
        #     inter_channel,
        #     kernel_size=1,
        #     bias=use_bias,
        #     norm=get_norm(norm, inter_channel),
        #     activation=F.relu,
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=inter_channel, kernel_size=1, bias=use_bias),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )
        # weight_init.c2_xavier_fill(self.conv1)

        # self.conv2 = Conv2d(
        #     inter_channel,
        #     out_channel,
        #     kernel_size=3,
        #     stride=1,
        #     dilation=dilation_rate,
        #     padding=dilation_rate,
        #     bias=use_bias,
        #     norm=get_norm(norm, out_channel),
        #     activation=F.relu,
        # )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=inter_channel, out_channels=out_channel, kernel_size=3, stride=1,
                      dilation=dilation_rate, padding=dilation_rate, bias=use_bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        # weight_init.c2_xavier_fill(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DASPPBlock(nn.Module):
    def __init__(self, cfg):
        super(DASPPBlock, self).__init__()

        enc_last_channel = cfg.MODEL.ENCODER.CHANNEL[-1]
        adap_channel = cfg.MODEL.DASPP.ADAP_CHANNEL
        # self.adap_layer = Conv2d(
        #     enc_last_channel,
        #     adap_channel,
        #     kernel_size=1,
        #     bias=False,
        #     norm=get_norm('BN', adap_channel),
        #     activation=F.relu,
        # )
        self.adap_layer = nn.Sequential(
            nn.Conv2d(in_channels=enc_last_channel, out_channels=adap_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(adap_channel),
            nn.ReLU()
        )
        # weight_init.c2_xavier_fill(self.adap_layer)

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

        # self.after_daspp = Conv2d(
        #     adap_channel*2 + dil_branch_ch*self.convlayers,
        #     adap_channel,
        #     kernel_size=1,
        #     bias=False,
        #     norm=get_norm("BN", adap_channel),
        #     activation=F.relu,
        # )
        self.after_daspp = nn.Sequential(
            nn.Conv2d(in_channels=adap_channel*2 + dil_branch_ch*self.convlayers,
                      out_channels=adap_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(adap_channel),
            nn.ReLU()
        )
        # weight_init.c2_xavier_fill(self.after_daspp)

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


class CoSODNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super().__init__()

        self.mode = mode
        self.last_fea_name = cfg.MODEL.ENCODER.NAME[-1]
        self.gr_fea_name = cfg.MODEL.GROUP_ATTENTION.NAME[0]

        self.encoder = B2_VGG()

        self.daspp_block = DASPPBlock(cfg)

        self.group_att = GroupAttention(cfg, 512)

        self.cosod_former = CoFormer_Decoder(cfg)

    def forward(self, input):
        N, _, _, _ = input.size()

        enc_feas = self.encoder(input)  # 所有feature, dict格式
        last_fea = enc_feas[self.last_fea_name]  # 最高层feature
        daspp_fea = self.daspp_block(last_fea)  # 空洞卷积提取更多feature
        gr_fea = self.group_att(daspp_fea)  # 先一步R2R
        enc_feas.update({self.gr_fea_name: gr_fea})

        result = self.cosod_former(enc_feas)

        return result

