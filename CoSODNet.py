import torch.nn as nn

from models.B2_VGG import B2_VGG

from models.DASPP import DASPPBlock


class CoSODNet(nn.Module):
    def __init__(self, args, cfg, mode='train'):
        super().__init__()

        # model = cfg.model

        if args.model == 'DMT+':
            from models.DMT_plus.GroupAttention import GroupAttention
            from models.DMT_plus.CoFormer_Decoder import CoFormer_Decoder
        else:
            from models.DMT_plus_O.GroupAttention import GroupAttention
            from models.DMT_plus_O.CoFormer_Decoder import CoFormer_Decoder

        self.mode = mode
        self.last_fea_name = cfg.MODEL.ENCODER.NAME[-1]
        self.gr_fea_name = cfg.MODEL.GROUP_ATTENTION.NAME[0]

        self.encoder = B2_VGG()

        self.daspp_block = DASPPBlock(cfg)
        self.group_att = GroupAttention(cfg, 512)
        self.cosod_former = CoFormer_Decoder(cfg)

    def forward(self, input):
        N, _, _, _ = input.size()

        enc_feas = self.encoder(input)
        last_fea = enc_feas[self.last_fea_name]
        daspp_fea = self.daspp_block(last_fea)
        # gr_fea = self.group_att(daspp_fea)
        enc_feas.update({self.gr_fea_name: daspp_fea})

        result = self.cosod_former(enc_feas)

        return result

