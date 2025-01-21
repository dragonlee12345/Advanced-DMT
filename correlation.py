r""" Provides functions that builds/manipulates correlation tensors """
import torch
from einops import rearrange

import torch.nn.functional as F


class Correlation:
    @classmethod
    def multilayer_correlation(cls, query_feat, support_feats,):
        eps = 1e-5

        corrs = []
        for idx, support_feat in enumerate(support_feats):
            # if support_feat.shape[2] > 8:
            #     support_feat = F.interpolate(support_feat, size=(8, 8), mode="bilinear", align_corners=True)
            bsz, ch, hb, wb = support_feat.size()
            support_feat = rearrange(support_feat, "b c h w -> b h w c").contiguous()
            support_feat = support_feat.view(-1, ch)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = query_feat.size()
            query_feat_ = rearrange(query_feat, "b c h w -> b h w c").contiguous()
            query_feat_ = query_feat_.view(-1, ch)
            query_feat_ = query_feat_ / (query_feat_.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(query_feat_.unsqueeze(0), support_feat.transpose(0, 1).unsqueeze(0)).squeeze()
            corr = corr.view(bsz, ha, wa, bsz, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        fina_corr = torch.stack(corrs).transpose(0, 1).contiguous()
        return fina_corr
