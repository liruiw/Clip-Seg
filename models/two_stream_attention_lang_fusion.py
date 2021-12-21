import torch.nn as nn
from models.resnet_lat import ResNet45_10s
from models.clip_lingunet_lat import CLIPLingUNetLat
from models.fusion import FusionAdd


class TwoStreamAttentionLangFusionLat(nn.Module):
    def __init__(self):
        super(TwoStreamAttentionLangFusionLat, self).__init__()
        self.fusion_type = FusionAdd
        self.batchnorm = True

        self._build_nets()

    def _build_nets(self):
        # ('plain_resnet_lat', 'clip_lingunet_lat')
        self.attn_stream_one = ResNet45_10s(1, self.batchnorm)
        self.attn_stream_two = CLIPLingUNetLat(1, self.batchnorm)
        self.fusion = FusionAdd(input_dim=1)
        self.params_exclude_clip = list(self.attn_stream_one.parameters()) + \
            self.attn_stream_two.params_exclude_clip + list(self.fusion.parameters())

    def forward(self, imgs, tokens):
        x1, lat = self.attn_stream_one(imgs)
        x2 = self.attn_stream_two(imgs[:, :3], lat, tokens)
        logits = self.fusion(x1, x2)

        return logits

