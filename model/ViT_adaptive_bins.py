import torch
import torch.nn as nn
import torch.nn.functional as F
from miniViT import mViT
from layers import (
    FeatureFusionBlock_custom,
    _make_encoder,
    forward_vit,
)


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DPT(nn.Module):
    def __init__(
            self,
            head,
            features=256,
            backbone="vitb_rn50_384",
            readout="project",
            channels_last=True,
            use_bn=False,
            enable_attention_hooks=False,
    ):
        super(DPT, self).__init__()
        self.channels_last = channels_last

        hooks = {
            "vitb_rn50_384": [0, 1, 8, 11],
            "vitb16_384": [2, 5, 8, 11],
            "vitl16_384": [5, 11, 17, 23],
        }

        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
            enable_attention_hooks=enable_attention_hooks,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head

    def forward(self, x):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class ViTAdaBins(DPT):
    def __init__(self, n_bins=256, min_val=10, max_val=1000, norm='linear', **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        head = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Identity(),
        )
        super(ViTAdaBins, self).__init__(head, **kwargs)
        self.adaptive_bins_layer = mViT(128, n_query_channels=128, patch_size=16,
                         dim_out=n_bins, embedding_dim=128, norm=norm)

        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
            )

    def forward(self, x):
        inv_depth = super().forward(x)
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(inv_depth)
        out = self.conv_out(range_attention_maps)
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        pred = torch.sum(out * centers, dim=1, keepdim=True)
        pred[pred < 1e-8] = 1e-8
        return bin_edges, pred

    @classmethod
    def build(cls, n_bins, **kwargs):
        model = cls(n_bins=n_bins, **kwargs)
        return model


if __name__ == '__main__':
    device = torch.device("cuda")
    with torch.no_grad():
        model = ViTAdaBins.build(256, min_val=1e-3, max_val=10).to(device)
        img = torch.rand(1, 3, 480, 640).cuda()
        bins, pred = model(img)
        print(pred.shape)
