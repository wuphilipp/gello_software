import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ViTConv(torch.nn.Module):
    """
    Perform a sequence of convolutional operations and also merge with a pretrained ViT model.
    Note: Code operates on assumption of three layers being extracted from ViT.
    Different modes exist for fusing all three, the last two, or the last one layer.
    """

    def __init__(
        self,
        in_channels,
        shape,
        out_shape,
        pretrained_feat_info,
        version="default",
        conv_size=1,
        pretrained_feature_dim=64,
        channel_mask="default",
        pretrained_input_dims=[384, 384, 384],
        use_dense=True,
    ):
        super().__init__()
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []

        self.out_shape = out_shape
        fcs = [64, 64, 64]

        self.layerwise_feature = {val: key for key, val in pretrained_feat_info.items()}

        self.adapters = []
        self.linears = []
        self.pretrained_input_dims = pretrained_input_dims
        self.version = version
        self.pretrained_feature_dims = [pretrained_feature_dim] * len(fcs)
        self.conv_size = conv_size

        self.use_dense = use_dense
        assert version in [
            "default",
            "last_two",
            "last_only",
        ], f"Version {version} not supported (default, last_two, last_only)"

        self.feature_spatial_sizes = [28, 14, 7]
        for i, (feature_spatial_size, num_ch, pretrained_feature_dim) in enumerate(
            zip(self.feature_spatial_sizes, fcs, self.pretrained_feature_dims)
        ):
            if version == "last_two" and i < 1:
                continue
            elif version == "last_only" and i < 2:
                continue
            if conv_size == -1:
                conv_size = 55 // feature_spatial_size
            self.adapters.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.pretrained_input_dims[i],
                        pretrained_feature_dim,
                        kernel_size=conv_size,
                        stride=conv_size,
                        padding=0,
                    ),
                    nn.ReLU(),
                    nn.Upsample(
                        size=(feature_spatial_size, feature_spatial_size),
                        mode="bilinear",
                    ),
                )
            )
            if i != 2:
                self.linears.append(nn.Linear(num_ch + pretrained_feature_dim, num_ch))

        self.stem = nn.Conv2d(in_channels, fcs[0], kernel_size=4, stride=4)
        in_channels = fcs[0]
        out_shape = self.out_shape

        for layer, num_ch in enumerate(fcs):
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))
            in_channels = num_ch

            if self.version == "last_only" and layer != len(fcs) - 1:
                feat_dim = num_ch
            elif self.version == "last_two" and layer < len(fcs) - 2:
                feat_dim = num_ch
            else:
                feat_dim = num_ch + self.pretrained_feature_dims[layer]

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=feat_dim,
                        out_channels=feat_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=feat_dim,
                        out_channels=feat_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)
        self.adapters = nn.ModuleList(self.adapters)
        self.linears = nn.ModuleList(self.linears)
        self.channel_mask = channel_mask

    def forward_feature(self, feature, pretrained_feats):
        """
        Input: Feature: (B x F x V) x 4 x H x W
        Return: Feature of size (B x F x V) x D x pH (7) x pW (7)
        """
        assert feature.shape[1] == 4
        if self.channel_mask == "default":
            pass
        elif self.channel_mask == "rgb_only":
            feature[:, 3:] = 0
        elif self.channel_mask == "depth_only":
            feature[:, :3] = 0
        elif self.channel_mask == 'no_rgbd':
            feature[:, :] = 0

        x = self.stem(feature)
        res_input = None

        attn = []

        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            # Pretrained feature: (B x F x V) x (1 + pH x pW) x D
            if self.use_dense:
                pretrained_x = pretrained_feats[self.layerwise_feature[i]][
                    :, 1:
                ]  # Remove cls token
            else: # let's use the cls token only. make it match the shape.
                _, phpw, _ = pretrained_feats[self.layerwise_feature[i]].shape
                phpw -=1
                pretrained_x = pretrained_feats[self.layerwise_feature[i]][:, 0]
                pretrained_x = repeat(pretrained_x, 'b d -> b h d', h=phpw)

            def view_as_patches(feat):
                h = int(np.sqrt(feat.shape[-2]))
                return rearrange(feat, "b (h w) d -> b d h w", h=h, w=h)

            if self.version == "default":
                pretrained_x = view_as_patches(pretrained_x)

                for j in range(3):
                    pretrained_x = self.adapters[i][j](pretrained_x)
                    if j == 1:
                        # extract post RELU, pre downsample features
                        attn.append(pretrained_x.clone().detach())
                x = torch.cat([x, pretrained_x], dim=1)
            elif self.version == "last_only":
                if i == 2:
                    pretrained_x = view_as_patches(pretrained_x)
                    for j in range(3):
                        pretrained_x = self.adapters[0][j](pretrained_x)
                        if j == 1:
                            # extract post RELU, pre downsample features
                            attn.append(pretrained_x.clone().detach())
                    x = torch.cat([x, pretrained_x], dim=1)

            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input
            # Map back the dimension
            if i != 2 and self.version == "default":
                b, d, h, w = x.shape
                x = rearrange(x, "b d h w -> (b h w) d")
                x = self.linears[i](x)
                x = rearrange(x, "(b h w) d -> b d h w", b=b, h=h, w=w)
            elif i == 1 and self.version == "last_two":
                b, d, h, w = x.shape
                x = rearrange(x, "b d h w -> (b h w) d")
                x = self.linears[0](x)
                x = rearrange(x, "(b h w) d -> b d h w", b=b, h=h, w=w)

        attn = [F.interpolate(a, size=(28, 28), mode="bilinear") for a in attn]
        attn = torch.stack(attn, dim=1)
        attn = rearrange(attn, "b g d h w -> b g h w d")
        attn = torch.norm(attn, dim=-1)
        attn_b, attn_g, attn_h, attn_w = attn.shape
        attn = rearrange(attn, "b g h w -> (b g) (h w)")
        attn = torch.nn.functional.softmax(attn / 0.1, dim=-1)
        attn = rearrange(
            attn, "(b g) (h w) -> b g h w", b=attn_b, g=attn_g, h=attn_h, w=attn_w
        )

        x = rearrange(x, "b d h w -> b (h w d)")
        return x, attn