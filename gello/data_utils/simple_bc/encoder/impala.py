import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from simple_bc._interfaces.encoder import Encoder
from simple_bc.utils.torch_utils import to_torch, to_numpy

import hydra
from omegaconf import DictConfig


class IMPALA(Encoder):
    def __init__(self,
                 in_channels,
                 shape,
                 use_depth=True,
                 large=True,
                 larger=True,
                 num_views=2,
                 **kwargs):
        super().__init__(**kwargs)
        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []
        self.convs = []

        self.use_depth = use_depth
        self.num_views = num_views

        H, W = shape
        if larger:
            fcs = [128, 128, 128]
        else:
            fcs = [64, 64, 64]
        self.shape = [H, W]

        self.large = large
        if self.large:
            in_channels = 4
            print("IMPALA: using large network, so using 4 channels, and not convolving over time and views.")
        self.stem = nn.Conv2d(in_channels, fcs[0], kernel_size=4, stride=4)
        in_channels = fcs[0]

        for num_ch in fcs:
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
            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=num_ch,
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
        self.img_feat_size = (H * W) // (4 ** len(fcs) * 16) * fcs[-1]

        if self.large:
            self.fc = nn.Identity()
            self.out_shape = self.img_feat_size * self.num_views * self.num_frames
        else:
            self.fc = nn.Linear(self.img_feat_size, self.out_shape)

        self._update_out_shape(self.out_shape)

    def preprocess(self, obs):
        with torch.no_grad():
            feature = []
            if "rgb" in self.obs_shapes and "rgb" in obs:
                B = len(obs["rgb"])
                assert (
                    obs["rgb"].shape[-3:] == self.obs_shapes["rgb"]
                ), f"Observation shape of rgb is {obs['rgb'].shape}, but should be {(B, *self.obs_shapes['rgb'])}"
                # B, F (Frames), V (views), C, H, W
                if self.large:
                    rgb = rearrange(obs["rgb"], "b f v c h w -> (b f v) c h w")
                else:
                    rgb = rearrange(obs["rgb"], "b f v c h w -> b (f v c) h w")
                feature.append((rgb / 255.0))

            if "depth" in self.obs_shapes and "depth" in obs:
                B = len(obs["depth"])
                assert (
                    obs["depth"].shape[-3:] == self.obs_shapes["depth"]
                ), f"Observation shape of depth is {obs['depth'].shape}, but should be {(B, *self.obs_shapes['depth'])}"

                if self.large:
                    depth = rearrange(obs["depth"], "b f v c h w -> (b f v) c h w")
                else:
                    depth = rearrange(obs["depth"], "b f v c h w -> b (f v c) h w")

                if not self.use_depth:
                    depth = torch.zeros_like(depth)

                feature.append(depth)

            feature = torch.cat(feature, dim=1)
            if "state" in obs:
                state = obs["state"]
            else:
                state = None

        return feature, state

    def forward(self, obs):
        """Return feature and info."""
        feature, state = self.preprocess(obs)
        x = self.stem(feature)
        res_input = None

        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.reshape(x.shape[0], self.img_feat_size)
        x = F.relu(self.fc(x))

        if self.large:
            x = rearrange(x, "(b f v) c -> b (f v c)", f=self.num_frames, v=self.num_views)

        B = x.shape[0]
        state = state.view(B, -1)

        out = torch.cat([x, state], dim=1)
        return out, {}

    def _update_out_shape(self, out_shape):
        if "state" in self.obs_shapes:
            state_shape = np.prod(self.obs_shapes["state"]) * self.num_frames
            print(f"IMPALA: updated out shape to {self.out_shape + state_shape}")
            self.out_shape = out_shape + state_shape
        else:
            self.out_shape = out_shape


@hydra.main(config_path="../../conf/encoder", config_name="impala", version_base="1.1")
def test(cfg):
    cfg = DictConfig(cfg)
    cfg.in_channels = 32
    cfg.shape = [224, 224]
    cfg.obs_shapes = {"rgb": [3, 224, 224], "depth": [1, 224, 224], "state": [26]}
    cfg.num_frames = 4
    cfg.out_shape = 384
    print(cfg)

    encoder = IMPALA(**cfg)
    print(encoder)

    obs = {
        "rgb": to_torch(torch.randn(6, 4, 2, 3, 224, 224)),
        "depth": to_torch(torch.randn(6, 4, 2, 1, 224, 224)),
        "state": to_torch(torch.randn(6, 4, 26)),
    }

    out, _ = encoder(obs)
    assert (
        out.shape[1:] == encoder.out_shape
    ), f"out shape is {out.shape}, but should be {encoder.out_shape}"


if __name__ == "__main__":
    test()
