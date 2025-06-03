import r3m
import os
from gpu_info import vulkan_cuda_idxes

from simple_bc._interfaces.encoder import Encoder
from einops import rearrange, repeat
from r3m.models.models_r3m import R3M

import os
from os.path import expanduser
import omegaconf
import hydra
import gdown
import torch


def load_r3m(modelid):
    # copied from Suraj Nair's R3M repo, repurposed for different HYDRA launchers
    home = os.path.join(expanduser("~"), ".r3m")
    if "HYDRA_LAUNCHER" not in os.environ:
        r3m.device = 0
    else:
        cuda_gpu_idxes, _ = vulkan_cuda_idxes(os.environ["HYDRA_LAUNCHER"], 1)
        r3m.device = cuda_gpu_idxes[0]
    if modelid == "resnet50":
        foldername = "r3m_50"
        modelurl = "https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA"
        configurl = "https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8"
    elif modelid == "resnet34":
        foldername = "r3m_34"
        modelurl = "https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE"
        configurl = "https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW"
    elif modelid == "resnet18":
        foldername = "r3m_18"
        modelurl = "https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-"
        configurl = "https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6"
    else:
        raise NameError("Invalid Model ID")

    if not os.path.exists(os.path.join(home, foldername)):
        os.makedirs(os.path.join(home, foldername))
    modelpath = os.path.join(home, foldername, "model.pt")
    configpath = os.path.join(home, foldername, "config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = r3m.cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep, device_ids=[r3m.device])
    r3m_state_dict = r3m.remove_language_head(
        torch.load(modelpath, map_location=torch.device(r3m.device))["r3m"]
    )
    rep.load_state_dict(r3m_state_dict)
    return rep


class R3MEncoder(Encoder):
    def __init__(self, model_type="resnet50", freeze_pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.model = load_r3m(model_type)

        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        self.out_shape = eval(str(self.out_shape))

        self.batch_norm = torch.nn.BatchNorm1d(
            self.model.module.outdim
        )  # as described in R3M paper, this occurs prior to MLP layers

    def forward(self, obs):
        # obs is of shape b f v c h w, (state is shape b f d)
        img = obs["rgb"]

        B, F, V, _, _, _ = obs["rgb"].shape
        img = rearrange(img, "b f v c h w -> (b f v) c h w")

        feats = self.model(img)
        feats = self.batch_norm(feats)
        feats = rearrange(feats, "(b f v) d -> b (f v d)", b=B, f=F, v=V)

        if "state" in obs:
            state = obs["state"]
            state = repeat(state, "b f d -> b f v d", v=V)
            state = rearrange(state, "b f v d -> b (f v d)")
            feats = torch.cat([feats, state], dim=1)

        return feats, {}

    def preprocess(self, obs):  # for encoder interface
        return obs
