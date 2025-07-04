import numpy as np
import torch
import torch.nn as nn
import torch.nn.parameter as P
from einops import rearrange, repeat

import simple_bc.utils.torch_utils as utils
from simple_bc._interfaces.policy import Policy


class MLP(Policy):
    """
    Simple MLP.
    """

    def __init__(
        self,
        obs_shape,
        act_shape,
        act_bounds,
        layers,
        activation,
        normalization,
        zero_init_output,
        final_act,
        num_frames,
        num_views=2,
        action_horizon=1,
        encoder_cfg=None,
    ):
        super().__init__()
        self.obs_shape, self.act_shape = eval(str(obs_shape)), act_shape
        self.act_bounds = act_bounds
        self.num_frames = num_frames
        self.num_views = num_views
        if isinstance(self.obs_shape, tuple) or isinstance(
            self.obs_shape, list
        ):  # For ViT
            in_dim = int(self.obs_shape[-1]) * num_frames * num_views
        else:
            in_dim = int(self.obs_shape)

        out_dim = int(np.prod(self.act_shape)) * action_horizon
        self.action_horizon = action_horizon

        ### build trunk
        act = eval(activation)
        norm = eval(normalization)
        layers = [in_dim] + layers + [out_dim]
        trunk_layers = []
        for i in range(len(layers) - 2):
            trunk_layers += [
                nn.Linear(layers[i], layers[i + 1]),
            ]
            if act is not None:
                trunk_layers += [act()]
            if norm is not None:
                trunk_layers += [norm(layers[i + 1])]

        last_layer = nn.Linear(layers[-2], layers[-1])
        if zero_init_output:
            self._zero_init_output(last_layer)
        trunk_layers += [last_layer]
        if final_act:
            trunk_layers += [act()]

        self.trunk = nn.Sequential(*trunk_layers)

        ### build head
        if self.act_bounds is None:
            self.scale, self.bias = 1.0, 0.0
        else:
            lb = utils.to_torch(self.act_bounds[0])
            ub = utils.to_torch(self.act_bounds[1])
            self.scale = P.Parameter((ub - lb) / 2.0, requires_grad=False)
            self.bias = P.Parameter((ub + lb) / 2.0, requires_grad=False)

    def forward(self, obs):
        feat, enc_info = obs
        if "cls" in enc_info:
            cls, proprio = enc_info["cls"], enc_info["proprio"]
            B, F, V, C = enc_info["cls"].shape
            proprio = repeat(proprio, "b f c -> b f v c", v=V)
            feat = torch.cat([cls, proprio], dim=-1)
            feat = rearrange(feat, "b f v c -> b (f v c)")

        mean = self.trunk(feat)  # This includes MLP
        info = {}

        action = rearrange(mean, "b (t a) -> b t a", t=self.action_horizon)
        action = action * self.scale + self.bias

        if "attn" in enc_info:
            info["attn"] = enc_info["attn"]

        return action, info

    def _zero_init_output(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
