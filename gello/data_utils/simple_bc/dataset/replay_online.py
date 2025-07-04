import numpy as np
import torch
from einops import rearrange

import gdict
from simple_bc.utils.torch_utils import pack_one, to_cpu, unpack_one
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from einops import repeat

"""
This file contains the dataset for DAgger training.
"""


class ReplayBuffer(object):
    def __init__(
        self,
        aug_cfg,
        shuffle=True,
        obs_shapes=None,
        act_shape=None,
        action_horizon=1,
        MAX_LEN=200,
        use_rotation=True,  # Not used
        use_proprio=True,
    ):
        self.obs_shapes = obs_shapes
        self.act_shape = act_shape
        self.aug_cfg = aug_cfg
        self.shuffle = shuffle
        self.action_horizon = action_horizon

        # stacking
        self.stack_idx = aug_cfg.stack_idx
        self.stack_window = max(aug_cfg.stack_idx) + 1

        # Buffer
        self.all_trajs = []
        self.MAX_LEN = MAX_LEN
        self.num_trajs, self.num_timesteps = 0, 0

        # random shift augmentation, as described in DRQ v1 by Denis Yarats
        # and in "Revisiting LfS Baseline" by Hansen et al.
        if self.aug_cfg.aug_prob > 0:
            aug_pad_length = 5
            aug_crop_min = 0.7

            self.aug_transform = T.Compose(
                [
                    T.Pad(aug_pad_length, padding_mode="edge"),
                    T.RandomResizedCrop(
                        size=224,
                        scale=(aug_crop_min, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                    ),
                ]
            )

        else:
            self.aug_transform = None

    def _augment_frames(self, rgb_frames, depth_frames):
        """
        Augment the trajectory according to the augmentation config.
        This includes RGB augmentation and frame stacking.
        """
        p = np.random.rand()
        if p < self.aug_cfg.aug_prob and self.aug_transform is not None:
            rgb_frames, sh = pack_one(rgb_frames, "* c h w")
            depth_frames, _ = pack_one(depth_frames, "* c h w")
            depth_frames = repeat(depth_frames, "b c h w -> b (c d) h w", d=3)

            all_frames = torch.cat(
                [rgb_frames.float() / 255.0, depth_frames / torch.max(depth_frames)],
                axis=0,
            )
            all_frames = self.aug_transform(all_frames)
            B, _, _, _ = rgb_frames.shape
            aug_rgb_frames, aug_depth_frames = all_frames[:B], all_frames[B:]

            aug_rgb_frames *= 255.0
            aug_rgb_frames = unpack_one(aug_rgb_frames, sh, "* c h w")

            aug_depth_frames *= torch.max(depth_frames)
            aug_depth_frames = aug_depth_frames[:, 0:1]  # depth is 1-channel
            aug_depth_frames = unpack_one(aug_depth_frames, sh, "* c h w")

            return aug_rgb_frames, aug_depth_frames
        else:
            return rgb_frames, depth_frames

    def add_traj(self, obses, actions):
        """List (time) of obses and actions as input.
        Each action is B x A. Save into list of T x A.
        """
        actions = rearrange(torch.stack(actions, dim=0), "t b a -> b t a")
        rgb = rearrange(
            torch.stack([obs["rgb"] for obs in obses], dim=0), "t b ... -> b t ..."
        )
        depth = rearrange(
            torch.stack([obs["depth"] for obs in obses], dim=0), "t b ... -> b t ..."
        )
        state = rearrange(
            torch.stack([obs["state"] for obs in obses], dim=0), "t b ... -> b t ..."
        )
        actions, rgb, depth, state = (
            to_cpu(actions),
            to_cpu(rgb),
            to_cpu(depth),
            to_cpu(state),
        )
        num_traj = len(actions)
        for i in range(num_traj):
            traj = gdict.GDict(
                {
                    "actions": actions[i],
                    "obs": gdict.GDict(
                        {"state": state[i], "rgb": rgb[i], "depth": depth[i]}
                    ),
                }
            )
            self.all_trajs.append(traj)
            if len(self.all_trajs) > self.MAX_LEN:
                self.all_trajs = self.all_trajs[-self.MAX_LEN :]
        self.num_trajs += num_traj
        self.num_timesteps += num_traj * actions.shape[1]

    def sample(self, batch_size):
        traj_id = np.random.randint(len(self.all_trajs), size=batch_size)
        trajs = [self.all_trajs[i] for i in traj_id]

        Ts = (
            np.array([traj["actions"].shape[0] for traj in trajs])
            - self.stack_window
            + 1
        )
        ts = np.random.randint(Ts)
        ret_traj = [
            trajs[i].slice(np.arange(ts[i], ts[i] + self.stack_window))
            for i in range(batch_size)
        ]

        ret_traj = gdict.GDict.stack(ret_traj, axis=0)

        ret_traj["obs"]["rgb"], ret_traj["obs"]["depth"] = self._augment_frames(
            ret_traj["obs"]["rgb"], ret_traj["obs"]["depth"]
        )

        if self.action_horizon > 1:
            pass
        return ret_traj
