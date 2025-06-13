import copy
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import IterableDataset
from torchvision.transforms.functional import InterpolationMode

from simple_bc.utils.data_utils import load_traj_from_memory, load_traj_files
from simple_bc.utils.torch_utils import pack_one, unpack_one

"""
This file contains the dataset for BC training.
"""


class CachedTrajLoader(object):
    def __init__(
        self,
        shuffle,
        stack_idx,
        stack_window,
        all_traj_cache=dict(),  # note that this is shared between all CachedTrajLoader instances
    ):
        """
        shuffle: If False, the trajectories are loaded in order. If True, the trajectories are shuffled.
        """
        self.shuffle = shuffle
        self.stack_idx = np.array(stack_idx)
        self.stack_window = stack_window
        self.num_cached_traj = 4
        self.all_traj_cache = all_traj_cache

        self.worker_filenames, self.unload_filenames = None, None
        self.cached_trajs, self.cached_trajs_starts, self.cached_trajs_shuffled_idx = (
            None,
            None,
            None,
        )

        MAX_LEN = 2000
        ts = np.arange(MAX_LEN)
        self.padded_ts = np.concatenate(
            [np.zeros(self.stack_window, dtype=np.int32), ts]
        )

    def reset(self, worker_filenames):
        "Reset the buffer"
        self.worker_filenames = copy.copy(worker_filenames)
        self.unload_filenames = copy.copy(worker_filenames)
        del self.cached_trajs, self.cached_trajs_starts, self.cached_trajs_shuffled_idx
        self.cached_trajs = []
        self.cached_trajs_starts = (
            []
        )  # The starting index of each trajectory which has not been sampled
        self.cached_trajs_shuffled_idx = []

    def _load_to_cache(self):
        if len(self.unload_filenames) == 0:
            return -1  # No more trajectory to load

        while (
            len(self.cached_trajs) < self.num_cached_traj
            and len(self.unload_filenames) > 0
        ):
            traj_filename = self.unload_filenames.pop()
            if traj_filename in self.all_traj_cache:
                traj = self.all_traj_cache[traj_filename]
            else:
                traj = load_traj_from_memory(traj_filename)
            self.cached_trajs.append(traj)
            self.cached_trajs_starts.append(0)

            T = len(traj)
            ts = np.arange(
                T - self.stack_window - 1 + len(self.stack_idx), dtype=np.int32
            )
            if self.shuffle:
                np.random.shuffle(ts)
            self.cached_trajs_shuffled_idx.append(
                ts
            )  # The oldest time step of frame stacking

        return 0

    def sample(self):
        """
        Randomly sample one time step from one trajectory. No repeat
        Return (traj, ts) the corresponding trajectory and the sampled time steps (For frame stacking)
        Return (None, None) if no trajectory is available
        """
        if len(self.cached_trajs) == 0:
            ret = self._load_to_cache()
            if ret == -1:
                return None, None

        traj_idx = np.random.randint(len(self.cached_trajs)) if self.shuffle else 0
        traj = self.cached_trajs[traj_idx]
        start_t = self.cached_trajs_shuffled_idx[traj_idx][
            self.cached_trajs_starts[traj_idx]
        ]
        ts = self.padded_ts[start_t + self.stack_idx]
        self.cached_trajs_starts[traj_idx] += 1

        if self.cached_trajs_starts[traj_idx] == len(
            self.cached_trajs_shuffled_idx[traj_idx]
        ):
            self.cached_trajs.pop(traj_idx)
            self.cached_trajs_starts.pop(traj_idx)
            self.cached_trajs_shuffled_idx.pop(traj_idx)
        return traj, ts


class ReplayDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir,
        aug_cfg,
        token_name,
        shuffle=True,
        obs_shapes=None,
        act_shape=None,
        action_horizon=1,
        stride=1,
        cache_all_traj=False,
        **kwargs,
    ):
        self.dataset_dir = dataset_dir
        self.obs_shapes = obs_shapes
        self.act_shape = act_shape
        self.token_name = token_name
        self.aug_cfg = aug_cfg
        self.shuffle = shuffle
        self.cache_all_traj = cache_all_traj

        self.stride = stride

        # stacking
        self.stack_idx = aug_cfg.stack_idx
        self.stack_window = max(aug_cfg.stack_idx)

        # augmentation
        if self.aug_cfg.aug_prob > 0:
            self.aug_transform = T.Compose(
                [
                    T.RandomResizedCrop(
                        size=224,
                        scale=(0.7, 1.0),
                        interpolation=InterpolationMode.BILINEAR,
                        antialias=False,
                    ),
                    T.ColorJitter(brightness=0.3),
                ]
            )
        else:
            self.aug_transform = None

        # proprioception:
        self.use_proprio = aug_cfg.use_proprio
        if "use_rotation" in aug_cfg:
            self.use_rotation = aug_cfg.use_rotation
        else:
            self.use_rotation = True
        if "use_mv" in aug_cfg:
            self.use_mv = aug_cfg.use_mv
        else:
            self.use_mv = True

        if "use_depth" in aug_cfg:
            self.use_depth = aug_cfg.use_depth
            if not self.use_depth:
                print("ReplayDataset: not using depth.")
        else:
            self.use_depth = True
        self.buffer_filenames = load_traj_files(
            self.dataset_dir, self.token_name, stride=self.stride
        )
        self.all_trajs = {}
        if self.cache_all_traj:
            for traj_filename in self.buffer_filenames:
                self.all_trajs[traj_filename] = load_traj_from_memory(traj_filename)
            print(f"ReplayDataset: cached all trajectories.")

        assert (
            len(self.buffer_filenames) > 0
        ), "No trajectories found in the specified folders."
        print(
            f"ReplayDataset: found {len(self.buffer_filenames)} trajectories in the specified folders."
        )

        # for multiprocessing on workers. see utils/_worker_init_fn.
        self.world_rng = None
        self.worker_filenames = []
        self.worker_start, self.worker_end = None, None

        self.cached_rgb_frames = None
        self.action_horizon = action_horizon
        self.traj_loader = CachedTrajLoader(
            shuffle=shuffle,
            stack_idx=self.stack_idx,
            stack_window=self.stack_window,
            all_traj_cache=self.all_trajs,
        )

    def __iter__(self):
        self.reset_buffer()

        return self.__next__()

    def __next__(self):
        while True:
            traj, ts = self.traj_loader.sample()
            if ts is None:
                break
            ret_traj = traj.slice(ts)

            ret_traj["obs"]["rgb"] = self._augment_frames(ret_traj["obs"]["rgb"])
            if not self.use_depth:
                ret_traj["obs"]["depth"] = np.zeros_like(ret_traj["obs"]["depth"])
            elif not self.use_proprio:
                ret_traj["obs"]["state"] = np.zeros_like(ret_traj["obs"]["state"])
            last_t = ts[-1]
            actions = traj["actions"][last_t : last_t + self.action_horizon]
            # pad actions to the same length
            if len(actions) < self.action_horizon:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros((self.action_horizon - len(actions), *self.act_shape)),
                    ]
                )
            ret_traj["actions"] = actions
            ret_traj["dones"] = np.array([ts[-1] == len(traj) - 1], dtype=np.int32)
            ret_traj["steps"] = np.array([ts[-1]], dtype=np.int32)
            yield ret_traj

    def _augment_frames(self, rgb_frames):
        """
        Augment the trajectory according to the augmentation config.
        This includes RGB augmentation and frame stacking.
        """
        p = np.random.rand()
        if p < self.aug_cfg.aug_prob and self.aug_transform is not None:
            rgb_frames, sh = pack_one(rgb_frames, "* c h w")
            rgb_frames = torch.from_numpy(rgb_frames).float() / 255.0
            rgb_frames = self.aug_transform(rgb_frames).numpy()
            ret = unpack_one(rgb_frames, sh, "* c h w")
            # back to [0, 255]
            ret = np.clip(ret * 255, 0, 255).astype(np.uint8)
            return ret
        else:
            return rgb_frames

    def reset_buffer(self):
        if self.shuffle:
            self.world_rng.shuffle(self.buffer_filenames)
        self.worker_filenames = self.buffer_filenames[
            self.worker_start : self.worker_end
        ]
        self.traj_loader.reset(self.worker_filenames)
