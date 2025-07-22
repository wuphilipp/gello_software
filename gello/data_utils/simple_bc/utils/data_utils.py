import os
import torch
import cv2 as cv
import numpy as np
import gdict as gd
import glob
from natsort import natsorted
from functools import partial


# worker functions
def _worker_init_fn(worker_id, rank=0, world_size=1):
    """
    For each process, each worker will get a slice of the buffer files (worker_start, worker_end)
    For each epoch, the buffer_files will be permuted (with the same random seed across workers)
    When there are multiple processes, each global worker will get a slice of the permuted buffer files
    """
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    num_global_workers = num_workers * world_size
    global_worker_id = rank * num_workers + worker_id
    dataset = worker_info.dataset

    N = len(dataset.buffer_filenames)
    per_worker = N // num_global_workers

    dataset.worker_start = global_worker_id * per_worker
    dataset.worker_end = min((global_worker_id + 1) * per_worker, N)
    dataset.world_rng = np.random.RandomState(41248)


# collate functions
def collate_fn(batch):
    return gd.GDict.stack(batch, axis=0).to_torch(
        non_blocking=True, dtype="float32", use_copy=True
    )


def get_token_name(encoder_cfg):
    return "none"


def normalize_quat(quats, dim):
    assert quats.shape[1] == 4
    quats = quats.copy()
    signs = np.sign(quats[:, dim])
    return quats * signs[:, None]


def load_traj_from_memory(filename):
    trajs = gd.GDict.from_hdf5(filename)
    key = natsorted(trajs.keys())[
        0
    ]  # Take the first key, assuming there is only one traj per file
    traj = gd.DictArray(trajs[key])
    traj = traj.select_by_keys(["obs", "actions"])
    traj = traj.to_two_dims()

    return traj


def load_traj_files(folder, token_name="none", stride=1):
    """
    Load the trajectory filenames from the folders.
    """
    # Recursive
    buffer_filenames = glob.glob(
        os.path.join(folder, token_name, "**/*.h5"), recursive=True
    )
    buffer_filenames = natsorted(buffer_filenames)

    buffer_filenames = [
        buffer_filenames[i] for i in range(len(buffer_filenames)) if i % stride == 0
    ]
    print(f"ReplayDataset: loading {len(buffer_filenames)} trajectories from {folder}")

    return buffer_filenames


def replace_abs_dataset_path(dataset_path):
    data_dir = os.environ.get("DATASET_DIR", None)
    if data_dir is None:
        return dataset_path
    else:
        dataset_name = os.path.basename(dataset_path)
        return os.path.join(data_dir, dataset_name)


def get_dataloader(replay, mode, num_workers, batch_size):
    if mode == "val":
        num_workers = min(num_workers, 3)
    context = (
        None if num_workers == 1 else torch.multiprocessing.get_context("forkserver")
    )
    world_rank, world_size = 0, 1
    loader = torch.utils.data.DataLoader(
        replay,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=partial(_worker_init_fn, rank=world_rank, world_size=world_size),
        persistent_workers=True if num_workers > 1 else False,
        multiprocessing_context=context if num_workers > 1 else None,
        drop_last=mode == "train",
        prefetch_factor=5,
    )
    return loader


def list_batch_to_episodic(all_dones, all_list):
    """
    Convert a list of batched data to a list of episodes based on the dones.
    all_dones: list of dones, each dones is a 1D array of shape (episode_length,)
    all_list: A dictionary, where each value is a list of batched data
    """
    all_dones = np.concatenate(all_dones, axis=0)
    for key in all_list:
        all_list[key] = np.concatenate(all_list[key], axis=0)
    split_idx = np.where(all_dones == 1)[0] + 1
    episodic_list = {}
    for key in all_list:
        episodic_list[key] = np.split(all_list[key], split_idx, axis=0)[:-1]
    return episodic_list
