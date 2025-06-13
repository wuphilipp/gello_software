import os

import hydra
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import gdict
import simple_bc
from simple_bc._interfaces.encoder import Encoder
from simple_bc._interfaces.policy import Policy
from simple_bc.dataset.replay_dataset import ReplayDataset
from simple_bc.utils import log_utils, data_utils, torch_utils
import time
import tqdm


def update_dataset(cfg):
    from simple_bc.constants import BC_DATASET
    from omegaconf import open_dict  # Allows to modify cfg in place

    with open_dict(cfg):
        cfg.train_dataset = BC_DATASET[cfg.task]["train_dataset"]
        cfg.val_dataset_l1 = BC_DATASET[cfg.task]["val_dataset_l1"]
        cfg.val_dataset_l2 = BC_DATASET[cfg.task]["val_dataset_l2"]


@hydra.main(config_path="../conf", version_base="1.3")
def main(cfg: DictConfig):
    work_dir = HydraConfig.get().runtime.output_dir
    setup(cfg)
    encoder = Encoder.build_encoder(cfg.encoder).to(cfg.device)
    policy = Policy.build_policy(encoder.out_shape, cfg.policy, cfg.encoder).to(
        cfg.device
    )  # Updated shape

    encoder_trainable_params = torch_utils.get_named_trainable_params(encoder)
    print(
        "Encoder trainable parameters:",
        sum(p.numel() for (name, p) in encoder_trainable_params) / 1e6,
        "M",
    )
    print(
        "Policy trainable parameters:",
        sum(p.numel() for p in policy.parameters()) / 1e6,
        "M",
    )
    token_name = data_utils.get_token_name(cfg.encoder)
    cfg.dataset.token_name = token_name
    update_dataset(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    val_replay_l2 = ReplayDataset(dataset_dir=cfg.val_dataset_l2, **cfg.dataset)
    val_dataloader_l2 = data_utils.get_dataloader(
        val_replay_l2, "val", cfg.num_workers, 1
    )

    print(f"profiling run time for x1000: no cache")
    all_times = []
    enc_times = []
    pol_times = []
    batch = next(iter(val_dataloader_l2))
    obs, act = batch["obs"], batch["actions"]
    obs = gdict.GDict(obs).cuda(device="cuda")
    act = act.to(device="cuda")

    with torch.no_grad():
        for _ in tqdm.tqdm(range(1000)):
            start = time.time()
            enc_start = time.time()
            feats = encoder(obs)
            enc_end = time.time()
            pol_start = time.time()
            _ = policy(feats)
            pol_end = time.time()
            end = time.time()

            all_times += [end - start]
            enc_times += [enc_end - enc_start]
            pol_times += [pol_end - pol_start]

    # report the mean and std of the times in ms
    print(f"all: {torch.tensor(all_times).mean() * 1000} +- {torch.tensor(all_times).std() * 1000}ms")
    print(f"enc: {torch.tensor(enc_times).mean() * 1000} +- {torch.tensor(enc_times).std() * 1000}ms")
    print(f"pol: {torch.tensor(pol_times).mean() * 1000} +- {torch.tensor(pol_times).std() * 1000}ms")

    if isinstance(encoder, simple_bc.encoder.SpawnNet):
        print(f"profiling run time for x1000: with cache")

        all_times = []
        enc_times = []
        pol_times = []

        batch = next(iter(val_dataloader_l2))
        obs, act = batch["obs"], batch["actions"]
        obs = gdict.GDict(obs).cuda(device="cuda")
        act = act.to(device="cuda")

        pretrained_feats = encoder.get_pretrained_feats(obs)
        with torch.no_grad():
            for _ in tqdm.tqdm(range(1000)):
                start = time.time()
                enc_start = time.time()
                feats = encoder(obs, pretrained_feats)
                enc_end = time.time()
                pol_start = time.time()
                _ = policy(feats)
                pol_end = time.time()
                end = time.time()
                all_times += [end - start]
                enc_times += [enc_end - enc_start]
                pol_times += [pol_end - pol_start]

        # report in ms
        print(f"all: {torch.tensor(all_times).mean() * 1000} +- {torch.tensor(all_times).std() * 1000}ms")
        print(f"enc: {torch.tensor(enc_times).mean() * 1000} +- {torch.tensor(enc_times).std() * 1000}ms")
        print(f"pol: {torch.tensor(pol_times).mean() * 1000} +- {torch.tensor(pol_times).std() * 1000}ms")

def setup(cfg):
    if cfg.gpu is not None:
        print(f"Using GPU {cfg.gpu}")
        torch.cuda.set_device(cfg.gpu)

    import warnings

    warnings.simplefilter("ignore")

    from simple_bc.utils.log_utils import set_random_seed

    set_random_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)


if __name__ == "__main__":
    main()
