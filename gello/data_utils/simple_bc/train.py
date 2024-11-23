import os

import hydra
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import gdict
from simple_bc.eval import make_visualization, evaluate
from simple_bc._interfaces.encoder import Encoder
from simple_bc._interfaces.policy import Policy
from simple_bc.dataset.replay_dataset import ReplayDataset
from simple_bc.utils import log_utils, data_utils, torch_utils
import time


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

    val_vis_stride = cfg.get("val_vis_stride", 1)

    token_name = data_utils.get_token_name(cfg.encoder)
    cfg.dataset.token_name = token_name
    update_dataset(cfg)
    OmegaConf.save(config=cfg, f=os.path.join(work_dir, "config.yaml"))

    train_replay = ReplayDataset(dataset_dir=cfg.train_dataset, **cfg.dataset)
    train_dataloader = data_utils.get_dataloader(
        train_replay, "train", cfg.num_workers, cfg.batch_size
    )

    val_replay_l1 = ReplayDataset(dataset_dir=cfg.val_dataset_l1, **cfg.dataset)
    val_dataloader_l1 = data_utils.get_dataloader(
        val_replay_l1, "val", cfg.num_workers, cfg.batch_size * 2
    )

    val_replay_l2 = ReplayDataset(dataset_dir=cfg.val_dataset_l2, **cfg.dataset)
    val_dataloader_l2 = data_utils.get_dataloader(
        val_replay_l2, "val", cfg.num_workers, cfg.batch_size * 2
    )

    log_utils.init_wandb(cfg)

    optimizer = setup_optimizer(cfg.optimizer_cfg, encoder, policy)
    scheduler = setup_lr_scheduler(optimizer, cfg.scheduler_cfg)

    # Pick ckpt based on  the average of the last 5 epochs
    metric_logger = log_utils.MetricLogger(delimiter=" ")
    best_loss_logger = log_utils.BestAvgLoss(window_size=5)

    for epoch in metric_logger.log_every(range(cfg.epochs), 1, ""):
        train_metrics = run_one_epoch(
            encoder,
            policy,
            train_dataloader,
            optimizer,
            scheduler,
            clip_grad=cfg.clip_grad,
        )

        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        metric_logger.update(**train_metrics)
        wandb.log(train_metrics, step=epoch)

        if epoch % cfg.val_freq == 0:
            val_metrics_l1, _ = evaluate(encoder, policy, val_dataloader_l1, "_L1")
            val_metrics_l2, _ = evaluate(encoder, policy, val_dataloader_l2, "_L2")
            combined_metrics = dict(**val_metrics_l1, **val_metrics_l2)

            # Save best checkpoint
            metric_logger.update(**combined_metrics)

            loss_metric = combined_metrics["val/mse_L1"]
            is_best = best_loss_logger.update_best(loss_metric, epoch)

            if is_best:
                encoder.save(f"{work_dir}/encoder_best.ckpt")
                policy.save(f"{work_dir}/policy_best.ckpt")
                with open(f"{work_dir}/best_epoch.txt", "w") as f:
                    f.write(
                        "Best epoch: %d, Best %s: %.4f"
                        % (epoch, "loss", best_loss_logger.best_loss)
                    )
            wandb.log(combined_metrics, step=epoch)

        if epoch % cfg.save_freq == 0:
            encoder.save(f"{work_dir}/encoder_{epoch}.ckpt")
            policy.save(f"{work_dir}/policy_{epoch}.ckpt")

            def visualize():
                val_replay_l1 = ReplayDataset(
                    dataset_dir=cfg.val_dataset_l1,
                    **cfg.dataset,
                    shuffle=False,
                    stride=val_vis_stride,
                )
                val_replay_l1.aug_transform = None
                val_dataloader_l1 = data_utils.get_dataloader(
                    val_replay_l1, "val", 1, cfg.batch_size
                )

                val_replay_l2 = ReplayDataset(
                    dataset_dir=cfg.val_dataset_l2,
                    **cfg.dataset,
                    shuffle=False,
                    stride=val_vis_stride,
                )
                val_replay_l2.aug_transform = None
                val_dataloader_l2 = data_utils.get_dataloader(
                    val_replay_l2, "val", 1, cfg.batch_size
                )

                _, info_l1 = evaluate(
                    encoder, policy, val_dataloader_l1, "_L1", rgb=True
                )
                _, info_l2 = evaluate(
                    encoder, policy, val_dataloader_l2, "_L2", rgb=True
                )
                save_dir = os.path.join(work_dir, f"visualization_{epoch}")
                make_visualization(
                    val_replay_l1, info_l1, save_dir=save_dir, prefix="l1_"
                )
                make_visualization(
                    val_replay_l2, info_l2, save_dir=save_dir, prefix="l2_"
                )

            print("Visualizing current epoch...")

            visualize()

    encoder.save(f"{work_dir}/encoder_final.ckpt")
    policy.save(f"{work_dir}/policy_final.ckpt")
    print(f"finished training in {wandb.run.dir}")

    def visualize_best():
        encoder.load(f"{work_dir}/encoder_best.ckpt")
        policy.load(f"{work_dir}/policy_best.ckpt")

        val_replay_l1 = ReplayDataset(
            dataset_dir=cfg.val_dataset_l1,
            **cfg.dataset,
            shuffle=False,
            stride=val_vis_stride,
        )
        val_replay_l1.aug_transform = None
        val_dataloader_l1 = data_utils.get_dataloader(
            val_replay_l1, "val", 1, cfg.batch_size
        )

        val_replay_l2 = ReplayDataset(
            dataset_dir=cfg.val_dataset_l2,
            **cfg.dataset,
            shuffle=False,
            stride=val_vis_stride,
        )
        val_replay_l2.aug_transform = None
        val_dataloader_l2 = data_utils.get_dataloader(
            val_replay_l2, "val", 1, cfg.batch_size
        )
        _, info_l1 = evaluate(encoder, policy, val_dataloader_l1, "_L1", rgb=True)
        _, info_l2 = evaluate(encoder, policy, val_dataloader_l2, "_L2", rgb=True)
        save_dir = os.path.join(work_dir, "visualization_best_epoch")
        make_visualization(val_replay_l1, info_l1, save_dir=save_dir, prefix="l1_")
        make_visualization(val_replay_l2, info_l2, save_dir=save_dir, prefix="l2_")

    print("Visualizing the best epoch...")
    visualize_best()

    wandb.finish()


def setup_optimizer(optim_cfg, encoder, policy):
    """
    Setup the optimizer. Return the optimizer.
    """
    from torch import optim

    optimizer = eval(optim_cfg.type)
    encoder_trainable_params = torch_utils.get_named_trainable_params(encoder)
    # Print size of trainable parameters
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
    if len(encoder_trainable_params) > 0:
        return optimizer(
            list(encoder.parameters()) + list(policy.parameters()), **optim_cfg.params
        )
    else:
        return optimizer(list(policy.parameters()), **optim_cfg.params)


def setup_lr_scheduler(optimizer, scheduler_cfg):
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    from simple_bc.utils.lr_scheduler import CosineAnnealingLRWithWarmup

    sched = eval(scheduler_cfg.type)
    if sched is None:
        return None
    return sched(optimizer, **scheduler_cfg.params)


def run_one_epoch(
    encoder, policy, dataloader, optimizer, scheduler=None, clip_grad=None
):
    """
    Optimize the policy. Return a dictionary of the loss and any other metrics.
    """
    running_loss, running_mse, running_abs, tot_items = 0, 0, 0, 0

    encoder.train()
    policy.train()
    loss_fn = nn.MSELoss(reduction="none")

    for batch in dataloader:
        obs, act = batch["obs"], batch["actions"]
        obs = gdict.GDict(obs).cuda(device="cuda")
        act = act.to(device="cuda")

        optimizer.zero_grad()
        pred, _ = policy(encoder(obs))  # pred: (B, H, A)
        loss = loss_fn(pred, act).mean()

        loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)

        optimizer.step()
        running_mse += ((pred - act) ** 2).sum(0).mean().item()
        running_abs += (torch.abs(pred - act)).sum(0).mean().item()
        running_loss += loss.item() * act.shape[0]
        tot_items += act.shape[0]

    out_dict = {
        "train/mse": running_mse / tot_items,
        "train/abs": running_abs / tot_items,
        "train/loss": running_loss / tot_items,
    }

    if scheduler is not None:
        scheduler.step()

    return out_dict

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
