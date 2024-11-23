import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import repeat
from omegaconf import OmegaConf

import gdict
from simple_bc._interfaces.encoder import Encoder
from simple_bc._interfaces.policy import Policy
from simple_bc.dataset.replay_dataset import ReplayDataset
from simple_bc.utils import data_utils
from simple_bc.utils.torch_utils import to_numpy, to_torch
from simple_bc.utils.visualization_utils import make_grid_video_from_numpy

from natsort import natsorted
import glob
from tqdm import tqdm
from einops import rearrange


@torch.no_grad()
def evaluate(encoder, policy, dataloader, suffix="", rgb=False):
    """
    Evaluate the policy. Return a dictionary of the loss and any other metrics.
    """
    encoder.eval()
    policy.eval()
    running_mse, running_abs, tot_items = 0, 0, 0

    all_pred_actions, all_actions, all_attn, all_rgb, all_dones = [], [], [], [], []
    for batch in dataloader:
        obses, actions, dones = batch["obs"], batch["actions"], batch["dones"]
        obses = gdict.GDict(obses).cuda(device="cuda")
        actions = actions.to(device="cuda")
        pred_actions, info = policy(encoder(obses))

        if "attn" in info:
            all_attn.append(to_numpy(info["attn"]))
            if rgb:
                rgb_frame = obses["rgb"]  # (B, F, V, C, H, W); float32 in [0, 255]
                rgb_frame = rearrange(rgb_frame, "b f v c h w -> b f v h w c")[:, -1]
                all_rgb.append(to_numpy(rgb_frame))

        all_pred_actions.append(to_numpy(pred_actions))
        all_actions.append(to_numpy(actions))
        all_dones.append(to_numpy(dones))
        first_pred_actions = pred_actions[:, 0]
        first_gt_actions = actions[:, 0]
        running_mse += (
            ((first_pred_actions - first_gt_actions) ** 2).sum(0).mean()
        ).item()
        running_abs += (
            (torch.abs(first_pred_actions - first_gt_actions)).sum(0).mean()
        ).item()
        B = actions.shape[0]
        tot_items += B

    metrics = {
        f"val/mse{suffix}": running_mse / tot_items,
        f"val/abs{suffix}": running_abs / tot_items,
    }

    batch_list = {"gt_actions": all_actions, "pred_actions": all_pred_actions}
    if len(all_attn) > 0:
        batch_list["attn"] = all_attn
    if len(all_rgb) > 0:
        batch_list["rgb"] = all_rgb

    episodic_list = data_utils.list_batch_to_episodic(all_dones, batch_list)
    return metrics, episodic_list


def make_rgb_attn_video(rgb, attn, save_name):
    """
    rgb: (T, V, H, W, C)
    attn: (T, F, V, G, pH, pW)
    """

    T, F, V, G = attn.shape[:4]
    H, W = rgb.shape[2:4]
    # rgb pad the first frame by F-1
    rgb = np.concatenate([np.expand_dims(rgb[0], 0)] * (F - 1) + [rgb], axis=0)
    # Normalize the attention map per image
    attn_min = np.min(attn, axis=(-1, -2), keepdims=True)
    attn_max = np.max(attn, axis=(-1, -2), keepdims=True)
    attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)

    all_videos = []
    for v in range(V):
        for f in range(F):
            if f != F - 1:
                continue  # Only keep the last frame for visualization
            rgb_t = rgb[f : f + T, v]  # T H W C
            attn_t = attn[:, f, v]  # Take the last frame, (T G pH pW)
            attn_t_torch = to_torch(attn_t)
            attn_t = torch.nn.functional.interpolate(
                attn_t_torch, size=(224, 224), mode="nearest"
            )
            rgb_t = to_torch(rearrange(rgb_t, "t h w c -> t c h w")) / 255.0
            rgb_t = torch.nn.functional.interpolate(
                rgb_t, size=(224, 224), mode="nearest"
            )

            rgb_t = to_numpy(rgb_t)  # T, C, H, W
            rgb_t = rearrange(rgb_t, "t c h w -> t h w c") * 255.0

            attn_t = to_numpy(attn_t)  # T, G, H, W
            attn_t = repeat(attn_t, "t g h w -> g t h w c", c=3)

            attn_t *= 255.0  # T, G, H, W, C

            import cv2

            attn_t = attn_t.astype(np.uint8)
            attn_t = [
                [
                    cv2.applyColorMap(attn_t[g_index][t_index], cv2.COLORMAP_VIRIDIS)
                    for t_index in range(T)
                ]
                for g_index in range(G)
            ]

            attn_t = np.array(attn_t)[..., ::-1].astype(np.float32)

            attn_t = attn_t * 0.5 + repeat(rgb_t, "t h w c -> g t h w c", g=G) * 0.5

            all_videos.append(rgb_t)
            all_videos.extend(attn_t)

    combined = make_grid_video_from_numpy(all_videos, ncol=G + 1, output_name=save_name)
    return combined


def make_action_comparison(pred_actions, gt_actions, save_name):
    """
    pred_actions: (T, H, D)
    gt_actions: (T, H, D)
    """
    T, H, D = pred_actions.shape

    row_size = (D // 4) + int(D % 4 == 0)

    plt.switch_backend("agg")
    fig, ax = plt.subplots(row_size, 4, figsize=(16, 8))
    plt.tight_layout()

    for d in range(D):
        d1, d2 = d // 4, d % 4
        if H > 1:
            for t in range(0, T, 10):
                if t == 0:
                    ax[d1, d2].plot(
                        range(t, t + H), pred_actions[t, :, d], label=f"pred", color="b"
                    )
                else:
                    ax[d1, d2].plot(range(t, t + H), pred_actions[t, :, d], color="b")
        else:
            ax[d1, d2].plot(pred_actions[:, 0, d], label="pred", color="b")

        ax[d1, d2].plot(gt_actions[:, 0, d], label="gt", color="orange")
        ax[d1, d2].set_title(f"Action {d}")
        ax[d1, d2].legend()
    plt.savefig(save_name)
    plt.close()


def make_visualization(val_replay, info, save_dir, prefix=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attn_list = info["attn"] if "attn" in info else None
    rgb_frames = info["rgb"] if "rgb" in info else None
    gt_actions_list, pred_actions_list = info["gt_actions"], info["pred_actions"]
    for idx, (gt_actions, pred_actions) in enumerate(
        zip(gt_actions_list, pred_actions_list)
    ):
        if attn_list is not None:
            rgb = rgb_frames[idx]
            make_rgb_attn_video(
                rgb,
                attn_list[idx],
                save_name=os.path.join(save_dir, f"{prefix}traj_{idx}.mp4"),
            )
        make_action_comparison(
            pred_actions,
            gt_actions,
            save_name=os.path.join(save_dir, f"{prefix}traj_{idx}.png"),
        )


@click.command()
@click.option(
    "--dataset",
    type=str,
    default=None,
    help="Dataset for evaluation. If in simulation, must contain .json files organized by instance!",
)
@click.option("--policy_path", "-a", type=str, default=None)
def main(dataset, policy_path):
    data_dir = os.environ["DATASET_DIR"]
    eval_dir = os.path.join(data_dir, "eval")
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = os.path.dirname(policy_path)
    encoder_path = policy_path.replace("policy", "encoder")
    cfg_path = os.path.join(exp_dir, "config.yaml")
    train_cfg = OmegaConf.load(cfg_path)

    print(f"Loading encoder from {encoder_path}")
    encoder = Encoder.build_encoder(train_cfg.encoder).to(device)
    policy = Policy.build_policy(
        encoder.out_shape, train_cfg.policy, train_cfg.encoder
    ).to(
        device
    )  # Updated shape
    encoder.load(encoder_path)
    policy.load(policy_path)

    eval_replay = ReplayDataset(dataset_dir=dataset, **train_cfg.dataset, shuffle=False)
    # Have to use worker = 1 to reserve the order
    eval_dataloader = data_utils.get_dataloader(
        eval_replay, "val", 1, train_cfg.batch_size
    )

    _, info = evaluate(encoder, policy, eval_dataloader)
    make_visualization(eval_replay, info, eval_dir, prefix="eval_")


if __name__ == "__main__":
    main()
