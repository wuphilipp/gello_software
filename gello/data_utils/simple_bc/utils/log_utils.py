import wandb
import os
import torch
import torch.distributed as dist
import numpy as np
import random
import builtins
import datetime
import time
from collections import defaultdict, deque
from omegaconf import DictConfig, OmegaConf
import json


def init_wandb(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    pretty_print_cfg(cfg)
    wandb_cfg = prepare_wandb_cfg(cfg)

    wandb.init(
        config=wandb_cfg,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
    )
    OmegaConf.save(cfg, f"{wandb.run.dir}/config.yaml")


def log_wandb(metrics, step):
    """
    Log the metrics to wandb. The metrics should be hierarchical, with the top level keys being "train" and "val".
    """
    for key, value in metrics.items():
        wandb.log(value, step=step, commit=False)


def pretty_print_cfg(cfg):
    """
    Pretty print the config as cascading bullet points.
    """
    print("Config:")
    for key, value in cfg.items():
        print(f"- {key}:")
        if isinstance(value, DictConfig) or isinstance(value, dict):
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"  - {value}")


def prepare_wandb_cfg(cfg):
    wandb_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            wandb_cfg[key] = prepare_wandb_cfg(value)
        else:
            wandb_cfg[key] = value

    return wandb_cfg


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class BestAvgLoss(object):
    # Best loss is
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.best_loss, self.best_epoch = None, None

    def update_best(self, loss, epoch):
        self.deque.append(loss)
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            is_best = True
        else:
            is_best = False
        return is_best


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        """Window size is for the best key"""
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f}s/it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def save_cfg(cfg, save_name):
    # cfg is a dictionary
    with open(save_name, "w") as f:
        json.dump(cfg, f, indent=2)
