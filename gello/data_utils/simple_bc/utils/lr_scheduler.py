import math
import torch
from torch import nn


class CosineAnnealingLRWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_lr, warmup_epoch, T_max, last_epoch=-1):
        self.warmup_lr = warmup_lr
        self.warmup_epoch = warmup_epoch
        self.T_max = T_max
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for i in range(len(self.base_lrs)):
            if self.last_epoch < self.warmup_epoch:
                lr = (
                    self.warmup_lr
                    + (self.base_lrs[i] - self.warmup_lr)
                    * self.last_epoch
                    / self.warmup_epoch
                )
            else:
                lr = (
                    0.5
                    * self.base_lrs[i]
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (self.last_epoch - self.warmup_epoch)
                            / (self.T_max - self.warmup_epoch)
                        )
                    )
                )
            lrs.append(lr)
        return lrs


if __name__ == "__main__":
    # Test code
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    simple_nn = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))

    optimizer = torch.optim.SGD(simple_nn.parameters(), lr=0.1)
    scheduler = CosineAnnealingLRWithWarmup(
        optimizer, warmup_lr=0.01, warmup_epoch=10, T_max=100
    )
    lrs = []
    plt.figure(figsize=(10, 5))
    for i in range(100):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
    plt.plot(lrs)
    plt.show()
    plt.savefig("lr.png")
