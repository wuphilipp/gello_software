import torch
import torch.nn as nn
import numpy as np
from einops import pack, unpack, repeat, reduce, rearrange


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length)


def pack_one(x, pattern):
    return pack([x], pattern)


def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


def to_torch(array, device="cpu"):
    if isinstance(array, torch.Tensor):
        return array.to(device)
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)
    else:
        return torch.tensor(array).to(device)


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array


def to_cpu(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu()
    elif isinstance(array, tuple):
        return tuple(to_cpu(a) for a in array)


@torch.no_grad()
def batch_pred(func, kwargs, batch_size=2048, collate_fn=None, get_cpu=False):
    rand_key = list(kwargs.keys())[0]
    N = len(kwargs[rand_key])
    if N <= batch_size:
        return func(**kwargs)
    else:
        all_pred = []
        for i in range(0, N, batch_size):
            new_kwargs = {}
            for key, val in kwargs.items():
                if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
                    new_kwargs[key] = val[i : min(i + batch_size, N)]
                    pred = func(**new_kwargs)
                    if get_cpu:
                        pred = to_cpu(pred)
                else:
                    new_kwargs[key] = val
                    pred = func(**new_kwargs)
            all_pred.append(pred)
        if collate_fn is None:
            return torch.cat(all_pred, dim=0)
        else:
            return collate_fn(all_pred)


def get_named_trainable_params(model):
    return [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
