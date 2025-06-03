from typing import Optional

import numpy as np
from numbers import Number
from .misc import equal
from .type_utils import is_np_arr, is_type, get_dtype, is_integer, is_torch, is_np, is_seq_of, is_num


""" Convert tensor type """


def as_dtype(item, dtype: str):
    if is_np(item):
        return item.astype(dtype)
    elif is_torch(item):
        import torch

        return item.to(getattr(torch, dtype))
    else:
        try:
            dtype = eval(dtype)
            return dtype(item)
        except (RuntimeError, NameError, ValueError):
            return item


def to_torch(
    item, use_copy: bool = False, device: Optional[str] = "cpu", non_blocking: bool = False, dtype: Optional[str] = None, requires_grad: bool = False
):
    import torch

    dtype_torch_map = {"uint16": "int"}

    item_dtype = get_dtype(item)
    same_type = equal(item_dtype, dtype)
    if hasattr(device, "type"):
        device = f"{device.type}:{device.index}" if device.index is not None else f"{device.type}"

    if dtype is not None and dtype:
        dtype = getattr(torch, dtype)

    if is_seq_of(item, Number) or is_num(item):
        item = to_np(item)

    if is_np(item):
        if item_dtype in dtype_torch_map:
            item = item.astype(dtype_torch_map[item_dtype])
        if use_copy or not device.startswith("cpu") or requires_grad or np.isscalar(item) or not same_type:
            extra_kwargs = {} if dtype is None else {"dtype": dtype}
            return torch.tensor(item, requires_grad=requires_grad, device=device, **extra_kwargs)
        else:
            return torch.from_numpy(item)
    elif is_torch(item):
        if not equal(item.device.type, device):
            item = item.to(device=device, non_blocking=non_blocking)
        if not same_type:
            item = item.to(dtype=dtype, non_blocking=non_blocking)
        if use_copy:
            item = item.clone().detach()
        item = item.requires_grad_(requires_grad)
        return item
    else:
        return item


def to_np(item, use_copy=False, dtype=None):
    use_copy = use_copy or np.isscalar(item) or not equal(get_dtype(item), dtype)
    if isinstance(item, (str, bytes)):
        return np.array([item], dtype=object)
    elif is_seq_of(item, Number):
        return np.array(item, dtype=get_dtype(item[0]) if dtype is None else dtype)

    if is_np(item):
        kwargs = {} if dtype is None else {"dtype": dtype}
        return np.array(item, **kwargs) if use_copy else item
    elif is_torch(item):
        item = item.detach().cpu().numpy()
        return to_np(item, False, dtype)
    else:
        return item


def to_array(item):
    if is_torch(item):
        return item.reshape(1) if item.nelement() == 1 else item
    elif is_np_arr(item):
        return item if item.ndim > 0 else item.reshape(1)
    elif is_num(item) or (hasattr(item, "ndim") and item.ndim == 0):
        return np.array([item]).reshape(1)
    else:
        try:
            return np.array([item], dtype=object).reshape(1)
        except:
            print(item)


""" Convert normal python type """


def dict_to_seq(x):
    keys = list(sorted(x.keys()))
    values = [x[k] for k in keys]
    return keys, values


def seq_to_dict(keys, values):
    return {keys[i]: values[i] for i in range(len(keys))}


def dict_to_str(x):
    ret = ""
    for key in x:
        if ret != "":
            ret += " "
        if isinstance(x[key], (float, np.float32, np.float64)):
            from math import log10

            if abs(x[key]) < 1e-8:
                ret += f"{key}: 0"
            elif -1 <= log10(abs(x[key])) <= 5:  # > 10000 or < 0.0001
                ret += f"{key}: {x[key]:.3f}"
            else:
                ret += f"{key}: {x[key]:.3e}"
        else:
            ret += f"{key}: {x[key]}"
    return ret


def list_to_str(x):
    return '[' + ','.join([f'{x[i]:.3f}' for i in range(len(x))]) + ']'


def slice_to_range(item):
    start = item.start if item.start is not None else 0
    step = item.step if item.step is not None else 1
    return range(start, item.stop, step)


def range_to_slice(item):
    return slice(item.start, item.stop, item.step)


def index_to_slice(index):
    if len(index) == 1:
        return index
    diff = np.diff(index)
    is_sorted = np.all(diff[0] == diff)
    if is_sorted:
        si, ei = index[0], index[-1]
        index = slice(si, ei + 1, diff[0])
    return index