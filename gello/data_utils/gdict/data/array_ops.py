from io import BytesIO
import random
import sys
import base64
import numpy as np

from .converter import range_to_slice, to_np, to_torch, slice_to_range
from .type_utils import get_dtype, is_np, is_np_arr, is_num, is_torch, is_integer, is_torch_distribution, is_not_null, is_arr, is_h5
from .wrappers import seq_to_np


""" Unified API for torch and numpy """

# We import torch inside the function to reduce the memory usage when you only want to work with numpy array


def to_float(item):
    if np.isscalar(item):
        item = float(item)
    elif isinstance(item, np.ndarray):
        item = item.astype(np.float32)
    return item


def deepcopy(item):
    from copy import deepcopy

    if is_np_arr(item):
        item = item.copy()
    elif is_torch(item):
        item = item.clone()
    elif not is_h5(item):
        item = deepcopy(item)

    return item


def unsqueeze(item, axis):
    if is_np_arr(item):
        # Trick to speed up, expand_dims is very slow ....
        if axis == 0:
            return item[None]
        elif axis == -1:
            return item[..., None]
        item = np.expand_dims(item, axis)
    elif is_torch(item):
        if axis == 0:
            return item[None]
        elif axis == -1:
            return item[..., None]
        item = item.unsqueeze(axis)
    elif is_torch_distribution(item):
        item = ops_single_torch_distribution(item, unsqueeze, axis=axis)
    return item


def squeeze(item, axis):
    if is_np_arr(item):
        if axis == 0:
            return item[0]
        elif axis == -1:
            return item[..., 0]

        if axis is None:
            return np.squeeze(item)
        elif item.shape[axis] != 1:
            return item
        else:
            return np.squeeze(item, axis)
    elif is_torch(item):
        if axis == 0:
            return item[0]
        elif axis == -1:
            return item[..., 0]

        if axis is None:
            return item.squeeze()
        elif item.shape[axis] != 1:
            return item
        else:
            return item.squeeze(axis)
    elif is_torch_distribution(item):
        return ops_single_torch_distribution(item, squeeze, axis=axis)
    else:
        return item


def zeros_like(item):
    if is_np_arr(item):
        return np.zeros_like(item)
    elif is_torch(item):
        import torch

        return torch.zeros_like(item)
    else:
        return item


def ones_like(item):
    if is_np_arr(item):
        return np.ones_like(item)
    elif is_torch(item):
        import torch

        return torch.ones_like(item)
    else:
        return item


def repeat(item, rep, axis=None):
    # when axis=0, we will use tile for numpy and repeat for torch
    if not (is_np_arr(item) or is_torch(item)):
        return item
    if is_np_arr(item):
        if axis is None:
            return np.tile(item, rep)
        else:
            return np.repeat(item, rep, axis)
    else:
        import torch

        if axis is None:
            return item.repeat(*rep)
        else:
            return torch.repeat_interleave(item, rep, axis)


def tile(item, rep):
    if is_integer(rep):
        rep = (rep,)
    if is_np_arr(item):
        return np.tile(item, rep)
    elif is_torch(item):
        import torch

        return torch.tile(item, rep)
    else:
        return item


def slice_item(item, slice, axis=0):
    # Avoid copying the data for numpy array and torch tensor
    if is_arr(item) or is_torch(item):
        if axis == 0:
            ret = item[slice]
        elif axis == 1:
            ret = item[:, slice]
        elif axis == 2:
            ret = item[:, :, slice]
        elif axis == 3:
            ret = item[:, :, :, slice]
        else:
            raise NotImplementedError("Axis is too large!")
        return ret
    else:
        return item


def take(item, indices, axis=None):
    # It will copy the data for np array and torch tensor
    if isinstance(item, list):
        assert axis == 0 and is_num(indices), "For list we only support operation on the first dimension!"
        return item[indices]

    # Convert slice and range to np array
    if isinstance(indices, slice):
        indices = slice_to_range(indices)
    if isinstance(indices, range):
        indices = list(indices)
    indices = np.array(indices, dtype=np.int64) if isinstance(indices, (list, tuple)) else indices

    if is_np_arr(item):
        return item.take(indices=indices, axis=axis)
    elif is_torch(item):
        single = False
        if not is_torch(indices):
            single = is_integer(indices)
            indices = to_torch(indices, device=item.device, non_blocking=True)
        if indices.ndim > 1:
            new_shape = list(item.shape)
            new_shape = new_shape[:axis] + list(indices.shape) + new_shape[axis + 1 :]
            ret = item.index_select(index=indices.reshape(-1), dim=axis).reshape(new_shape)
        else:
            ret = item.index_select(index=indices, dim=axis)
        if single:
            ret = ret.squeeze(dim=axis)
        return ret
    elif is_torch_distribution(item):
        return ops_single_torch_distribution(item, take, indices=indices, axis=axis)
    else:
        return item


def shuffle(item, axis=0):
    if isinstance(item, (list, tuple)):
        is_tuple = type(item) == tuple
        ret = list(item) if is_tuple else item
        random.shuffle(ret)
        return tuple(ret) if is_tuple else item
    elif is_np_arr(item):
        indices = np.random.permutation(item.shape[axis])
        return take(item, indices, axis=axis)
    elif is_torch(item):
        import torch

        indices = torch.randperm(item.shape[axis], device=item.device)
        return take(item, indices, axis=axis)
    else:
        return item


def reshape(item, newshape):
    if hasattr(item, "reshape"):
        item = item.reshape(newshape)
    return item


def split_dim(item, axis, newaxes):
    if is_arr(item) and len(newaxes) > 1:
        und_index = np.where(np.array(newaxes) == -1)[0]
        assert len(und_index) <= 1
        if len(und_index) == 1:
            und_index = und_index[0]
            newaxes[und_index] = 1
            newaxes[und_index] = item.shape[axis] // np.prod(newaxes)
        assert np.prod(newaxes) == item.shape[axis]
        item_shape = list(item.shape)
        item_shape = item_shape[:axis] + newaxes + item_shape[axis + 1 :]
        item = item.reshape(item_shape)
    elif is_torch_distribution(item) and len(newaxes) > 1:
        item = ops_single_torch_distribution(item, split_dim, axis=axis, newaxes=newaxes)
    return item


def transpose(item, axis0, axis1, contiguous=True):
    if is_np_arr(item):
        item = np.swapaxes(item, axis0, axis1)
    elif is_torch(item):
        import torch

        item = torch.transpose(item, axis0, axis1)
        if contiguous:
            item = item.contiguous()
    return item


def contiguous(item):
    if is_torch(item):
        item = item.contiguous()
    return item


def einsum(subscripts, *items):
    items = list(items)
    if is_np_arr(items[0]):
        return np.einsum(subscripts, *items)
    elif is_torch(items[0]):
        import torch

        return torch.einsum(subscripts, *items)
    return items


def concat(item, axis=0):
    if len(item) == 1:
        return item[0]
    elif is_np_arr(item[0]):
        return np.concatenate(item, axis=axis)
    elif is_torch(item[0]):
        import torch

        return torch.cat(item, dim=axis)
    elif is_torch_distribution(item[0]):
        return concat_torch_distribution(item, axis=axis)
    else:
        return item


def stack(item, axis=0):
    if len(item) == 1:
        return unsqueeze(item[0], axis)
    elif is_np_arr(item[0]):
        return np.stack(item, axis=axis)
    elif is_torch(item[0]):
        import torch

        return torch.stack(item, dim=axis)
    else:
        return item


def share_memory(x, y):
    if type(x) != type(y):
        return False
    elif is_np_arr(x):
        ret = x.base is not None and y.base is not None and x.base == y.base
        return ret.any() if is_np_arr(ret) else ret
    elif is_torch(x):
        sign = x.storage().data_ptr() == y.storage().data_ptr()
        return sign if isinstance(sign, bool) else sign.any()
    else:
        if isinstance(x, (int, str, float)):
            return False
        else:
            return id(x) == id(y)


def to_cpu(x):
    import torch

    if isinstance(x, torch.Tensor):
        x = x.cpu()
    return x


def to_cuda(x, device="cuda"):
    import torch

    if isinstance(x, torch.Tensor):
        x = x.to(device)
    return x


def type_as(item, other):
    if is_np_arr(item):
        return item.astype(other.dtype)
    elif is_torch(item):
        return item.type_as(other)
    else:
        return item


@seq_to_np(True)
def arr_sum(item, axis=None, keepdim=False, mask=None, dtype=None):
    if is_np_arr(item):
        item = item if mask is None else (type_as(mask, item) * item)
        return np.sum(item, axis, dtype, None, keepdim)
    elif is_torch(item):
        import torch

        item = item if mask is None else (type_as(mask, item) * item)
        if axis is None:
            return torch.sum(item)
        else:
            return torch.sum(item, axis, keepdim, dtype=dtype)
    else:
        return item


@seq_to_np(True)
def arr_mean(item, axis=None, keepdim=False, mask=None, dtype=None, mask_clip=1e-12):
    if is_np_arr(item) or is_torch(item):
        if mask is None:
            if is_np_arr(item):
                return np.mean(item, axis, dtype, None, keepdim)
            elif is_torch(item):
                import torch

                return torch.mean(item, axis, keepdim, dtype=dtype) if axis is not None else torch.mean(item, dtype=dtype)
        else:
            return arr_sum(item, axis, keepdim, mask) / (arr_sum(mask, axis, keepdim) + mask_clip)
    else:
        return item


# def mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue):
# @array_function_dispatch(_amin_dispatcher)
# def amin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
#  where=np._NoValue):


@seq_to_np(True)
def arr_min(item, axis=None, keepdim=False, mask=None, inf=1e30):
    if is_np_arr(item) or is_torch(item):
        if mask is not None:
            item = item * mask + inf * (1 - mask)  # Both torch and numpy can deal with inf
        if is_np_arr(item):
            return np.min(item, axis, None, keepdim)
        else:
            import torch

            return torch.min(item, axis, keepdim).values
    else:
        return item


@seq_to_np(True)
def arr_max(item, axis=None, keepdim=False, mask=None, inf=1e30):
    if is_np_arr(item) or is_torch(item):
        if mask is not None:
            item = item * mask + -inf * (1 - mask)  # Both torch and numpy can deal with inf
        if is_np_arr(item):
            return np.max(item, axis, None, keepdim)
        else:
            import torch

            return torch.max(item, axis, keepdim).values
    else:
        return item


def to_item(item):
    if is_np_arr(item):
        if item.size == 1:
            return item.reshape(-1)[0]
    elif is_torch(item):
        if item.numel() == 1:
            return item.item()
    return item


def select_with_mask(item, mask):
    if is_arr(item):
        return item[mask]
    else:
        return ops_single_torch_distribution(item, select_with_mask, mask=mask)


def recover_with_mask(item, mask):
    ret_shape = list(mask.shape) + list(item[0].shape)
    if is_np_arr(item):
        ret = np.zeros(ret_shape, dtype=item.dtype, device=item.device)
    else:
        import torch

        ret = torch.zeros(*ret_shape, dtype=item.dtype, device=item.device)
    ret[mask] = item
    return ret


def get_nbytes(item):
    if is_np_arr(item):
        if item.dtype == object:
            tmp = item.reshape(-1)
            return sum([get_nbytes(i) for i in tmp])
        else:
            return item.nbytes
    elif is_torch(item):
        return item.view(-1).shape[0] * item.element_size()
    else:
        return sys.getsizeof(item)


def split(item, split_size_or_sections, axis=0):
    # Use the torch style
    if is_np_arr(item):
        if is_integer(split_size_or_sections):
            num_blocks = int(item.shape[axis] // split_size_or_sections)
            tmp = [
                split_size_or_sections,
            ] * num_blocks
            if split_size_or_sections * num_blocks < item.shape[axis]:
                tmp.append(item.shape[axis] - split_size_or_sections * num_blocks)
            split_size_or_sections = tmp
        elif np.sum(split_size_or_sections) != item.shape[axis]:
            split_size_or_sections.append(item.shape[axis] - np.sum(split_size_or_sections))
        split_size_or_sections = np.cumsum(split_size_or_sections)
        return np.split(item, split_size_or_sections, axis=axis)[:-1]
    elif is_torch(item):
        import torch

        return torch.split(item, split_size_or_sections, dim=axis)
    return item


def norm(item, ord=None, axis=None, keepdim=False):
    if is_np_arr(item):
        return np.linalg.norm(item, ord, axis, keepdim)
    elif is_torch(item):
        return item.norm(ord, axis, keepdim)
    return item


def normalize(item, p=2.0, axis=1, eps=1e-12):
    if is_np_arr(item):
        return item / np.maximum(norm(item, p, axis, True), eps)
    elif is_torch(item):
        import torch.nn.functional as F

        return F.normalize(item, p, axis, eps)
    return item


def clip(item, a_min=None, a_max=None):
    if is_np_arr(item):
        return np.clip(item, a_min, a_max)
    elif is_torch(item):
        import torch

        return torch.clamp(item, a_min, a_max)


def to_gc(item, dim=None):
    """
    To generealized coordinates
    dim = 3 means transform 3-dim vectors to 4-dim vectors.
    """
    if dim is not None:
        assert item.shape[-1] == dim or item.shape[-1] == dim + 1
        if item.shape[-1] == dim + 1:
            return item
    return concat([item, ones_like(item[..., :1])], axis=-1)


def to_nc(item, dim=None):
    """
    To normal coordinates
    dim = 3 means transform 4-dim vectors to 3-dim vectors.
    """

    if dim is not None:
        assert item.shape[-1] == dim or item.shape[-1] == dim + 1
        if item.shape[-1] == dim:
            return item
    return item[..., :-1] / item[..., -1:]


def is_pcd(item, axis=-1):
    return item.shape[axis] == 3


def minimum(a, b):
    if is_np(a) and is_np(b):
        return np.minimum(a, b)
    elif is_torch(a) and is_torch(b):
        import torch

        return torch.minimum(a, b)
    else:
        raise ValueError(f"Bad inputs {type(a)} {type(b)}")


def broadcast_to(item, shape):
    if is_np_arr(item):
        return np.broadcast_to(item, shape)
    elif is_torch(item):
        return item.expand(shape)
    return item


def expand_as(item, other, exclude_axis=[]):
    if is_np_arr(item) or is_torch(item):
        assert item.ndim == other.ndim, f"{item.ndim}, {other.ndim}"
        other_shape = other.shape
        item_shape = item.shape
        rep_shape = [(other_shape[i] // item_shape[i] if i not in exclude_axis else 1) for i in range(item.ndim)]
        return repeat(item, rep=rep_shape)
    return item


def gather(item, axis, index):
    """
    Refer
        https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy
    """
    if is_np_arr(item):
        if item.ndim != index.ndim:
            return item
        index = expand_as(
            index,
            item,
            [
                axis,
            ],
        )
        index_xsec_shape = index.shape[:axis] + index.shape[axis + 1 :]
        item_xsec_shape = item.shape[:axis] + item.shape[axis + 1 :]
        if index_xsec_shape != item_xsec_shape:
            raise ValueError(f"Except for dimension {axis}, all dimensions of index and self should be the same size")
        data_swaped = np.swapaxes(item, 0, axis)
        index_swaped = np.swapaxes(index, 0, axis)
        gathered = np.choose(index_swaped, data_swaped)
        return np.swapaxes(gathered, 0, axis)
    elif is_torch(item):
        if item.ndim != index.ndim:
            return item
        import torch

        return torch.gather(
            item,
            axis,
            expand_as(
                index,
                item,
                [
                    axis,
                ],
            ),
        )
    else:
        return item


def batch_perm(item, axis=1, num_samples=None):
    # This is slow for large arries.
    if is_np_arr(item) or is_torch(item):
        assert axis > 0
        if num_samples is None:
            num_samples = item.shape[axis]
        num_samples = min(num_samples, item.shape[axis])
        shape = [item.shape[0], item.shape[axis]]
        if is_np_arr(item):
            index = np.argsort(np.random.rand(*shape), axis)
        else:
            import torch

            index = torch.rand(*shape, device=item.device).argsort(axis)
        index = index[:, :num_samples]
        rep = [
            1,
        ]
        for i in range(1, axis):
            index = index[..., None, :]
            rep.append(item.shape[i])
        rep.append(1)
        for i in range(axis + 1, item.ndim):
            index = index[..., None]
            rep.append(item.shape[i])
        index = repeat(index, rep, axis=None)
        return index
    else:
        return item


def batch_shuffle(item, axis=1, num_samples=None):
    """
    item [B, ...]
    For each item in batch, we use independently shuffle the items.
    = concat([shuffle(item[i], axis) for i in range(item.shape[0])], axis=0)
    """
    if is_np_arr(item) or is_torch(item):
        index = batch_perm(item, axis, num_samples)
        return gather(item, axis, index)
    else:
        return item


def clip_item(item, num, axis=1):
    if (is_np_arr(item) or is_torch(item)) and item.shape[axis] > num:
        item = take(item, slice(0, num), axis)
    return item


def pad_item(item, num, axis=1, pad_value=None):
    if (is_np_arr(item) or is_torch(item)) and item.shape[axis] < num:
        padded_shape = list(item.shape)
        padded_shape[axis] = num - padded_shape[axis]
        if is_not_null(pad_value):
            if is_np_arr(item):
                pad = np.full(padded_shape, pad_value, dtype=item.dtype)
            else:
                import torch

                pad = torch.ones(padded_shape, dtype=item.dtype, device=item.device)
        else:
            pad = repeat(take(item, range(1), axis), padded_shape[axis], axis)
        item = concat([item, pad], axis)
    return item


def pad_clip(item, num, axis=1, pad_value=None):
    item = pad_item(item, num, axis, pad_value)
    item = clip_item(item, num, axis)
    return item


def to_two_dims(item):
    if (is_np_arr(item) or is_torch(item)) and item.ndim == 1:
        return item[..., None]
    return item


def to_list(item):
    if is_np_arr(item) or is_torch(item):
        item = item.reshape(-1)
        item = [item[i] for i in range(item.shape[0])]
    return item


def allreduce(item, op="MEAN", device="cuda"):
    assert op in ["MEAN", "SUM", "AVG", "PRODUCT", "MIN", "MAX", "BAND", "BOR"]  # 'BXOR' is not supported for NCLL
    """Allreduce items.
    # allreduce is a inplaced operation for torch tensor.
    Args:
        items ([Number, numpy, tensor]): any numbers or tensors.
        coalesce (bool, optional): Whether allreduce parameters as a whole. Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB. Defaults to -1.
    """
    from ..torch import get_dist_info
    import torch
    from torch import distributed as dist
    from gdict.data import as_dtype

    _, world_size = get_dist_info()
    if world_size == 1:
        return item

    if is_num(item):
        data_type = ("number", type(item))
        item = torch.tensor(item, device=device)
    elif is_np(item):
        data_type = ("np", item.dtype)
        item = torch.tensor(item, device=device)
    elif is_torch(item):
        data_type = ("torch", item.dtype)
        item_device = item.device
        if item_device != device:
            item = item.to(device=device)
    else:
        return item
    if op == "BAND":
        op = "PRODUCT"
    elif op == "BOR":
        op = "SUM"

    item = item.double()

    if op == "MEAN":
        dist.all_reduce(item.div_(world_size), op=torch.distributed.ReduceOp.SUM)
    else:
        dist.all_reduce(item, op=getattr(torch.distributed.ReduceOp, op))

    if data_type[0] == "number":
        item = item > 0.5 if item is bool else data_type[1](item.item())
    elif data_type[0] == "np":
        item = item > 0.5 if data_type[1].name == "bool" else as_dtype(item.detach().cpu().numpy(), dtype=data_type[1])
    elif data_type[0] == "torch":
        item = item > 0.5 if "bool" in str(data_type[1]) else item.to(dtype=data_type[1])
        if item_device != device:
            item = item.to(device=item_device)
    return item


""" Torch only functions, which means it will works only for torch.Tensor  """


def detach(item):
    if is_torch(item):
        return item.detach()
    else:
        return item

def batch_index_select(input, index, axis):
    """Batch index_select

    Args:
        input (torch.Tensor): [B, ...]
        index (torch.Tensor): [B, N] or [B]
        dim (int): the dimension to index

    References:
        https://discuss.pytorch.org/t/batched-index-select/9115/7
        https://github.com/vacancy/AdvancedIndexing-PyTorch
    """
    import torch

    if index.dim() == 1:
        index = index.unsqueeze(1)
        squeeze_dim = True
    else:
        assert index.dim() == 2, "index is expected to be 2-dim (or 1-dim), but {} received.".format(index.dim())
        squeeze_dim = False
    assert input.size(0) == index.size(0), "Mismatched batch size: {} vs {}".format(input.size(0), index.size(0))
    views = [1 for _ in range(input.dim())]
    views[0] = index.size(0)
    views[axis] = index.size(1)
    expand_shape = list(input.shape)
    expand_shape[axis] = -1
    index = index.view(views).expand(expand_shape)
    out = torch.gather(input, axis, index)
    if squeeze_dim:
        out = out.squeeze(1)
    return out


""" Numpy only functions """


def encode_np(item):
    from gdict.file import dump

    item = dump(item, file_format="pkl")
    item = base64.binascii.b2a_base64(item)
    return item


def decode_np(item, dtype=None, shape_template=None):
    if is_num(shape_template):
        shape_template = (shape_template,)
    if isinstance(item, (bytes, np.void, str)):
        item = base64.binascii.a2b_base64(item)
        from gdict.file import load

        item = load(BytesIO(item), file_format="pkl")
        if is_not_null(shape_template):
            item = item.reshape(*shape_template)
    elif isinstance(item, np.ndarray) and True and get_dtype(item) == "object":
        item_shape = item.shape
        ret = [decode_np(item_i, dtype, shape_template) for item_i in item.reshape(-1)]
        item = np.array(ret, dtype=object)
        item = item.reshape(*item_shape)
    return item


def sample_and_pad(n, num=1200):
    index = np.arange(n)
    if n == 0:
        return np.zeros(num, dtype=np.int64)
    if index.shape[0] > num:
        np.random.shuffle(index)
        index = index[:num]
    elif index.shape[0] < num:
        num_repeat = num // index.shape[0]
        index = np.concatenate([index for i in range(num_repeat)])
        index = np.concatenate([index, index[: num - index.shape[0]]])
    return index
