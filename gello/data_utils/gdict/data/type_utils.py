from collections.abc import Sequence
from numbers import Number
import numpy as np


"""  For python basic type  """


def is_null(item):
    return item is None


def is_not_null(item):
    return item is not None


def is_slice(item):
    return isinstance(item, slice)


def is_str(item):
    return isinstance(item, str)


def is_dict(item):
    return isinstance(item, dict)


def is_num(item):
    return isinstance(item, Number)


def is_integer(item):
    return isinstance(item, (int, np.integer))


def is_type(item):
    return isinstance(item, type)


def is_seq_of(seq, expected_type=None, seq_type=None):
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    if expected_type:
        for item in seq:
            if not isinstance(item, expected_type):
                return False
    return True


def is_list_of(seq, expected_type=None):
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type=None):
    return is_seq_of(seq, expected_type, seq_type=tuple)


def is_iterable(item):
    return isinstance(item, (dict, tuple, list))


"""  For numpy and torch type  """


def get_dtype(item):
    if isinstance(item, (list, tuple)):
        item = item[0]
    if hasattr(item, "dtype"):
        return str(item.dtype).split(".")[-1]
    elif isinstance(item, (int, float, bytes, str)):
        return type(item)
    else:
        return None


def is_np(item):
    return isinstance(item, np.ndarray) or is_num(item)


def is_np_arr(item):
    return isinstance(item, np.ndarray)


def is_torch(item):
    import torch

    return isinstance(item, torch.Tensor)


def is_torch_distribution(item):
    import torch

    return isinstance(item, torch.distributions.Distribution)


def is_arr(item, arr_type=None):
    if is_num(item):
        return False
    if arr_type is not None:
        assert arr_type in ["np", "torch"]
        return eval(f"is_{arr_type}")(item)
    elif is_np(item):
        return True
    else:
        # Torch as the last option to reduce memory usage
        return is_torch(item)


"""  For HDF5 type  """


def is_h5(item):
    from h5py import File, Group, Dataset

    return isinstance(item, (File, Group, Dataset))
