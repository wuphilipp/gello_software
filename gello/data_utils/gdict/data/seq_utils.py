import itertools
from copy import deepcopy
from random import shuffle
from .type_utils import is_seq_of


def concat_seq(in_list, dtype):
    assert dtype in [list, tuple]
    return dtype(itertools.chain(*in_list))


def concat_list(in_list):
    return concat_seq(in_list, list)


def concat_tuple(in_list):
    return concat_seq(in_list, tuple)


def auto_pad_seq(a, b):
    """
    Input two sequence, then output two list of objects with the same size.
    """
    a = list(a) if isinstance(a, (list, tuple)) else [a]
    b = list(b) if isinstance(b, (list, tuple)) else [b]
    if len(a) > len(b):
        for i in range(len(a) - len(b)):
            b.append(a[0])
    elif len(a) < len(b):
        for i in range(len(b) - len(a)):
            a.append(b[0])
    return a, b


def flatten_seq(x, dtype=list):
    if not is_seq_of(x, (tuple, list)):
        return x
    return dtype(concat_list([flatten_seq(_) for _ in x]))


def split_list_of_parameters(num_procsess, *args, **kwargs):
    from ..math import split_num

    args = [_ for _ in args if _ is not None]
    kwargs = {_: __ for _, __ in kwargs.items() if __ is not None}
    assert len(args) > 0 or len(kwargs) > 0
    first_item = args[0] if len(args) > 0 else kwargs[list(kwargs.keys())[0]]
    n, running_steps = split_num(len(first_item), num_procsess)
    start_idx = 0
    paras = []
    for i in range(n):
        slice_i = slice(start_idx, start_idx + running_steps[i])
        start_idx += running_steps[i]
        args_i = list([_[slice_i] for _ in args])
        kwargs_i = {_: kwargs[_][slice_i] for _ in kwargs}
        paras.append([args_i, kwargs_i])
    return paras


def select_by_index(files, indices):
    return [files[i] for i in indices]


def random_pad_clip_list(x, num):
    x = deepcopy(list(x))
    if len(x) > num:
        shuffle(x)
        return x[:num]
    else:
        ret = []
        for i in range(num // len(x)):
            shuffle(x)
            ret = ret + x
        ret = ret + x[: num - len(ret)]
        return ret
