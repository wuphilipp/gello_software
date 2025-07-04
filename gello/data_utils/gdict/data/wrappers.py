from functools import wraps

from numbers import Number
from .converter import to_np, to_torch, to_array


def seq_to_np(to_arr=True):
    def decorator(func):
        @wraps(func)
        def wrapper(item, *args, **kwargs):
            if isinstance(item, (list, tuple, Number)):
                item = to_np(item)
            if to_arr:
                item = to_array(item)
            return func(item, *args, **kwargs)

        return wrapper

    return decorator


def check_consistent(keys, dtypes):
    if dtypes:
        dtypes = (dtypes,) if not isinstance(dtypes, (list, tuple)) else dtypes
    if keys:
        keys = (keys,) if not isinstance(keys, (list, tuple)) else keys
        if dtypes:
            assert len(keys) == len(dtypes)
    else:
        if dtypes:
            assert len(dtypes) == 1


def apply_func(func, x):
    if isinstance(x, (list, tuple, set)):
        return type(x)(map(func, x))
    elif isinstance(x, dict):
        for k in x:
            x[k] = func(x[k])
        return x
    else:
        return func(x)


def change_dtype(x, keys=None, dtypes=None, np=False):
    if dtypes is None:
        return x
    processor = to_np if np else to_torch
    if not isinstance(dtypes, (list, tuple)):
        dtypes = [dtypes]

    if not isinstance(x, (tuple, list, dict)) or keys is None:
        assert len(dtypes) == 1
        return processor(x, dtypes[0])

    if not isinstance(keys, (list, tuple)):
        keys = [keys]
    # key and dtypes are list or tuple, dtypes is a list, x is a list, tuple or dict

    ret = list(x) if isinstance(x, (list, tuple)) else x
    if len(dtypes) == 1:
        dtypes = [dtypes[0] for i in range(len(keys))]
    for k, dtype in enumerate(keys, dtypes):
        ret[k] = processor(ret[k], dtype)
    return type(x)(ret)


def process_output(keys=None, dtypes=None, np=True):
    check_consistent(keys, dtypes)

    def decorator(func):
        wraps(func)

        def wrapper(*args, **kwargs):
            ret = func(*args, **kwargs)
            return change_dtype(ret, keys, dtypes, np)

        return wrapper

    return decorator


def process_input(keys=None, dtypes=None, np=True):
    check_consistent(keys, dtypes)

    def decorator(func):
        wraps(func)

        def wrapper(*args, **kwargs):
            args = list(args)
            kwargs = dict(kwargs)
            args = change_dtype(args, keys, dtypes, np)
            kwargs = change_dtype(kwargs, keys, dtypes, np)
            return func(*args, **kwargs)

        return wrapper

    return decorator
