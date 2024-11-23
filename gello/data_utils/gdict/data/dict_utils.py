from copy import deepcopy
from .type_utils import is_dict, is_seq_of


def update_dict(x, y):
    """
    Update x with y
    """
    assert type(x) == type(y), f"{type(x), type(y)}"
    if is_dict(x):
        ret = deepcopy(x)
        for key in y:
            if key in x:
                ret[key] = update_dict(x[key], y[key])
            else:
                ret[key] = deepcopy(y[key])
    else:
        ret = deepcopy(y)
    return ret


def update_dict_with_begin_keys(x, y, keys, begin=False, history_key=()):
    if len(keys) == 0:
        if type(x) == type(y):
            return update_dict(x, y)
        elif is_seq_of(x, dict) and is_dict(y):
            return [update_dict(_, y) for _ in x]
        else:
            raise NotImplementedError()
    if not is_dict(x):
        return deepcopy(x)

    ret = {}
    for key in x:
        if key == keys[0]:
            ret[key] = update_dict_with_begin_keys(x[key], y, keys[1:], True, history_key + (key,))
        elif not begin:
            ret[key] = update_dict_with_begin_keys(x[key], y, keys, False, history_key + (key,))
        else:
            ret[key] = deepcopy(x[key])
    return ret


def first_dict_key(item):
    return sorted(item.keys())[0]


def map_dict_keys(inputs, keys_map, logger_print=None):
    from .string_utils import regex_replace, regex_match, is_regex
    import re

    outputs = {}
    for key, value in inputs.items():
        new_key = key
        for in_pattern, out_pattern in keys_map.items():
            if regex_match(key, in_pattern):
                new_key = regex_replace(key, in_pattern, out_pattern)
                break
        if new_key == "None" or new_key is None:
            if logger_print is not None:
                logger_print(f"Delete {key}!")
                continue
        if new_key != key and logger_print is not None:
            logger_print(f"Change {key} to {new_key}.")
        outputs[new_key] = value
    return outputs
