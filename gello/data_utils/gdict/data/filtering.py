from .string_utils import regex_match
from .type_utils import is_dict, is_tuple_of, is_list_of


def custom_filter(item, func, value=True):
    """
    Recursively filter all elements with function func.
    Assumptions:
        None means the item does not pass func.
    """
    if is_tuple_of(item):
        item = list(item)
    if is_list_of(item):
        ret = []
        for i in range(len(item)):
            x = custom_filter(item[i], func, value)
            if x is not None:
                ret.append(x)
        item = ret
    elif is_dict(item):
        ret = {}
        for key in item:
            x = custom_filter(item[key], func, value)
            if x is not None:
                ret[key] = x
        item = ret
    return item if not value or (item is not None and func(item)) else None


def filter_none(x):
    func = lambda _: _ is not None
    return custom_filter(x, func, True)


def filter_with_regex(x, regex, value=True):
    func = lambda _: _ is not None and regex_match(_, regex)
    return custom_filter(x, func, value)
