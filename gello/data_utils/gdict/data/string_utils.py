"""
Useful regex expression
    1. nothing else classifier: '^((?!classifier).)*$'
    2. any string: '(.*?)'
"""

import re


any_string = r"(.*?)"


def custom_format(template_string, **kwargs):
    template_string = template_string.replace("{", "{{")
    template_string = template_string.replace("}", "}}")
    template_string = template_string.replace("&lformat ", "{")
    template_string = template_string.replace(" &rformat", "}")
    return template_string.format_map(kwargs)


def regex_match(string, pattern):
    return re.match(pattern, string) is not None


def regex_replace(string, pattern, new_pattern):
    return re.sub(pattern, new_pattern, string)


def prefix_match(string, prefix=None):
    """Check if the string matches the given prefix"""
    if prefix is None or len(prefix) == 0:
        return True
    return re.match(f"({prefix})+(.*?)", string) is not None


def is_regex(s):
    try:
        re.compile(s)
        return True
    except:
        return False


def float_str(num, precision):
    format_str = "%.{0}f".format(precision)
    return format_str % num


def num_to_str(num, unit=None, precision=2, number_only=False, auto_select_unit=False):
    unit_list = ["K", "M", "G", "T", "P"]
    if auto_select_unit and unit is None:
        for i, tmp in enumerate(unit_list):
            unit_num = 1024 ** (i + 1)
            if num < unit_num:
                break
            unit = tmp
    if unit is not None:
        unit_num = 1024 ** (unit_list.index(unit) + 1)
        num = num * 1.0 / unit_num
    else:
        unit = ""
    if number_only:
        return num
    else:
        return float_str(num, precision) + unit
