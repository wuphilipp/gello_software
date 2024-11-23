import numpy as np, base64
from .dict_array import GDict
from .array_ops import encode_np, decode_np
from .converter import as_dtype
from .type_utils import is_np_arr, get_dtype, is_dict, is_not_null, is_null, is_seq_of

def float_to_int(data, vrange=[0.0, 1.0], res=None, dtype="uint8"):
    data_dtype = get_dtype(data)
    if "int" in data_dtype:
        return as_dtype(data, dtype) if data_dtype != dtype else data
    assert data_dtype.startswith("float"), f"{type(data), data}"
    min_v = np.iinfo(getattr(np, dtype)).min
    max_v = np.iinfo(getattr(np, dtype)).max
    if is_not_null(vrange):
        assert vrange[0] < vrange[1] and is_null(res)
        data = (np.clip(data, a_min=vrange[0], a_max=vrange[1]) - vrange[0]) / (vrange[1] - vrange[0])  # Normalize value to [0, 1]
        data = data * max_v + (1 - data) * min_v
    else:
        assert is_not_null(res)
        data = data / res

    data = as_dtype(np.clip(data, a_min=min_v, a_max=max_v), dtype)
    return data


def int_to_float(data, vrange=[0.0, 1.0], res=None, *dtype):
    data_dtype = get_dtype(data)
    if data_dtype == "object":
        assert data.shape == (1,)
        data = data[0]
    elif data_dtype.startswith("float"):
        return as_dtype(data, dtype) if data_dtype != dtype else data

    data_dtype = get_dtype(data)

    assert data_dtype.startswith("int") or data_dtype.startswith("uint"), f"{data_dtype}"
    min_v = np.float32(np.iinfo(getattr(np, data_dtype)).min)
    max_v = np.float32(np.iinfo(getattr(np, data_dtype)).max)
    if is_not_null(vrange):
        assert vrange[0] < vrange[1] and is_null(res)
        data = (data - min_v) / (max_v - min_v)  # [0, 1]
        data = data * np.float32(vrange[1]) + (1 - data) * np.float32(vrange[0])
    else:
        assert is_not_null(res)
        res = np.float32(res)
        data = data * res
    return as_dtype(data, "float32")


def f64_to_f32(item):
    """
    Convert all float64 data to float32
    """
    from .type_utils import get_dtype
    from .converter import as_dtype

    sign = get_dtype(item) in ["float64", "double"]
    return as_dtype(item, "float32") if sign else item


def to_f32(item):
    return as_dtype(item, "float32")


def to_f16(item):
    return as_dtype(item, "float16")
