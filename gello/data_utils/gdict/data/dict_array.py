"""
TODO: Merge or improved with pytree in jax.
"""

from collections import defaultdict
import numpy as np
from functools import wraps
from multiprocessing.shared_memory import SharedMemory

from .array_ops import (
    squeeze,
    unsqueeze,
    zeros_like,
    repeat,
    tile,
    shuffle,
    take,
    share_memory,
    concat,
    stack,
    arr_mean,
    to_item,
    select_with_mask,
    recover_with_mask,
    detach,
    get_nbytes,
    split,
    batch_shuffle,
    decode_np,
    to_two_dims,
    to_list,
    gather,
    reshape,
    transpose,
    contiguous,
    split_dim,
    to_item,
    to_cpu,
    to_cuda,
    allreduce,
    slice_item,
    deepcopy,
)
from .converter import as_dtype, to_np, to_torch, slice_to_range, to_array
from .type_utils import get_dtype, is_list_of, is_dict, is_h5, is_arr, is_num, is_np, is_str

SMM, use_shared_mem = None, False


def create_smm():
    global SMM, use_shared_mem
    if not use_shared_mem:
        from multiprocessing.managers import SharedMemoryManager

        use_shared_mem = True
        SMM = SharedMemoryManager()
        SMM.start()


def delete_smm():
    global SMM, use_shared_mem
    if use_shared_mem:
        use_shared_mem = False
        SMM.shutdown()


def replace_empty_with_none(*args):
    args = list(args)
    for i, x in enumerate(args):
        if x is not None and isinstance(x, (list, dict)) and len(x) == 0:
            x = None
        args[i] = x
    return args


def count_none(*args):
    ret = 0
    for _ in list(args):
        if _ is None:
            ret += 1
    return ret


def get_first_not_none(*args):
    for _ in list(args):
        if _ is not None:
            return _
    return None


class GDict:
    """
    Generalized Dict(GDict)
    Unified interface for dict, single element, HDF5 File.
    GDict are defined with syntax:
        GDict = GDict-Final | GDict-List | GDict-Dict
        GDict-Final = Any object not with type list, tuple, dict
        GDict-Dict or GDict-List = Dict or List of GDict

    Examples:
        1. GDict-Final:
           1) np-array: x = np.zeros(100)
           2) tensor: x = torch.tensor(100)
           3) HDF5 File: x = File('tmp.h5', 'r')
           4) Other python basic element: string, scalar, object.
        3. GDict-Dict or GDict-List or GDict-Tuple:
            GDict-Dict: x = {'0': {'b': np.zeros(100)}}
            GDict-List: x = [{'b': np.zeros(100)}, ]
                x['0/b'][0] = 1 (x['0/b/0'] is wrong!)
    Rules:
        1. No '\<>|:&?*"' in any keys (Compatible with filename rules in windows and unix)
           '/' is used to separate two keys between two layers.
        2. All integer key will be converted to string
        3. tuple object will be converted to list
        4. key does not contain any index in GDict-Final (See example 3)
        5. Rules for converting a GDict object to HDF5
            1) any number in keys of GDict-Dict will be converted to 'int_hdf5_' + number
            2) For GDict-List, the list will be converted to a dict with key 'list_int_hdf5_' + number
            3) GDict-Final:
                1) torch.Tensor will be converted to numpy array when is saved as HDF5 File and cannot be recovered.
                2) np.array will be saved as h5py.Dataset
                3) h5py object will be deep copied.
                4) other object will be serialized with pickle

    More Examples:
    >>> GDict(np.ones(3)).memory
    array([1., 1., 1.])
    >>> GDict(np.ones(3)).shape.memory
    3
    >>> d={'a': np.ones([1,1]), 'b': np.ones([2,3])}
    >>> GDict(d).memory
    {'a': array([[1.]]), 'b': array([[1., 1., 1.],
        [1., 1., 1.]])}
    >>> GDict(d).shape.memory
    {'a': (1, 1), 'b': (2, 3)}
    >>> l = [d,d]
    >>> GDict(l).memory
    [{'a': array([[1.]]), 'b': array([[1., 1., 1.],
           [1., 1., 1.]])}, {'a': array([[1.]]), 'b': array([[1., 1., 1.],
           [1., 1., 1.]])}]
    >>> GDict(l).shape.memory
    [{'a': (1, 1), 'b': (2, 3)}, {'a': (1, 1), 'b': (2, 3)}]
    """

    def __init__(self, item=None, faster=False, **kwargs):
        self.memory = item if faster else self.to_item(item)
        self.capacity = getattr(item, "capacity", None)

    @classmethod
    def _is_final(cls, item):
        return not isinstance(item, (list, dict))

    @classmethod
    def to_item(cls, item):
        if isinstance(item, GDict):
            return cls.to_item(item.memory)
        elif is_dict(item):
            ret = {key: cls.to_item(item[key]) for key in item}
            return ret
        elif isinstance(item, (list, tuple)):
            return [cls.to_item(x) for x in item]
        else:
            return item

    @classmethod
    def check_item(cls, item):
        if isinstance(item, dict):
            for key in item:
                if not cls.check_item(item[key]):
                    return False
        elif isinstance(item, list):
            for x in item:
                if not cls.check_item(x):
                    return False
        elif isinstance(item, (tuple, GDict)):
            return False
        return True

    @classmethod
    def assert_item(cls, item):
        assert cls.check_item(item), "Tuple and GDict should be missing in self.memory"

    @classmethod
    def _recursive_do_on_memory(cls, memory, function, new=True, ignore_list=False, *args, **kwargs):
        """Apply an operation to all elements in GDict. The operator can be functions in array_ops."""
        if isinstance(memory, dict):
            ret = {} if new else memory
            for key, value in memory.items():
                if cls._is_final(value):
                    ret[key] = function(value, *args, **kwargs)
                else:
                    ret[key] = cls._recursive_do_on_memory(memory[key], function, new, ignore_list, *args, **kwargs)
            return ret
        elif isinstance(memory, list) and not ignore_list:
            ret = [None for x in memory] if new else memory
            for key, value in enumerate(memory):
                if cls._is_final(value):
                    ret[key] = function(value, *args, **kwargs)
                else:
                    ret[key] = cls._recursive_do_on_memory(memory[key], function, new, ignore_list, *args, **kwargs)
            return ret
        else:
            return function(memory, *args, **kwargs)

    @classmethod
    def _recursive_do(cls, memory, function, new=True, wrapper=True, capacity=None, *args, **kwargs):
        item = cls._recursive_do_on_memory(memory, function, new, *args, **kwargs)
        return cls(item, capacity=capacity, faster=True) if wrapper else item

    @classmethod
    def _recursive_do_gdict(cls, memory, function, new=True, wrapper=True, *args, **kwargs):
        item = cls._recursive_do_on_memory(memory, function, new, *args, **kwargs)
        return GDict(item, faster=True) if wrapper else item

    @classmethod
    def _recursive_compare(cls, a, b, function):
        if isinstance(a, dict):
            inter_set = set(a.keys()) & set(b.keys())
            for key in inter_set:
                if not cls._recursive_compare(a[key], b[key], function):
                    return False
        elif isinstance(a, list):
            for i in range(min(len(a), len(b))):
                if not cls._recursive_compare(a[i], b[i], function):
                    return False
        else:
            return function(a, b)
        return True

    @classmethod
    def _get_item(cls, memory, keys):
        if len(keys) == 0 or memory is None:
            return memory
        elif is_dict(memory):
            key = keys[0]
            return cls._get_item(memory.get(key, None), keys[1:])
        elif is_list_of(memory):
            key = eval(keys[0])
            return cls._get_item(memory[key], keys[1:])
        else:
            print(f"Error! Keys should not cover the item in {type(memory)}, recent keys {keys}.")

    @classmethod
    def _set_item(cls, memory, keys, value):
        if isinstance(memory, GDict):
            memory = memory.memory
        if len(keys) == 0:
            return value
        elif is_dict(memory):
            key = keys[0]
            memory[key] = cls._set_item(memory.get(key, None), keys[1:], value)
        elif is_list_of(memory):
            key = eval(keys[0])
            if key > len(memory):
                for i in range(key - len(memory) + 1):
                    memory.append(None)
            memory[key] = cls._set_item(memory[key], keys[1:], value)
        else:
            print(f"Error! Keys should not cover the item in {type(memory)}, recent keys {keys}.")
        return memory

    @classmethod
    def _update_memory(cls, target, other):
        if is_list_of(target):
            if len(other) > len(target):
                for i in range(len(other) - len(target)):
                    target.append(None)
            for i in range(len(other)):
                target[i] = cls._update_memory(target[i], other[i])
        elif is_dict(target):
            for key in other:
                target[key] = cls._update_memory(target.get(key, None), other[key])
        else:
            target = other
        return target

    def update(self, other):
        if isinstance(other, GDict):
            other = other.memory
        self.memory = self._update_memory(self.memory, other)

    def compatible(self, other):
        if isinstance(other, GDict):
            other = other.memory

        def _compatible(a, b):
            return type(a) == type(b)

        return self._recursive_compare(self.memory, other, _compatible)

    def shared_memory(self, other):
        other = type(self)(other)
        return self._recursive_compare(self.memory, other.memory, share_memory)

    def copy(self, wrapper=True):
        return self._recursive_do(self.memory, deepcopy, wrapper=wrapper)

    def to_torch(self, use_copy=False, device="cpu", non_blocking=False, dtype=None, requires_grad=False, wrapper=True):
        return self._recursive_do(
            self.memory,
            to_torch,
            use_copy=use_copy,
            device=device,
            non_blocking=non_blocking,
            dtype=dtype,
            requires_grad=requires_grad,
            wrapper=wrapper,
        )

    def to_array(self, wrapper=True):
        return self._recursive_do(self.memory, to_array, wrapper=wrapper)

    def to_numpy(self, use_copy=False, dtype=None, wrapper=True):
        return self._recursive_do(self.memory, to_np, use_copy=use_copy, dtype=dtype, wrapper=wrapper)

    def to_hdf5(self, file):
        from gdict.file import dump_hdf5

        dump_hdf5(self.memory, file)

    @classmethod
    def from_hdf5(cls, file, wrapper=True):
        from gdict.file import load_hdf5

        ret = load_hdf5(file)
        if wrapper:
            ret = cls(ret)
        return ret

    @property
    def shape(self):
        def get_shape(x):
            shape = getattr(x, "shape", None)
            if shape is not None and len(shape) == 1:
                shape = shape[0]
            return shape

        return self._recursive_do_on_memory(self.memory, get_shape)

    @property
    def list_shape(self):
        def get_shape(x):
            shape = getattr(x, "shape", None)
            if shape is not None and len(shape) == 1:
                shape = shape[0]
            else:
                shape = list(shape)  # For torch.Size
            return shape

        return self._recursive_do_on_memory(self.memory, get_shape)

    @property
    def type(self):
        return self._recursive_do_on_memory(self.memory, type)

    @property
    def dtype(self):
        return self._recursive_do_on_memory(self.memory, get_dtype)

    @property
    def nbytes(self):
        return self._recursive_do_on_memory(self.memory, get_nbytes)

    @property
    def is_np(self):
        return self._recursive_do_on_memory(self.memory, is_np)

    @property
    def is_np_all(self):
        ret = self._flatten(self._recursive_do_on_memory(self.memory, is_np))
        return np.alltrue([v for k, v in ret.items()]) if isinstance(ret, dict) else ret

    @property
    def nbytes_all(self):
        ret = self._flatten(self._recursive_do_on_memory(self.memory, get_nbytes))
        return sum([v for k, v in ret.items()]) if isinstance(ret, dict) else ret

    @property
    def is_big(self):
        return self.nbytes_all / 1024 / 1024 > 1

    @property
    def device(self):
        def get_device(x):
            device = getattr(x, "device", None)
            if device is not None:
                device = f"{device.type}:{device.index}" if device.index is not None else f"{device.type}"
            return device

        return self._recursive_do_on_memory(self.memory, get_device)

    def cpu(self, wrapper=True):
        return self._recursive_do_gdict(self.memory, to_cpu, wrapper=wrapper)

    def cuda(self, device="cuda", wrapper=True):
        return self._recursive_do_gdict(self.memory, to_cuda, device=device, wrapper=wrapper)

    def item(self, wrapper=True):
        return self._recursive_do_gdict(self.memory, to_item, wrapper=wrapper)

    def item(self, wrapper=True):
        return self._recursive_do_gdict(self.memory, to_item, wrapper=wrapper)

    def astype(self, dtype, wrapper=True):
        return self._recursive_do(self.memory, as_dtype, dtype=dtype, wrapper=wrapper, capacity=self.capacity)

    def float(self, wrapper=True):
        return self.astype("float32", wrapper=wrapper)

    def f64_to_f32(self, wrapper=True):
        from .compression import f64_to_f32

        return self._recursive_do(self.memory, f64_to_f32, wrapper=wrapper, capacity=self.capacity)

    def squeeze(self, axis=None, wrapper=True):
        return self._recursive_do(self.memory, squeeze, axis=axis, wrapper=wrapper)

    def unsqueeze(self, axis, wrapper=True):
        return self._recursive_do(self.memory, unsqueeze, axis=axis, wrapper=wrapper,
                                  capacity=self.capacity if axis != 0 else 1)

    def detach(self, wrapper=True):
        return self._recursive_do(self.memory, detach, wrapper=wrapper, capacity=self.capacity)

    def to_zeros(self, wrapper=True):
        return self._recursive_do(self.memory, zeros_like, wrapper=wrapper, capacity=self.capacity)

    def repeat(self, rep, axis=None, wrapper=True):
        return self._recursive_do(
            self.memory, repeat, rep=rep, axis=axis, wrapper=wrapper,
            capacity=self.capacity if axis != 0 and axis is not None else None
        )

    def reshape(self, newshape, wrapper=True):
        return self._recursive_do(self.memory, reshape, newshape=newshape, wrapper=wrapper, capacity=newshape)

    def split_dim(self, axis, newaxes, wrapper=True):
        assert isinstance(newaxes, (list, tuple))
        return self._recursive_do(
            self.memory, split_dim, axis=axis, newaxes=newaxes, wrapper=wrapper,
            capacity=self.capacity if axis != 0 else newaxes[0]
        )

    def transpose(self, axis0, axis1, contiguous=True, wrapper=True):
        return self._recursive_do(
            self.memory,
            transpose,
            axis0=axis0,
            axis1=axis1,
            contiguous=contiguous,
            wrapper=wrapper,
            capacity=self.capacity if 0 not in [axis0, axis1] else None,
        )

    def contiguous(self, wrapper=True):
        return self._recursive_do(self.memory, contiguous, wrapper=wrapper, capacity=self.capacity)

    def tile(self, rep, wrapper=True):
        return self._recursive_do(self.memory, tile, rep=rep, wrapper=wrapper)

    def mean(self, axis=None, keepdim=False, wrapper=True):
        return self._recursive_do(
            self.memory, arr_mean, axis=axis, keepdim=keepdim, wrapper=wrapper,
            capacity=self.capacity if axis != 0 and axis is not None else None
        )

    @classmethod
    def _assign(cls, memory, indices, value, ignore_list=False):
        if isinstance(value, tuple):
            value = list(value)
        if is_dict(memory):
            assert type(memory) == type(value), f"{type(memory), type(value)}"
            for key in memory:
                if key in value:
                    memory[key] = cls._assign(memory[key], indices, value[key], ignore_list)
        elif is_arr(memory):
            assert type(memory) == type(value) or np.isscalar(value), f"{type(memory), type(value)}"
            if share_memory(memory, value):
                memory[indices] = deepcopy(value)
            else:
                memory[indices] = value
        elif is_list_of(memory):
            if ignore_list:
                memory[indices] = value
            else:
                # if is_num(indices):
                #     memory[indices] = value if is_num(value) else value[indices]
                # else:
                #     assert type(memory) == type(value), f"{type(memory), type(value)}"
                for i in range(min(len(memory), len(value))):
                    memory[i] = cls._assign(memory[i], indices, value[i], ignore_list)
        return memory

    def assign_list(self, index, value):
        if isinstance(value, GDict):
            value = value.memory
        assert is_num(index)
        self.memory = self._assign(self.memory, index, value, True)

    def to_two_dims(self, wrapper=True):
        return self._recursive_do(self.memory, to_two_dims, wrapper=wrapper)

    def take_list(self, index, wrapper=True):
        assert is_num(index)
        return self._recursive_do_gdict(self.memory, take, indices=index, axis=0, ignore_list=True, wrapper=wrapper)

    def to_list(self, wrapper=True):
        return self._recursive_do(self.memory, to_list, wrapper=wrapper)

    def select_with_mask(self, mask, wrapper=True):
        return self._recursive_do(self.memory, select_with_mask, mask=mask, wrapper=wrapper,
                                  capacity=to_item(mask.sum()))

    def recover_with_mask(self, mask, wrapper=True):
        return self._recursive_do(self.memory, select_with_mask, mask=mask, wrapper=wrapper, capacity=mask.shape[0])

    def allreduce(self, op="MEAN", device="cuda", wrapper=True):
        return self._recursive_do(self.memory, allreduce, op=op, device=device, wrapper=wrapper, capacity=self.capacity)

    def to_gdict(self):
        return GDict(self.memory, faster=True)

    @property
    def one_device(self):
        return self._get_one_attr(self.memory, "device")

    @property
    def one_shape(self):
        return self._get_one_attr(self.memory, "shape")

    @property
    def one_dtype(self):
        return self._get_one_attr(self.memory, "dtype")

    def _flatten(cls, memory, root_key="", full=True):
        if is_dict(memory):
            ret = {}
            for key in memory:
                ret.update(cls._flatten(memory[key], f"{root_key}/{key}", full))
        elif is_list_of(memory) and (full or len(memory) > 10):
            # Simplify flatten result for small list or tuple
            ret = {}
            for i in range(len(memory)):
                ret.update(cls._flatten(memory[i], f"{root_key}/{i}", full))
        else:
            return memory if root_key == "" else {root_key.replace("//", "/"): memory}
        return ret

    def flatten(self, full=True):
        return type(self)(self._flatten(self.memory, "", full))

    @classmethod
    def wrapper(cls, class_method=False):
        if not class_method:

            def decorator(func):
                @wraps(func)
                def wrapper(item, *args, **kwargs):
                    if isinstance(item, GDict):
                        return func(item, *args, **kwargs)
                    else:
                        return func(GDict(item), *args, **kwargs).memory

                return wrapper

        else:

            def decorator(func):
                @wraps(func)
                def wrapper(self, item, *args, **kwargs):
                    if isinstance(item, GDict):
                        return func(self, item, *args, **kwargs)
                    else:
                        return func(self, GDict(item), *args, **kwargs).memory

                return wrapper

        return decorator

    def select_by_keys(self, keys=None, to_list=False, wrapper=True):
        def _dfs_select(memory, keys=None):
            if keys is None:
                return memory
            if isinstance(memory, dict):
                new_keys = {}
                for key in keys:
                    fk = key[0]
                    if len(key) > 1:
                        if fk not in new_keys:
                            new_keys[fk] = []
                        new_keys[fk].append(key[1:])
                    else:
                        new_keys[fk] = None
                return {key: _dfs_select(memory[key], new_keys[key]) for key in new_keys}
            elif isinstance(memory, list):
                new_keys = {}
                for key in keys:
                    fk = eval(key[0]) if is_str(key[0]) else key[0]
                    if len(key) > 1:
                        if fk not in new_keys:
                            new_keys[fk] = []
                        new_keys[fk].append(key[1:])
                    else:
                        new_keys[fk] = None
                return [_dfs_select(memory[key], new_keys[key]) for key in sorted(new_keys)]
            else:
                raise ValueError(f"{keys}")

        if not isinstance(keys, (list, tuple)) and keys is not None:
            keys = [keys]
            single = True
        else:
            single = False
        keys = [self._process_key(key) for key in keys]
        memory = _dfs_select(self.memory, keys)
        if to_list:
            memory = type(self)(memory)
            memory = [memory[key] for key in keys]
            if single:
                memory = memory[0]
        if wrapper:
            memory = type(self)(memory)
        return memory

    def take(self, indices, axis=0, wrapper=True):  # will always copy data, needs double check
        if is_num(indices):
            return self._recursive_do_gdict(self.memory, take, indices=indices, axis=axis, wrapper=wrapper)
        else:

            if isinstance(indices, slice):
                len_indices = len(slice_to_range(indices))
            else:
                len_indices = len(indices)
            new_capacity = len_indices if axis == 0 else self.capacity
            return self._recursive_do(self.memory, take, indices=indices, axis=axis, wrapper=wrapper,
                                      capacity=new_capacity)

    def slice(self, slice, axis=0, wrapper=True):  # no copy
        return self._recursive_do(self.memory, slice_item, slice=slice, axis=axis, wrapper=wrapper)

    def assign_all(self, value):
        if isinstance(value, GDict):
            value = value.memory
        self.memory = self._assign(self.memory, slice(None, None, None), value)

    @classmethod
    def _do_on_list_of_array(cls, memories, function, **kwargs):
        for i in range(len(memories)):
            assert type(memories[i]) is type(memories[0]), f"{type(memories[i]), type(memories[0])}"
        if isinstance(memories[0], (tuple, list)):
            for i in range(len(memories)):
                assert len(memories[i]) == len(memories[0])
            ret = []
            for i in range(len(memories[0])):
                ret.append(cls._do_on_list_of_array([memories[j][i] for j in range(len(memories))], function, **kwargs))
        elif isinstance(memories[0], dict):
            for i in range(len(memories)):
                assert set(memories[i].keys()) == set(
                    memories[0].keys()), f"{set(memories[i].keys())}, {set(memories[0].keys())}"
            ret = {}
            for key in memories[0]:
                ret[key] = cls._do_on_list_of_array([memories[j][key] for j in range(len(memories))], function,
                                                    **kwargs)
        else:
            ret = function(memories, **kwargs)
        return ret

    @classmethod
    def concat(cls, items, axis=0, wrapper=True):
        ret = cls._do_on_list_of_array([_.memory if isinstance(_, GDict) else _ for _ in items], concat, axis=axis)
        if wrapper:
            capacity = 0
            for item in items:
                if isinstance(item, GDict) and item.capacity is not None:
                    capacity += item.capacity
                else:
                    capacity = None
                    break
            return cls(ret, capacity=capacity, faster=True)
        else:
            return ret

    @classmethod
    def stack(cls, items, axis=0, wrapper=True):
        ret = cls._do_on_list_of_array([_.memory if isinstance(_, GDict) else _ for _ in items], stack, axis=axis)
        if wrapper:
            if axis == 0:
                capacity = len(items)
            else:
                capacity = None
                for item in items:
                    if isinstance(item, cls) and item.capacity is not None:
                        capacity = item.capacity
                        break
            return cls(ret, capacity=capacity, faster=True)
        else:
            return ret

    @classmethod
    def _process_key(cls, key):
        if is_num(key):
            key = str(key)
        return key if isinstance(key, (list, tuple)) else key.strip("/").replace("//", "/").split("/")

    def __getitem__(self, key):
        return self._get_item(self.memory, self._process_key(key))

    def __setitem__(self, key, value):
        self.memory = self._set_item(self.memory, self._process_key(key), value)
        return self.memory

    def __str__(self):
        return str(self._flatten(self.memory, "", False))

    def __dict__(self):
        assert isinstance(self.memory, dict), "self.memory is not a dict!"
        return self.memory

    def __getattr__(self, key):
        if key == 'memory':
            assert False, "GDict should always have a memory attribute!"
        return getattr(self.memory, key)

    def __getstate__(self):
        return self.memory

    def __setstate__(self, state):
        self.memory = state

    def __contains__(self, key):
        if "/" in key:
            key = self._process_key(key)
            memory = self.memory
            for _ in key:
                if _ not in memory:
                    return False
                memory = memory[_]
            return True
        else:
            return key in self.memory

    def __delitem__(self, key):
        keys = list(self._process_key(key))
        last_memory = None
        memory = self.memory
        for i, key in enumerate(keys):
            if isinstance(last_memory, list) and isinstance(key, str):
                key = eval(key)
                keys[i] = key
            last_memory = memory
            memory = memory[key]

        if last_memory is None:
            self.memory = None
        elif isinstance(last_memory, (dict, list)):
            last_memory.pop(key)


class DictArray(GDict):
    """
    DictArray is a special GDict which requires the first dimension of all GDict-Final must be same
    """

    def __init__(self, item=None, capacity=None, faster=False):
        super(DictArray, self).__init__(item, faster=faster)
        if item is None:
            self.capacity = None
            return
        if capacity is not None:
            self.capacity = capacity
            if not faster:
                self.memory = self.to_array(wrapper=False)
                self.memory = self.unsqueeze(axis=0, wrapper=False)  # .to_zeros(wrapper=False)
                if capacity != 1:
                    self.memory = self.repeat(capacity, axis=0, wrapper=False)
        elif self.capacity is None:
            self.capacity = self._get_one_attr(self.memory, "shape")[0]
        if not faster:
            self.assert_shape(self.memory, self.capacity)

    @classmethod
    def _get_one_attr(cls, memory, attr):
        # print(type(memory), attr)
        if isinstance(memory, dict):
            for key in memory:
                if hasattr(memory[key], attr):
                    return getattr(memory[key], attr)
                ans = cls._get_one_attr(memory[key], attr)
                if ans is not None:
                    return ans
        elif isinstance(memory, list):
            for x in memory:
                if hasattr(x, attr):
                    return getattr(x, attr)
                ans = cls._get_one_attr(x, attr)
                if ans is not None:
                    return ans
        elif hasattr(memory, attr):
            return getattr(memory, attr)
        return None

    @classmethod
    def check_shape(cls, memory, capacity):
        if isinstance(memory, dict):
            for key in memory:
                if not cls.check_shape(memory[key], capacity):
                    return False
        elif isinstance(memory, list):
            for x in memory:
                if not cls.check_shape(x, capacity):
                    return False
        elif hasattr(memory, "shape"):
            return memory.shape[0] == capacity
        return True

    @classmethod
    def assert_shape(cls, memory, capacity):
        assert cls.check_shape(memory, capacity), f"The first dimension is not {capacity}!"

    def sample(self, batch_size, valid_capacity=None, wrapper=True):
        capacity = self.capacity if valid_capacity is None else valid_capacity
        indices = np.random.randint(low=0, high=capacity, size=batch_size)
        return self._recursive_do(self.memory, take, indices=indices, axis=0, wrapper=wrapper, capacity=batch_size)

    def shuffle(self, valid_capacity=None, wrapper=True, in_place=True):
        capacity = self.capacity if valid_capacity is None else valid_capacity
        indices = shuffle(np.arange(capacity), axis=0)
        # print(valid_capacity, self.capacity)
        # print(np.unique(indices).shape, len(indices))
        # exit(0)
        # print(capacity, self.capacity)
        if in_place:
            # print(indices)
            items = self.take(slice(0, capacity), wrapper=False)
            # print(items.shape, share_memory(items['actions'], self.memory['actions']))
            self.assign(indices, items)
            # self._recursive_do(self.memory, take, indices=indices, axis=0, wrapper=False, capacity=self.capacity)
        else:
            if capacity < self.capacity:
                indices = np.concatenate([indices, np.arange(self.capacity - capacity) + capacity], axis=0)
            return self._recursive_do(self.memory, take, indices=indices, axis=0, wrapper=wrapper,
                                      capacity=self.capacity)

    def assign(self, indices, value):
        if isinstance(value, GDict):
            value = value.memory
        self.memory = self._assign(self.memory, indices, value)

    def gather(self, axis, index, wrapper=True):
        return self._recursive_do(self.memory, gather, axis=axis, index=index, wrapper=wrapper)

    def to_dict_array(self):
        return DictArray(self.memory, capacity=self.capacity, faster=True)

    def __len__(self):
        return self.capacity


class SharedGDict(GDict):
    def __init__(self, gdict=None, shape=None, dtype=None, name=None):
        if gdict is not None:
            assert shape is None and dtype is None and name is None
            assert isinstance(gdict, GDict) and gdict.is_np_all
            shape = gdict.shape
            dtype = gdict.dtype
            nbytes = gdict.nbytes
        else:
            assert not (shape is None or dtype is None or name is None)
            nbytes = None

        self.is_new = name is None

        name, self.shared_memory = self._create_shared_memory(shape, dtype, nbytes, name)
        memory = self._create_np_from_memory(self.shared_memory, shape, dtype)

        self.shared_shape = shape
        self.shared_dtype = dtype
        self.shared_name = name

        super(SharedGDict, self).__init__(memory)

    def _create_np_from_memory(cls, shared_memory, shape, dtype):
        if isinstance(shared_memory, dict):
            memory = {k: cls._create_np_from_memory(shared_memory[k], shape[k], dtype[k]) for k in shared_memory}
        elif isinstance(shared_memory, list):
            memory = [cls._create_np_from_memory(shared_memory[k], shape[k], dtype[k]) for k in
                      range(len(shared_memory))]
        else:
            if isinstance(dtype, str):
                dtype = np.dtype(dtype)
            memory = np.ndarray(shape, dtype=dtype, buffer=shared_memory.buf)
        return memory

    def _create_shared_memory(cls, shape, dtype, nbytes, name=None):
        if name is None:
            # Create new shared buffer
            if isinstance(nbytes, dict):
                ret_name, ret_memory = {}, {}
                for key in nbytes:
                    name_k, memory_k = cls._create_shared_memory(shape[key], dtype[key], nbytes[key], None)
                    ret_name[key] = name_k
                    ret_memory[key] = memory_k
            elif isinstance(nbytes, (list, tuple)):
                ret_name, ret_memory = [], []
                for key in range(len(nbytes)):
                    name_k, memory_k = cls._create_shared_memory(shape[key], dtype[key], nbytes[key], None)
                    ret_name.append(name_k)
                    ret_memory.append(memory_k)
            else:
                assert is_num(nbytes), f"{nbytes}"
                ret_memory = SharedMemory(size=nbytes, create=True)
                ret_name = ret_memory.name
        else:
            ret_name = name
            if isinstance(name, dict):
                ret_memory = {k: cls._create_shared_memory(shape[k], dtype[k], None, name[k])[1] for k in name}
            elif isinstance(name, (list, tuple)):
                ret_memory = [cls._create_shared_memory(shape[k], dtype[k], None, name[k])[1] for k in range(len(name))]
            else:
                assert isinstance(name, str), f"{name}"
                ret_memory = SharedMemory(name=name, create=False)
        return ret_name, ret_memory

    def get_infos(self):
        return self.shared_shape, self.shared_dtype, self.shared_name

    def _unlink(self):
        memory = self._flatten(self.shared_memory)
        if isinstance(memory, dict):
            for k, v in memory.items():
                v.unlink()
        else:
            memory.unlink()

    def _close(self):
        memory = self._flatten(self.shared_memory)
        if isinstance(memory, dict):
            for k, v in memory.items():
                v.close()
        elif not callable(memory):
            memory.close()

    def __del__(self):
        self._close()
        if self.is_new:
            self._unlink()

    def get_full_by_key(self, key):
        ret = []
        for name in ["shared_shape", "shared_dtype", "shared_name"]:
            ret.append(self._get_item(getattr(self, name), self._process_key(key)))
        return type(self)(None, *ret)

    def __setitem__(self, key, value):
        assert False, "Please convert to GDict or Dictarray then change the value!"


class SharedDictArray(SharedGDict, DictArray):
    pass
