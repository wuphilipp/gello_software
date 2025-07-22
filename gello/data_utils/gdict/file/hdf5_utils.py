from h5py import File, Group, Dataset
import numpy as np

from ..data import is_h5, is_list_of, is_dict, is_arr, to_np, is_str, is_not_null


def load_hdf5(file, keys=None):
    """
    Load all elements in HDF5
    """

    def _load_hdf5(file, load_keys, only_one):
        only_one = only_one and load_keys is not None
        if is_not_null(load_keys):
            new_keys = {}
            for key in load_keys:
                if key[0] not in new_keys:
                    new_keys[key[0]] = []
                if len(key) > 1:
                    new_keys[key[0]].append(key[1:])
            load_keys = {key: (None if len(item) == 0 else item) for key, item in new_keys.items()}
        if isinstance(file, (File, Group)):
            keys = list(file.keys())
            if keys[0].startswith("list"):
                ret = []
                for key in range(len(keys)):
                    if is_not_null(load_keys) and f"{key}" not in load_keys:
                        continue
                    load_keys_i = load_keys[f"{key}"] if is_not_null(load_keys) else None
                    key = f"list_{type(key).__name__}_{key}"
                    ret.append(_load_hdf5(file[key], load_keys_i, only_one))
                ret = ret[0] if only_one else ret
            elif keys[0].startswith("dict"):
                ret = {}
                for key in keys:
                    if key.startswith("dict"):
                        key_type = eval(key.split("_")[1])
                        key_value = key_type(key[len(f"dict_{key.split('_')[1]}_") :])
                    else:
                        key_value = key
                    if is_not_null(load_keys) and f"{key_value}" not in load_keys:
                        continue
                    load_keys_i = load_keys[f"{key_value}"] if is_not_null(load_keys) else None
                    ret[key_value] = _load_hdf5(file[key], load_keys_i, only_one)
                ret = ret[list(ret.keys())[0]] if only_one and len(ret) > 0 else ret
            elif len(keys) == 1 and keys[0] == "GDict":
                ret = _load_hdf5(file["GDict"], load_keys, only_one)
            else:
                ret = {}
                for key in keys:
                    if key.startswith("int__"):
                        key_value = key[len("int__") :]
                    else:
                        key_value = key
                    # print(key_value, load_keys, key_value in load_keys)
                    if is_not_null(load_keys) and f"{key_value}" not in load_keys:
                        continue
                    load_keys_i = load_keys[f"{key_value}"] if is_not_null(load_keys) else None
                    ret[key_value] = _load_hdf5(file[key], load_keys_i, only_one)
                ret = ret[list(ret.keys())[0]] if only_one and len(ret) > 0 else ret
            return ret
        elif isinstance(file, Dataset):
            assert load_keys is None or len(load_keys) == 0, f"{load_keys}"
            ret = file[()]
            if isinstance(ret, np.void):
                from .serialization import load
                from io import BytesIO

                return load(BytesIO(ret), file_format="pkl")
            else:
                return ret

    if is_str(keys):
        keys = [keys]
        only_one = True
    else:
        only_one = False
    if is_not_null(keys):
        keys = [key.strip("/").replace("//", "/").split("/") for key in keys]
    if not is_h5(file):
        file = File(file, "r")
        ret = _load_hdf5(file, keys, only_one)
        file.close()
    else:
        ret = _load_hdf5(file, keys, only_one)
    return ret


def dump_hdf5(obj, file):
    def _dump_hdf5(memory, file, root_key=""):
        if isinstance(memory, (list, dict)):
            keys = range(len(memory)) if is_list_of(memory) else memory.keys()
            for key in keys:
                _dump_hdf5(memory[key], file, f"{root_key}/{type(memory).__name__}_{type(key).__name__}_{key}")
        else:
            root_key = root_key.replace("//", "/") if root_key != "" else "GDict"
            if is_arr(memory):
                memory = to_np(memory)
                file[root_key] = memory
            else:
                from .serialization import dump

                file[root_key] = np.void(dump(memory, file_format="pkl"))

    if not is_h5(file):
        file = File(file, "w")
        _dump_hdf5(obj, file, "")
        file.close()
    else:
        assert isinstance(file, Group)
        _dump_hdf5(obj, file, file.name)
