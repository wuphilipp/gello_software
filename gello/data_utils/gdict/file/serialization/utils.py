import pickle


def serialize(obj):
    return pickle.dumps(obj)


def deserialize(obj):
    return pickle.loads(obj)


def list_from_file(filename, prefix="", offset=0, max_num=-1):
    cnt = 0
    item_list = []
    with open(filename, "r") as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num >= 0 and cnt >= max_num:
                break
            item_list.append(prefix + line.rstrip("\n"))
            cnt += 1
    return item_list


def dict_from_file(filename, key_type=str, offset=0, max_num=-1):
    mapping = {}
    cnt = 0
    with open(filename, "r") as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num >= 0 and cnt >= max_num:
                break
            items = line.rstrip("\n").split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
            cnt += 1
    return mapping


def dict_to_csv_table(x):
    ret = []
    for key in x.keys():
        ret.append([key, x[key]])
    return ret


def csv_table_to_dict(x):
    for y in x:
        assert len(y) == 2
    ret = {}
    for y in x:
        ret[y[0]] = y[1]
    return ret
