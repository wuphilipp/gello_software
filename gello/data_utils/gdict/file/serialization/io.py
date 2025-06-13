from pathlib import Path
import os.path as osp
from io import BytesIO
from ...data import is_list_of, is_str
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler, CSVHandler, TxtHandler

file_handlers = {
    "json": JsonHandler(),
    "yaml": YamlHandler(),
    "yml": YamlHandler(),
    "pickle": PickleHandler(),
    "pkl": PickleHandler(),
    "pgz": PickleHandler(),
    "pbz2": PickleHandler(),
    "csv": CSVHandler(),
    "txt": TxtHandler(),
}


def load(file, file_format=None, **kwargs):
    if isinstance(file, Path):
        file = str(file)
    if isinstance(file, bytes):
        file = BytesIO(file)
    if is_str(file):
        if file_format is None:
            file_format = file.split(".")[-1]
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if is_str(file):
        file = osp.expanduser(file)
        obj = handler.load_from_path(file, **kwargs)
    elif hasattr(file, "read"):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split(".")[-1]
        elif file is None:
            raise ValueError("file_format must be specified since file is None")
    if file_format not in file_handlers:
        raise TypeError(f"Unsupported format: {file_format}")

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        file = osp.expanduser(file)
        handler.dump_to_path(obj, file, **kwargs)
    elif hasattr(file, "write"):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def _register_handler(handler, file_formats):
    if not isinstance(handler, BaseFileHandler):
        raise TypeError(f"handler must be a child of BaseFileHandler, not {type(handler)}")
    if isinstance(file_formats, str):
        file_formats = [file_formats]
    if not is_list_of(file_formats, str):
        raise TypeError("file_formats must be a str or a list of str")
    for ext in file_formats:
        file_handlers[ext] = handler


def register_handler(file_formats, **kwargs):
    def wrap(cls):
        _register_handler(cls(**kwargs), file_formats)
        return cls

    return wrap
