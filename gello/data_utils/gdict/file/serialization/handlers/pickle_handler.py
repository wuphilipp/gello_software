import pickle, importlib, bz2, gzip
from .base import BaseFileHandler
from ...path_utils import get_filename_suffix


class PickleProtocol:
    def __init__(self, level):
        self.previous = pickle.HIGHEST_PROTOCOL
        self.level = level

    def __enter__(self):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.level

    def __exit__(self, *exc):
        importlib.reload(pickle)
        pickle.HIGHEST_PROTOCOL = self.previous


class PickleHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 5)
        pickle.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 5)
        return pickle.dumps(obj, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        file_suffix = get_filename_suffix(filepath)
        assert file_suffix in ['pkl', 'pgz', 'pbz2'], f'{file_suffix} is not supported. Please use of pkl, pgz, pbz2'
        if file_suffix == 'pkl':
            with open(filepath, 'rb') as f:
                return self.load_from_fileobj(f, **kwargs)
        elif file_suffix == 'pgz':
            with gzip.GzipFile(filepath, 'r') as f:
                return self.load_from_fileobj(f, **kwargs)
        elif file_suffix == 'pbz2':
            with bz2.BZ2File(filepath, 'r') as f:
                return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        file_suffix = get_filename_suffix(filepath)
        assert file_suffix in ['pkl', 'pgz', 'pbz2'], f'{file_suffix} is not supported. Please use of pkl, pgz, pbz2'
        if file_suffix == 'pkl':
            with open(filepath, 'wb') as f:
                return self.dump_to_fileobj(obj, f, **kwargs)
        elif file_suffix == 'pgz':
            with gzip.GzipFile(filepath, 'w') as f:
                return self.dump_to_fileobj(obj, f, **kwargs)
        elif file_suffix == 'pbz2':
            with bz2.BZ2File(filepath, 'w') as f:
                return self.dump_to_fileobj(obj, f, **kwargs)
