from .base import BaseFileHandler


class TxtHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        return file.read()

    def dump_to_fileobj(self, obj, file, **kwargs):
        file.write(str(obj))

    def dump_to_str(self, obj, **kwargs):
        return str(obj)
