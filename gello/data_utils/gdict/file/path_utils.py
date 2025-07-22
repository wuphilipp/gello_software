import os, os.path as osp, shutil
from pathlib import Path
import glob


def to_abspath(x):
    return osp.abspath(x)


def get_filename(x):
    return osp.basename(str(x))


def get_dirname(x):
    return osp.dirname(str(x))


def get_filename_suffix(x):
    return get_filename(x).split(".")[-1]


def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)


def add_suffix_to_filename(x, suffix=""):
    dirname = get_dirname(x)
    filename = get_filename(x)
    dot_split = filename.split(".")
    dot_split[-2] += f"_{suffix}"
    return osp.join(dirname, ".".join(dot_split))


def replace_suffix(x, suffix=""):
    dirname = get_dirname(x)
    filename = get_filename(x)
    name_split = filename.split(".")
    name_split[-1] = suffix
    return osp.join(dirname, ".".join(name_split))


def fopen(filepath, *args, **kwargs):
    if isinstance(filepath, str):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)
    raise ValueError("`filepath` should be a string or a Path")


def check_files_exist(filenames, msg_tmpl='file "{}" does not exist'):
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        if not osp.isfile(str(filename)):
            raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == "":
        return
    dir_name = str(dir_name)
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    src, dst = str(src), str(dst)
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def copy_folder(from_path, to_path, overwrite=True):
    print(f"Copy files from {from_path} to {to_path}")
    from_path = str(from_path)
    to_path = str(to_path)
    if os.path.exists(to_path) and overwrite:
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def copy_folders(source_dir, folder_list, target_dir, overwrite=True):
    assert all(["/" not in _ for _ in folder_list])
    for i in folder_list:
        copy_folder(osp.join(source_dir, i), osp.join(target_dir, i), overwrite)


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the directory. Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def find_vcs_root(path, markers=(".git",)):
    """Finds the root directory (including itself) of specified markers.
    Args:
        path (str): Path of directory or file.
        markers (list[str], optional): List of file or directory names.
    Returns:
        The directory contained one of the markers or None if not found.
    """
    if osp.isfile(path):
        path = osp.dirname(path)

    prev, cur = None, osp.abspath(osp.expanduser(path))
    while cur != prev:
        if any(osp.exists(osp.join(cur, marker)) for marker in markers):
            return cur
        prev, cur = cur, osp.split(cur)[0]
    return None


def parse_files(filenames):
    """
    filenames can contain four types of files: txt, h5, record, record_episode
    """
    from maniskill2_learn.utils.data import is_seq_of, concat_list
    from maniskill2_learn.utils.file import load

    supported_types = ["txt", "h5", "record", "record_episode"]
    ret_names = []
    if isinstance(filenames, str):
        filenames = [filenames]
    assert is_seq_of(filenames, str)

    def process_txt(file):
        file = load(file)
        replacements = (",", ";")
        for r in replacements:
            file = file.replace(r, " ")
        return file.split()

    for name in filenames:
        name = osp.expanduser(name)
        if not osp.exists(name):
            continue
        if osp.isdir(name):
            for file_type in supported_types:
                files = list(glob.glob(osp.join(name, "**", f"*.{file_type}"))) + list(glob.glob(osp.join(name, f"*.{file_type}")))
                if len(files) == 0:
                    continue
                if file_type == "txt":
                    files = parse_files(concat_list([process_txt(_) for _ in files]))
                ret_names += files
        else:
            file_suffix = get_filename_suffix(name)
            if file_suffix == "txt":
                ret_names += parse_files(concat_list([process_txt(_) for _ in files]))
            elif file_suffix in supported_types:
                ret_names.append(name)
    return ret_names
