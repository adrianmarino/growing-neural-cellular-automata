import glob
import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_file_path(path, filename, ext=''):
    return os.path.join(create_path(path), f'{filename}{"." if ext else ""}{ext}')


def min_file_path_from(path: object, min_func: object) -> object:
    try:
        list_of_files = glob.glob(path)
        latest_file = min(list_of_files, key=min_func)
        return latest_file
    except:
        return None
