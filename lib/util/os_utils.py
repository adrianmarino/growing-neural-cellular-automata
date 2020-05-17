import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def create_file_path(path, filename, ext=''):
    return os.path.join(create_path(path), f'{filename}{"." if ext else ""}{ext}')
