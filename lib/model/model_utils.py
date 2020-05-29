import re

from lib.util.os_utils import min_file_path_from


def get_loss_model_weights_path(file_path):
    loss = re.search("weights_(.*)_(.*)", file_path).group(2)
    return float(loss)


def last_weights_file_path(path):
    return min_file_path_from(f'{path}/*', get_loss_model_weights_path)
