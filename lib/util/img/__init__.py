import logging
import time

import cv2
import numpy as np
import torch as t


def load_as_tensor(file_path):
    image = cv2.imread(file_path)
    array = np.moveaxis(image, 2, 0)
    logging.info(f'Load "{file_path}" image as tensor of shape {array.shape}...')
    return t.from_numpy(array)


def normalize_tensor(tensor, from_channel=0, to_channel=3, divider=255.):
    return tensor[from_channel:to_channel, :, :] / divider


def show_tensor(tensor, title='', size=(1000, 1000), close_key=0, delay=0.2):
    rgb_array = tensor[0:3, :]
    rgb_array = rgb_array.permute(1, 2, 0).detach().numpy()
    show_array(rgb_array, f'{tensor.size()} {title}', size, close_key, delay)


def show_array(array, title='', size=(1000, 1000), close_key=0, delay=0.5):
    if size:
        array = cv2.resize(array, size)

    cv2.imshow(title, array)
    cv2.moveWindow(title, 40, 40)

    if close_key is not None:
        cv2.waitKey(0)

    cv2.destroyWindow(title)
    cv2.destroyAllWindows()
    time.sleep(delay)
