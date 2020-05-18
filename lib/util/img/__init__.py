import logging

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


def show(tensor, title='', size=None, close_key=0):
    array = tensor.permute(1, 2, 0).detach().numpy()

    if size:
        array = cv2.resize(array, size)

    title = f'{tensor.size()} {title}'
    cv2.imshow(title, array)
    cv2.moveWindow(title, 40, 40)

    if close_key is not None:
        cv2.waitKey(0)
