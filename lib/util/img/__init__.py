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


def show(tensor, title=''):
    array = tensor.permute(1, 2, 0).detach().numpy()
    cv2.imshow(f'{tensor.size()} {title}', array)
    cv2.waitKey(0)
