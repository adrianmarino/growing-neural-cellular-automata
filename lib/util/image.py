import logging

import cv2
import numpy as np
import torch


def load_img_as_tensor(file_path):
    image = cv2.imread(file_path)
    array = np.moveaxis(image, 2, 0)
    logging.info(f'Load "{file_path}" image as tensor of shape {array.shape}...')
    return torch.from_numpy(array)


def normalize_img_tensor(tensor, divider=255.):
    return tensor[0:3, :, :] / divider


def show_img(tensor, title='', close_key=0):
    array = tensor.permute(1, 2, 0).numpy()
    cv2.imshow(f'{tensor.size()} {title}', array)
    cv2.waitKey(close_key)
    return tensor
