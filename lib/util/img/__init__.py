import logging
import time

import cv2
import numpy as np
import torch as t


def load_as_tensor(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # make mask of where the transparent bits are
    # trans_mask = image[:, :, 3] == 0

    # replace areas of transparency with white and not transparent
    # image[trans_mask] = [255, 255, 255, 255]

    image = np.moveaxis(image, 2, 0)
    logging.info(f'Load "{file_path}" image as tensor of shape {image.shape}...')
    return t.from_numpy(image)


def show_tensor(tensor, title='', size=(500, 500), close_key=0, delay=0.2):
    image = tensor[0:4, :]
    image = image.permute(1, 2, 0).detach().numpy()
    show_array(image, f'{tensor.size()} {title}', size, close_key, delay)


def show_array(image, title='', size=(500, 500), close_key=0, delay=0.5):
    if size:
        image = cv2.resize(image, size)

    cv2.imshow(title, image)
    cv2.moveWindow(title, 40, 80)

    if close_key is not None:
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        cv2.destroyAllWindows()

    time.sleep(delay)
