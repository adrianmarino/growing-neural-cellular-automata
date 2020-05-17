import logging
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from lib.conv import conv, kernel
from lib.util.config import Config
from lib.util.image import load_img_as_tensor, normalize_img_tensor
from lib.util.inspect import show_tensor
from lib.util.logger_factory import LoggerFactory

config = Config('./config.yaml')
logger_factory = LoggerFactory(config['logger']).create()

InputConf = namedtuple('InputConfig', 'channels width height')

input_config = InputConf(channels=512, width=64, height=64)


class CellGrowthModel(nn.Module):
    def __init__(self, filters):
        self.__filters = filters
        self.__filter = filter
        super(CellGrowthModel, self).__init__()

    def forward(self, input):
        outputs = [F.conv2d(input, f, stride=1, padding=1) for f in self.__filters]
        perception = torch.cat(outputs, 0)
        logging.info(perception.size())
        return perception


def create_filters(in_channels):
    return [
        conv.repeated_kernel_filter(kernel.SOLVER_X, in_channels),
        conv.repeated_kernel_filter(kernel.SOLVER_Y, in_channels),
        conv.repeated_kernel_filter(kernel.IDENTITY, in_channels)
    ]


def create_model(in_channels=16):
    filters = create_filters(in_channels)
    return CellGrowthModel(filters)


def as_input(file_path):
    image_tensor = load_img_as_tensor(file_path)
    return normalize_img_tensor(image_tensor)


input_tensor = as_input('data/dragon.png')
# show_img(input_tensor)

batch = input_tensor[None, :, :]

model = create_model()
output = model.forward(batch)

show_tensor(output)
