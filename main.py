import logging
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.conv import single_kernel_filter
from lib.util.config import Config
from lib.util.image import load_img_as_tensor, show_img, normalize_img_tensor
from lib.util.logger_factory import LoggerFactory

config = Config('./config.yaml')
logger_factory = LoggerFactory(config['logger']).create()

InputConf = namedtuple('InputConfig', 'channels width height')

input_config = InputConf(channels=512, width=64, height=64)



FILTERS = [
    single_kernel_filter(Kernel.SOLVER_X, in_channels=16)

]

class CellGrowthModel(nn.Module):
    def __init__(self, filters):
        self.__filters = filters
        self.__filter = filter
        super(CellGrowthModel, self).__init__()

    def forward(self, input):

        weights = conv_weights(_filter, out_channels=2)
        inspect('weights', weights)

        logging.info(input.size())
        logging.info(self.__filter.size())
        output = F.conv2d(input, self.__filter, stride=1, padding=1)
        logging.info(output.size())

        perception = torch.cat(outputs, 0)

        logging.info(perception.size())
        return []


def as_input(file_path):
    image_tensor = load_img_as_tensor(file_path)
    return normalize_img_tensor(image_tensor)


input_tensor = as_input('data/dragon.png')
# show_img(input_tensor)

model = CellGrowthModel(DEFAULT_FILTER)

batch = input_tensor[None, :, :]
output = model.forward(batch)
show_img(input_tensor)
