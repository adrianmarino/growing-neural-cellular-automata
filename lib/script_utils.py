

import argparse

from lib.model.cell_growth.data_tensor import DataTensor
from lib.util import img
from lib.util.config import Config
from lib.util.logger_factory import LoggerFactory


class ArgumentManager:
    def __init__(self):
        self.__parser = argparse.ArgumentParser(
            prog='Growing Neural Cellular Automata Model',
            description='This is a model that learn to generate an image from one initial pixel . This model is based to  the '
                        'way that real multi-cellular organisms growth. '
        )
        self.__parser.add_argument('--config-name', help='Configuration file name', default='lizard-16x16')
        self.__parser.add_argument('--action', choices=['train', 'test'], help='Specify train or test model', default='test')
        self.__args = self.__parser.parse_args()

    def config_name(self):
        return self.__args.config_name

    def action(self):
        return self.__args.action


def load_config(model_name):
    return Config(f'./config/config-{model_name}.yaml')


def init_logger(cfg):
    LoggerFactory(cfg['logger']).create()


def get_target(cfg, model_name):
    return DataTensor.target(
        image_array=img.load_as_tensor(f'./data/{model_name}.png'),
        in_channels=cfg['model.step.perception.in_channels']
    )
