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
        self.__parser.add_argument(
            '--config',
            help='Configuration file name',
            default='lizard-16x16'
        )
        self.__parser.add_argument(
            '--action',
            choices=['train', 'test'],
            help='Specify train or test model',
            default='test'
        )
        self.__parser.add_argument(
            '--show-output',
            dest='show_output',
            action='store_true',
            help='Show output evolution'
        )
        self.__parser.add_argument(
            '--hide-output',
            dest='show_output',
            action='store_false',
            help='Hide output evolution'
        )
        self.__parser.add_argument(
            '--show-loss-graph',
            dest='show_loss_graph',
            action='store_true',
            help='Show loss graph'
        )
        self.__parser.add_argument(
            '--hide-loss-graph',
            dest='show_loss_graph',
            action='store_false',
            help='Hide loss graph'
        )
        self.__parser.set_defaults(show_output=True)
        self.__parser.set_defaults(show_loss_graph=True)

        self.__args = self.__parser.parse_args()

    def config_name(self):
        return self.__args.config

    def action(self):
        return self.__args.action

    def show_output(self):
        return self.__args.show_output

    def show_loss_graph(self):
        return self.__args.show_loss_graph


def load_config(model_name):
    return Config(f'./config/config-{model_name}.yaml')


def init_logger(cfg):
    LoggerFactory(cfg['logger']).create()


def get_target(cfg, model_name):
    return DataTensor.target(
        image_array=img.load_as_tensor(f'./data/{model_name}.png'),
        in_channels=cfg['model.step.perception.in_channels']
    )
