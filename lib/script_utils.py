from lib.model.cell_growth.data_tensor import DataTensor
from lib.util import img
from lib.util.config import Config
from lib.util.logger_factory import LoggerFactory


def load_config(model_name):
    return Config(f'./config/config-{model_name}.yaml')


def init_logger(cfg):
    LoggerFactory(cfg['logger']).create()


def get_target(cfg, model_name):
    return DataTensor.target(
        image_array=img.load_as_tensor(f'./data/{model_name}.png'),
        in_channels=cfg['model.step.perception.in_channels']
    )
