import logging

import torch as t

from lib.model import input
from lib.model.cell_growth_model import CellGrowthModel
from lib.util import img
from lib.util.config import Config
from lib.util.logger_factory import LoggerFactory

if __name__ == "__main__":
    cfg = Config('config.yaml')
    LoggerFactory(cfg['logger']).create()

    model = CellGrowthModel.create(cfg)

    input_tensor = input.from_img('data/dragon.png', in_channels=cfg['model.in-channels'])
    # img.show(input_tensor)
    input_batch = t.stack([input_tensor], 0)
    print(input_batch.size())

    output = model.forward(input_batch)
    logging.info(output.size())

    img.show(output[0])
