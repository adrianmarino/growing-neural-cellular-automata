from lib.model import input
from lib.model.cell_growth_model import CellGrowthModelFactory
from lib.util import img
from lib.util.config import Config
from lib.util.logger_factory import LoggerFactory

if __name__ == "__main__":
    cfg = Config('config.yaml')
    LoggerFactory(cfg['logger']).create()

    input_tensor = input.from_img('data/dragon.png', in_channels=cfg['model.in-channels'])
    img.show(input_tensor)

    # Create model
    model = CellGrowthModelFactory.create(cfg)

    output = model.forward(input_tensor)

    img.show(output[0])
    img.show(output[1])
    img.show(output[2])
