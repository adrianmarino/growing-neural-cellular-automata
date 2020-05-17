from torch import nn

from lib.conv import kernel, conv
from lib.model.perception import PerceptionStep


class CellGrowthModel(nn.Module):
    def __init__(self, perception_step):
        super(CellGrowthModel, self).__init__()
        self.__perception_step = perception_step

    def forward(self, input):
        perception = self.__perception_step.apply_to(input)
        return perception


class PerceptionStepFactory:
    @staticmethod
    def create(cfg):
        return PerceptionStep(
            filters=PerceptionStepFactory.__create_filters(cfg),
            out_channels_per_filter=cfg['model.perception.out-channels'],
            stride=1,
            padding=1
        )

    @staticmethod
    def __create_filters(cfg):
        filters = []
        for name in cfg['model.perception.filters']:
            k = getattr(kernel, name)
            filters.append(conv.repeated_kernel_filter(k, cfg['model.in-channels']))
        return filters


class CellGrowthModelFactory:
    @staticmethod
    def create(cfg):
        perception_step = PerceptionStepFactory.create(cfg)
        return CellGrowthModel(perception_step)
