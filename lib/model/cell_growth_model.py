from torch import nn

from lib.model.step.perception_step import PerceptionStepFactory
from lib.model.step.stochastic_cell_update import StochasticCellUpdateStep
from lib.model.step.update_rule_step import UpdateRuleStep


class CellGrowthModel(nn.Module):
    @staticmethod
    def create(cfg):
        return CellGrowthModel(
            steps=[
                PerceptionStepFactory(cfg).create(),
                UpdateRuleStep(),
                StochasticCellUpdateStep()
            ]
        )

    def __init__(self, steps):
        super(CellGrowthModel, self).__init__()
        self.__steps = steps
        self.conv1 = nn.Conv2d(
            in_channels=9,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, input):
        output = input
        for step in self.__steps:
            output = step.perform(self, output)
        return output
