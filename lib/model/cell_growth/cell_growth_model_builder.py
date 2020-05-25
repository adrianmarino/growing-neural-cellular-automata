from lib.model.cell_growth.cell_growth_model import CellGrowthModel
from lib.model.cell_growth.step.living_cell_masking import LivingCellMaskingStep
from lib.model.cell_growth.step.perception_step import PerceptionStepFactory
from lib.model.cell_growth.step.stochastic_cell_update import StochasticCellUpdateStep
from lib.model.cell_growth.step.update_rule.update_rule_net import UpdateRuleNet
from lib.model.cell_growth.step.update_rule.update_rule_step import UpdateRuleStep
from lib.model.step_based_model import StepBasedModel


class CellGrowthModelBuilder:
    def __init__(self):
        self.__living_cell_masking_step = None
        self.__stochastic_cell_update_step = None
        self.__update_rule_step = None
        self.__perception_step = None
        self.__preview_size = (1500, 1500)

    def perception(self, filters, in_channels, out_channels_per_filter):
        self.__perception_step = PerceptionStepFactory().create(
            filters,
            in_channels,
            out_channels_per_filter
        )
        return self

    def update_rule(self, in_channels, hidden_channels, out_channels):
        net = UpdateRuleNet(in_channels, hidden_channels, out_channels)
        self.__update_rule_step = UpdateRuleStep(net)
        return self

    def stochastic_cell_update(self, threshold):
        self.__stochastic_cell_update_step = StochasticCellUpdateStep(threshold)
        return self

    def living_cell_masking(self, threshold):
        self.__living_cell_masking_step = LivingCellMaskingStep(threshold)
        return self

    def preview_size(self, size):
        self.__preview_size = size
        return self

    def build(self):
        return CellGrowthModel(
            steps=[
                self.__perception_step,
                self.__update_rule_step,
                self.__stochastic_cell_update_step,
                self.__living_cell_masking_step
            ],
            preview_size=self.__preview_size
        )
