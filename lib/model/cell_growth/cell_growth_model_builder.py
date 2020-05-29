from lib.model.cell_growth.cell_growth_model import CellGrowthModel
from lib.model.cell_growth.step.living_cell_masking import LivingCellMaskingStep
from lib.model.cell_growth.step.perception_step import PerceptionStepFactory
from lib.model.cell_growth.step.stochastic_cell_update import StochasticCellUpdateStep
from lib.model.cell_growth.step.update_rule.update_rule_net import UpdateRuleNet
from lib.model.cell_growth.step.update_rule.update_rule_step import UpdateRuleStep


class CellGrowthModelBuilder:
    def __init__(self):
        self.__living_cell_masking_step = None
        self.__stochastic_cell_update_step = None
        self.__update_rule_step = None
        self.__perception_step = None
        self.__preview_size = (1500, 1500)

    def perception(self, filters, in_channels, out_channels_per_filter, preview_size, show_preview):
        self.__perception_step = PerceptionStepFactory().create(
            filters,
            in_channels,
            out_channels_per_filter,
            preview_size,
            show_preview
        )
        return self

    def update_rule(self, in_channels, hidden_channels, out_channels, output_zero_weights=True):
        net = UpdateRuleNet(in_channels, hidden_channels, out_channels, output_zero_weights)
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


def build_model_from(cfg):
    return CellGrowthModelBuilder().perception(
        filters=cfg['model.step.perception.filters'],
        in_channels=cfg['model.step.perception.in_channels'],
        out_channels_per_filter=cfg['model.step.perception.out_channels_per_filter'],
        preview_size=(cfg['model.preview.width'], cfg['model.preview.height']),
        show_preview=(cfg['model.preview.perception'])

    ).update_rule(
        in_channels=cfg['model.step.update_rule.in_channels'],
        hidden_channels=cfg['model.step.update_rule.hidden_channels'],
        out_channels=cfg['model.step.update_rule.out_channels'],
        output_zero_weights=cfg['model.step.update_rule.output_zero_weights']
    ).stochastic_cell_update(
        threshold=cfg['model.step.stochastic_cell_update.threshold']
    ).living_cell_masking(
        threshold=cfg['model.step.living_cell_masking.threshold']
    ).preview_size(
        size=(cfg['model.preview.width'], cfg['model.preview.height'])
    ).build()
