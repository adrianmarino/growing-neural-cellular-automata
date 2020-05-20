from lib.model.cell_growth_model import CellGrowthModel
from lib.model.step.living_cell_masking import LivingCellMaskingStep
from lib.model.step.perception_step import PerceptionStepFactory
from lib.model.step.stochastic_cell_update import StochasticCellUpdateStep
from lib.model.step.update_rule_step import UpdateRuleStep


class CellGrowthModelFactory:
    @staticmethod
    def create(cfg):
        perception_step = PerceptionStepFactory(cfg).create()

        steps = [
            perception_step,
            UpdateRuleStep(),
            StochasticCellUpdateStep(),
            LivingCellMaskingStep()
        ]

        return CellGrowthModel(
            steps,
            dense_in_channels=cfg['model.dense.in_channels'],
            dense_out_channels=cfg['model.dense.out_channels']
        )
