import torch as t

from lib.model.step_based_model import ModelStep, batch_map
from lib.util import img


class StochasticCellUpdateStep(ModelStep):
    def __init__(self, threshold=0.5):
        self.__threshold = threshold

    def perform(self, input_batch, current_batch):
        return batch_map(
            current_batch,
            lambda index, ds_grid: self.__operation(input_batch[index], ds_grid)
        )

    def __operation(self, original_state_grid, ds_grid):
        rand_mask = self.__rand_mask(ds_grid)
        next_state_grid = original_state_grid + (ds_grid * rand_mask)
        return next_state_grid

    def __rand_mask(self, ds_grid):
        channels, width, height = ds_grid.size()
        rand_mask = t.rand(width, height) < self.__threshold
        rand_mask = rand_mask.type(t.float32)
        rand_mask = rand_mask.unsqueeze(0)
        rand_mask = rand_mask.repeat((channels, 1, 1))
        return rand_mask
