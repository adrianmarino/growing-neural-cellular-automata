import torch as t
import torch.nn.functional as F

from lib.model.step_based_model import ModelStep, batch_map


class LivingCellMaskingStep(ModelStep):
    def __init__(self, threshold=0.1):
        self.__threshold = threshold

    def perform(self, _, current_batch):
        return batch_map(current_batch, self.__update_alive_cells)

    def __update_alive_cells(self, _, state_grid):
        alive_cells = self.__alive_cells(state_grid)
        next_state_grid = state_grid * alive_cells
        return next_state_grid

    def __alive_cells(self, state_grid):
        alpha_channel = state_grid[3:4, :]
        alpha_channel = alpha_channel > self.__threshold
        alpha_channel = alpha_channel.type(t.float32)
        input_batch = alpha_channel.unsqueeze(0)

        weights = t.ones((1, 1, 3, 3)).type(t.float32)
        output_batch = F.conv2d(input_batch, weights, padding=1)

        alive_grid = output_batch[0][0]
        alive_grid = alive_grid > 0.0
        alive_grid = alive_grid.type(t.int)

        state_channels = state_grid.size()[0]
        return t.stack([alive_grid] * state_channels, 0)
