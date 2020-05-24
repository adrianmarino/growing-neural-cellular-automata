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
        return state_grid * alive_cells

    def __alive_cells(self, state_grid):
        state_channels = state_grid.size()[0]
        alpha_channel = state_grid[3, :]

        input_batch = alpha_channel.unsqueeze(0)
        output_batch = F.max_pool2d(
            input_batch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        alive_grid = output_batch[0]
        alive_grid = alive_grid > self.__threshold
        alive_grid = alive_grid.type(t.float32)

        alive_tensor = t.stack([alive_grid] * state_channels, 0)
        return alive_tensor
