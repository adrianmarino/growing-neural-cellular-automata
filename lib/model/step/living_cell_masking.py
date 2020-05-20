import torch as t
import torch.nn.functional as F

from lib.model.step import ModelStep, batch_map


class LivingCellMaskingStep(ModelStep):
    def perform(self, model, input_batch, current_batch):
        return batch_map(current_batch, self.__update_alpha_channel)

    def __update_alpha_channel(self, _, state_grid):
        state_grid[3, :] = state_grid[3, :] * self.__living_grid(state_grid)
        return state_grid

    def __living_grid(self, state_grid):
        alpha_channel = state_grid[3, :]
        alive = F.max_pool2d(
            alpha_channel[None, :],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        alive = alive[0] > 0.1
        return alive.type(t.float32)
