import torch as t
import torch.nn.functional as F

from lib.model.step import ModelStep, batch_map


class LivingCellMaskingStep(ModelStep):
    def perform(self, model, input_batch, current_batch):
        return batch_map(current_batch, self.__operation)

    def __operation(self, _, state_grid):
        alive = F.max_pool2d(state_grid[3, :, :], (3, 3)) > 0.1
        return state_grid[3, :, :] * alive.type(t.float32)
