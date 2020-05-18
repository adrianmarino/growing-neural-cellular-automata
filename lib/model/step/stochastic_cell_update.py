import torch as t

from lib.model.step import ModelStep, batch_map


class StochasticCellUpdateStep(ModelStep):
    def perform(self, model, input_batch, current_batch):
        return batch_map(
            current_batch,
            lambda index, ds_grid: self.__operation(index, ds_grid, input_batch, current_batch)
        )

    def __operation(self, index, ds_grid, input_batch, current_batch):
        rand_mask = self.__rand_mask(current_batch)
        return input_batch[index] + (ds_grid * rand_mask)

    def __rand_mask(self, current_batch):
        batch_size = current_batch.size()
        width, height = batch_size[-1], batch_size[-2]
        rand_mask = (t.rand(width, height) < 0.5).type(t.float32)
        return rand_mask
