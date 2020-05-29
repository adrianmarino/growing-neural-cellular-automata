import random

import torch as t

from lib.model.callback.callback import CallbackContext, Callback
from lib.model.model_utils import last_weights_file_path
from lib.model.step_based_model import StepBasedModel


class CellGrowthModel(StepBasedModel):
    def train(self, initial, target, epochs, steps, optimizer, scheduler, loss_fn, callbacks=()):
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            output_batch = initial.clone().unsqueeze(0)

            for step in range(0, random.randint(steps[0], steps[1])):
                output_batch = self.forward(output_batch)
                output_batch = t.clamp(output_batch, 0.0, 1.0)

            loss = self.__calculate_loss(loss_fn, output_batch, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.__exec_callbacks(callbacks, epochs, epoch, loss, output_batch, scheduler)

    def predict(self, initial, steps, show_output=True):
        batch = initial.clone().unsqueeze(0)
        for _ in range(1, steps + 1):
            batch = self.forward(batch, show_output=show_output)
            batch = t.clamp(batch, 0.0, 1.0)

    def __exec_callbacks(self, callbacks, epochs, epoch, loss, output_batch, scheduler):
        Callback.exec(
            callbacks,
            ctx=CallbackContext(
                epochs=epochs,
                epoch=epoch,
                loss=loss.item(),
                output_batch=output_batch,
                lr=scheduler.get_last_lr()[0],
                model=self
            )
        )

    @staticmethod
    def __calculate_loss(loss_fn, output_batch, target):
        samples_count, _, _, _ = output_batch.size()
        target_batch = target.repeat(samples_count, 1, 1, 1)
        return loss_fn(output_batch[:, 0:3, :], target_batch[:, 0:3, :])

    def parameters(self):
        return self.steps[1].parameters()

    def save(self, path):
        self.steps[1].save(path)

    def load(self, path):
        last_weights_file = last_weights_file_path(path)
        if last_weights_file:
            self.steps[1].load(last_weights_file)
