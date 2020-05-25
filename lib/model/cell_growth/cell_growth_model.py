from lib.model.callback.callback import CallbackContext, Callback
from lib.model.step_based_model import StepBasedModel


class CellGrowthModel(StepBasedModel):
    def train(self, initial, target, epochs, steps, optimizer, scheduler, loss_fn, callbacks=()):
        for epoch in range(0, epochs):
            output_batch = initial.clone().unsqueeze(0)
            for step in range(0, steps):
                output_batch = self.forward(output_batch)

            loss = self.__calculate_loss(loss_fn, output_batch, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.__exec_callbacks(callbacks, epoch, loss, output_batch, scheduler)

    @staticmethod
    def __exec_callbacks(callbacks, epoch, loss, output_batch, scheduler):
        Callback.exec(
            callbacks,
            ctx=CallbackContext(
                epoch=epoch,
                loss=loss.item(),
                output_batch=output_batch,
                lr=scheduler.get_last_lr()[0]
            )
        )

    @staticmethod
    def __calculate_loss(loss_fn, output_batch, target):
        samples_count, _, _, _ = output_batch.size()
        target_batch = target.repeat(samples_count, 1, 1, 1)
        return loss_fn(output_batch, target_batch)
