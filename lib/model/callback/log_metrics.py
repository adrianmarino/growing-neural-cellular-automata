import logging

from lib.model.callback.callback import Callback


class LogMetrics(Callback):
    def perform(self, ctx):
        logging.info(f'Epoch: {ctx.epoch} - Loss: {ctx.loss} - LR: {ctx.lr}')
