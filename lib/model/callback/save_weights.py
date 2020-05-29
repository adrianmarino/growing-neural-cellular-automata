from lib.model.callback.callback import Callback
from lib.util import os_utils


class SaveWeights(Callback):
    def __init__(self, path, every):
        self.__path = path
        os_utils.create_path(path)
        self.__every = every
        self.__current_loss = None

    def perform(self, ctx):
        if self.__current_loss is None:
            self.__current_loss = ctx.loss

        if self.__current_loss > ctx.loss and ctx.epoch % self.__every == 0:
            self.__current_loss = ctx.loss
            self.__save_weights(ctx)

        if ctx.epochs == ctx.epoch:
            self.__save_weights(ctx)

    def __save_weights(self, ctx):
        ctx.model.save(f'{self.__path}/weights_{ctx.epoch}_{ctx.loss}')
