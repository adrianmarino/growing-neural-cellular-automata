from lib.model.callback.callback import Callback
from lib.util import img


class PlotOutput(Callback):
    def __init__(self, init=10, every=10, window_size=(1000, 1000)):
        self.__init = init
        self.__every = every
        self.__window_size = window_size

    def perform(self, ctx):
        if ctx.epoch > self.__init and ctx.epoch % self.__every == 0:
            img.show_tensor(ctx.output_batch[0], size=self.__window_size)
