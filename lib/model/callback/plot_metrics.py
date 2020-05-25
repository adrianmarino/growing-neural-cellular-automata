import logging

import matplotlib.pyplot as plt
from lib.model.callback.callback import Callback


class PlotMetrics(Callback):
    def __init__(self, init=10, every=10, reset_every=None):
        self.__epochs = []
        self.__losses = []
        self.__lr = []
        self.__init = init
        self.__every = every
        self.__reset_every = reset_every

    def perform(self, ctx):
        if ctx.epoch > self.__init:
            self.__losses.append(ctx.loss)
            self.__epochs.append(ctx.epoch)
            self.__lr.append(ctx.lr)

        if ctx.epoch > self.__init and ctx.epoch % self.__every == 0:
            self.plot_loss(self.__epochs, self.__losses, self.__lr)
            logging.info(f'Epoch: {ctx.epoch} - Loss: {ctx.loss} - LR: {ctx.lr}')

        if self.__reset_every is not None and ctx.epoch % self.__reset_every == 0:
            self.__losses.clear()
            self.__epochs.clear()
            self.__lr.clear()

    @staticmethod
    def plot_loss(epochs, losses, lrs):
        plt.clf()
        plt.plot(epochs, losses, label='Loss')
        plt.plot(epochs, lrs, label='LR')
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.001)
