import abc

from collections import namedtuple

CallbackContext = namedtuple('CallbackContext', 'epochs epoch loss output_batch lr model')


class Callback:
    @abc.abstractmethod
    def perform(self, ctx):
        pass

    @staticmethod
    def exec(callbacks, ctx): [c.perform(ctx) for c in callbacks]
