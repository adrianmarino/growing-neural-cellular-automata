import abc

from collections import namedtuple

CallbackContext = namedtuple('CallbackContext', 'epoch loss output_batch lr')


class Callback:
    @abc.abstractmethod
    def perform(self, ctx):
        pass

    @staticmethod
    def exec(callbacks, ctx): [c.perform(ctx) for c in callbacks]
