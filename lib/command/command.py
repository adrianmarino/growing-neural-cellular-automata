import abc


class Command:
    @abc.abstractmethod
    def exec(self, cfg, args):
        pass
