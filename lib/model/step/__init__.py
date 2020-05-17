import abc


class ModelStep:
    @abc.abstractmethod
    def perform(self,  model, input_batch):
        pass
