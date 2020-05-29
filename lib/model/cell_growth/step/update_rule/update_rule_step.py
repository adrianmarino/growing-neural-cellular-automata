import logging

from lib.model.step_based_model import ModelStep
import torch as t


class UpdateRuleStep(ModelStep):

    def __init__(self, model): self.__model = model

    def perform(self, _, current_batch):
        return self.__model(current_batch)

    def parameters(self):
        return self.__model.parameters()

    def save(self, path):
        t.save(self.__model.state_dict(), path)
        logging.info(f'Model weights save in {path} file...')

    def load(self, path):
        self.__model.load_state_dict(t.load(path))
        self.__model.eval()
        logging.info(f'Model weights loaded from {path} file...')
