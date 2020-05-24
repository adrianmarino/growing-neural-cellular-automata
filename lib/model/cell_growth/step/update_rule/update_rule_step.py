from lib.model.step_based_model import ModelStep


class UpdateRuleStep(ModelStep):

    def __init__(self, model): self.__model = model

    def perform(self, _, current_batch): return self.__model(current_batch)

    def parameters(self): return self.__model.parameters()
