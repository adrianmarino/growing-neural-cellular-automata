from lib.model.step import ModelStep


class StochasticCellUpdateStep(ModelStep):
    def perform(self, model, input):
        return input
