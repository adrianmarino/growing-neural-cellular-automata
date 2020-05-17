import torch.nn.functional as F

from lib.model.step import ModelStep


class UpdateRuleStep(ModelStep):
    def perform(self, model, x):
        print(x.size())

        x = model.conv1(x)
        print(x.size())

        x = F.relu(x)
        print(x.size())

        x = model.conv2(x)
        print(x.size())
        return x
