import torch.nn.functional as F

from lib.model.step import ModelStep


class UpdateRuleStep(ModelStep):
    def perform(self, model, _, current_batch):
        output_batch = model.conv1(current_batch)
        output_batch = F.relu(output_batch)
        output_batch = model.conv2(output_batch)
        return output_batch
