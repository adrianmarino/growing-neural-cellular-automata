from torch import nn
import torch.nn.functional as F

from lib.model.layer_utils import create_conv2d


class UpdateRuleNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(UpdateRuleNet, self).__init__()
        self.conv1 = create_conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        self.conv2 = create_conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels
        )

    def forward(self, input_batch):
        batch = self.conv1(input_batch)
        batch = F.relu(batch)
        batch = self.conv2(batch)
        return batch
