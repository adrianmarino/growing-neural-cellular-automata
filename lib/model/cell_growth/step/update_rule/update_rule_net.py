import torch.nn.functional as F
from torch import nn

from lib.model.layer_utils import create_conv2d


class UpdateRuleNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, output_zero_weights=True):
        super(UpdateRuleNet, self).__init__()
        self.conv1 = create_conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels
        )
        self.conv2 = create_conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            zero_weights=output_zero_weights
        )

    def forward(self, input_batch):
        output_batch = self.conv1(input_batch)
        output_batch = F.relu(output_batch)
        output_batch = self.conv2(output_batch)
        return output_batch
