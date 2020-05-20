from torch import nn


class CellGrowthModel(nn.Module):
    def __init__(self, steps, dense_in_channels, dense_out_channels):
        super(CellGrowthModel, self).__init__()
        self.__steps = steps

        self.conv1 = nn.Conv2d(
            in_channels=dense_in_channels,
            out_channels=128,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=dense_out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        )

    def forward(self, input_batch):
        current_batch = input_batch
        for step in self.__steps:
            current_batch = step.perform(self, input_batch, current_batch)
        return current_batch
