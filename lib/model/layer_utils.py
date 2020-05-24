import torch as t
from torch import nn


def create_conv2d(
        in_channels,
        out_channels,
        stride=1,
        padding=0,
        kernel_size=(1, 1),
        zero_weights=False):
    layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding
    )
    if zero_weights:
        layer.weight.data = t.zeros([out_channels, in_channels, 1, 1])

    return layer
