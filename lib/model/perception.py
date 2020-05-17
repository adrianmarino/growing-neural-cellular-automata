import torch as t
import torch.nn.functional as F

from lib.conv import conv


class PerceptionStep:
    def __init__(self, filters, out_channels_per_filter, stride=1, padding=1):
        self.__filters = filters
        self.__stride = stride
        self.__padding = padding
        self.__out_channels_per_filter = out_channels_per_filter

    def apply_to(self, input):
        outputs = [self.__conv(input, f) for f in self.__filters]
        return t.cat(outputs, 0)

    def __conv(self, input, filter_):
        weights_ = conv.weights(filter_, self.__out_channels_per_filter)
        input_batch = input[None, :]

        return F.conv2d(
            input_batch,
            weights_,
            stride=self.__stride,
            padding=self.__padding
        )
