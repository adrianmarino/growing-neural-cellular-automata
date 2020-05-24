import torch as t
import torch.nn.functional as F

from lib.model.conv import conv, kernel
from lib.model.step_based_model import ModelStep, batch_map


class PerceptionStep(ModelStep):
    def __init__(self, filters, out_channels_per_filter, stride=1, padding=1):
        self.__filters = filters
        self.__stride = stride
        self.__padding = padding
        self.__out_channels_per_filter = out_channels_per_filter

    def perform(self, _, input_batch):
        return batch_map(input_batch, self.perception_operation)

    def perception_operation(self, _, input):
        output = None

        for f in self.__filters:
            step_output = self.__conv(input, f)[0]
            if output is None:
                output = step_output
            else:
                output = t.cat([output, step_output], 0)

        return output

    def __conv(self, input, filter_):
        weights = conv.weights(filter_, self.__out_channels_per_filter)

        return F.conv2d(
            input[None, :],
            weights,
            stride=self.__stride,
            padding=self.__padding
        )


class PerceptionStepFactory:
    def create(self, filters, in_channels, out_channels_per_filter):
        return PerceptionStep(
            filters=self.__create_filters(filters, in_channels),
            out_channels_per_filter=out_channels_per_filter,
            stride=1,
            padding=1
        )

    def __create_filters(self, filter_names, in_channels):
        filters = []
        for name in filter_names:
            k = self.__get_kernel(name)
            filters.append(conv.repeated_kernel_filter(k, in_channels))
        return filters

    def __get_kernel(self, name):
        return getattr(kernel, name)
