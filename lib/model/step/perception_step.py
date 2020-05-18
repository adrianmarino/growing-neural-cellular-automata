import torch as t
import torch.nn.functional as F

from lib.conv import conv, kernel
from lib.model.step import ModelStep, batch_map


class PerceptionStep(ModelStep):
    def __init__(self, filters, out_channels_per_filter, stride=1, padding=1):
        self.__filters = filters
        self.__stride = stride
        self.__padding = padding
        self.__out_channels_per_filter = out_channels_per_filter

    def perform(self, _, input_batch, current_batch):
        return batch_map(current_batch, self.perception_operation)

    def perception_operation(self, _, input):
        output = input.clone()
        for f in self.__filters:
            output = t.cat([output, self.__conv(input, f)[0]], 0)
        return output

    def __conv(self, input, filter_):
        return F.conv2d(
            self.__to_batch(input),
            conv.weights(filter_, self.__out_channels_per_filter),
            stride=self.__stride,
            padding=self.__padding
        )

    def __to_batch(self, input):
        return input[None, :]


class PerceptionStepFactory:
    def __init__(self, cfg): self.__cfg = cfg

    def create(self):
        return PerceptionStep(
            filters=self.__create_filters(),
            out_channels_per_filter=self.__cfg['model.perception.out-channels'],
            stride=1,
            padding=1
        )

    def __create_filters(self):
        filters = []
        for name in self.__cfg['model.perception.filters']:
            k = self.__get_kernel(name)
            filters.append(conv.repeated_kernel_filter(k, self.__cfg['model.in-channels']))
        return filters

    def __get_kernel(self, name):
        return getattr(kernel, name)
