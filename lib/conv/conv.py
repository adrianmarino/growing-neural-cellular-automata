# ---------------------------------------------------------------------------------------------
# CONV2D
# ---------------------------------------------------------------------------------------------
# Each filter is collection of kernels, with there being one kernel for every single input
# channel.
#
# Each of the kernels of the filter “slides” over their respective input channels,
# producing a processed version of each.
#
# Each of the per-channel processed versions are then summed together to form one channel.
# The kernels of a filter each produce one version of each input channel, and the filter
# as a whole produces one overall output channel.
#
# See SOURCE: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
# ---------------------------------------------------------------------------------------------

import torch as t


def single_kernel_filter(kernel, in_channels=1): return conv_filter([kernel] * in_channels)


def conv_filter(in_channel_kernels=[]): return t.stack(in_channel_kernels, 0)


def weights(_filter, out_channels):
    return t.stack([_filter for _ in range(0, out_channels)], 0)
