import torch as t
import torch.nn.functional as F

from lib.model.conv import conv, kernel
from lib.util.inspect import show_tensor


def create_test_sample():
    """
    Create a test sample of shape (2, 5, 5):
        - 2 channels
        - width 5
        - height 5
    """
    channel = t.tensor(list(range(1, 25 + 1))).view((5, 5)).type(t.float)
    sample = t.stack([channel, channel * 2], 0)
    show_tensor('sample', sample)
    return sample


def create_samples_batch():
    """
    Create a test batch of samples with on samples of shape (2, 5, 5)
    Batch shape (1, 2, 5, 5)
    """
    sample = create_test_sample()
    samples_batch = sample.unsqueeze(0)
    show_tensor('samples_batch', samples_batch)
    return samples_batch


# ---------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------
if __name__ == "__main__":
    batch = create_samples_batch()
    #
    #
    #
    # ---------------------------------------------------------------------------------------------
    # CREATE FILTER: A filter has one kernel by each input channel.
    # ---------------------------------------------------------------------------------------------
    in_channels = batch.size()[1]
    filter_ = conv.repeated_kernel_filter(kernel.SOLVER_X, in_channels)
    show_tensor('filter', filter_)
    # ---------------------------------------------------------------------------------------------
    #
    #
    #
    # ---------------------------------------------------------------------------------------------
    # CREATE WEIGHTS: A weights.old.2 tensor have one filter by each out channel.
    # ---------------------------------------------------------------------------------------------
    weights = conv.weights(filter_, out_channels=2)
    show_tensor('weights.old.2', weights)
    # ---------------------------------------------------------------------------------------------
    #
    #
    #
    #
    # ---------------------------------------------------------------------------------------------
    # APPLY CONV2D OPERATION
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
    output = F.conv2d(batch, weights, stride=1, padding=0)
    show_tensor('output', output)
    # ---------------------------------------------------------------------------------------------
