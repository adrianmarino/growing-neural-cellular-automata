import torch as t
import torch.nn.functional as F

SOLVER_X_KERNEL = t.tensor([
    [-1, 0, +1],
    [-2, 0, +2],
    [-1, 0, +1],
]).type(t.float)


def show_tensor(name, value):
    print(f'{name}{tuple(value.size())}:\n {value.numpy()}')


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
    kernel_by_in_channel = [SOLVER_X_KERNEL] * in_channels
    conv_filter = t.stack(kernel_by_in_channel, 0)
    show_tensor('Filter (2 SOLVE_X kernels)', conv_filter)
    # ---------------------------------------------------------------------------------------------
    #
    #
    #
    # ---------------------------------------------------------------------------------------------
    # CREATE WEIGHTS: A weights tensor have one filter by each out channel.
    # ---------------------------------------------------------------------------------------------
    out_channels = 2
    filter_by_out_channel = [conv_filter] * out_channels
    weights = t.stack(filter_by_out_channel, 0)
    show_tensor('weights', weights)
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
