import torch as t
import torch.nn.functional as F


def inspect(name, value): print(f'{name}{tuple(value.size())}:\n {value.numpy()}')


def create_samples_batch():
    # Create a sample with two channels
    channel = t.tensor(list(range(1, 25 + 1))).view((5, 5)).type(t.float)
    sample = t.stack([channel, channel * 2], 0)
    inspect('sample', sample)

    samples_batch = sample[None, :]
    inspect('samples_batch', samples_batch)
    return samples_batch


# Prepare a filter with two kernels
SOLVER_X_KERNEL = t.tensor([
    [-1, 0, +1],
    [-2, 0, +2],
    [-1, 0, +1],
]).type(t.float)

batch = create_samples_batch()

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
_filter = t.stack([SOLVER_X_KERNEL, SOLVER_X_KERNEL], 0)

# Weights have one filter by out channel..
weights = t.stack([_filter, _filter], 0)
inspect('weights', weights)

# Apply conv2d to input batch
output = F.conv2d(batch, weights, stride=1, padding=0)

inspect('output', output)
