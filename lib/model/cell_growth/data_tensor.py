import math
import torch as t
from lib.util import img


class DataTensor:
    @staticmethod
    def initial(target_tensor): return InitialTensorFactory().create(target_tensor)

    @staticmethod
    def target(image_array, in_channels): return TargetTensorFactory().create(image_array, in_channels)


class InitialTensorFactory:
    def create(self, target_tensor):
        channels, width, height = target_tensor.size()
        central_x, central_y = math.floor(width / 2), math.floor(height / 2)

        central_cell = target_tensor[0:3, central_x, central_x]

        initial_tensor = t.zeros((channels, width, height))
        initial_tensor[0:3, central_x, central_y] = central_cell

        return initial_tensor


class TargetTensorFactory:
    def create(self, image_array, in_channels):
        # change range 0..255 to 0..1...
        image_array = img.normalize_tensor(image_array)
        alpha_channel = self.__get_alpha_channel(image_array)
        hidden_channels = self.__get_hidden_channels(image_array, in_channels)
        return t.cat((image_array, alpha_channel, hidden_channels), 0)

    @staticmethod
    def __get_alpha_channel(tensor):
        _, img_width, img_height = tensor.size()
        return t.zeros((1, img_width, img_height))

    @staticmethod
    def __get_hidden_channels(rgb_channels, in_channels):
        img_channels_count, img_width, img_height = rgb_channels.size()
        return t.rand((in_channels - img_channels_count - 1, img_width, img_height))
