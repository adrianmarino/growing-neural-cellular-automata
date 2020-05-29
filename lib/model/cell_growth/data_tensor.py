import math

import torch as t


class DataTensor:
    @staticmethod
    def initial(target_tensor): return InitialTensorFactory().create(target_tensor)

    @staticmethod
    def target(image_array, in_channels): return TargetTensorFactory().create(image_array, in_channels)


class InitialTensorFactory:
    def create(self, target_tensor):
        channels, width, height = target_tensor.size()
        central_x, central_y = math.floor(width / 2), math.floor(height / 2)

        central_cell = target_tensor.clone()[:, central_x, central_y]
        central_cell[4:] = 1

        initial_tensor = t.zeros((channels, width, height))
        initial_tensor[:, central_x, central_y] = central_cell

        return initial_tensor


class TargetTensorFactory:
    def create(self, image_array, in_channels):
        image_array = self.__normalize(image_array)
        hidden_channels = self.__get_hidden_channels(image_array, in_channels)
        return t.cat([image_array, hidden_channels], 0)

    @staticmethod
    def __normalize(image_array):
        """
        change range 0..255 to 0..1...
        """
        return image_array[0:4, :, :] / 255.

    @staticmethod
    def __get_hidden_channels(rgb_channels, in_channels):
        img_channels, img_width, img_height = rgb_channels.size()
        return t.rand((in_channels - img_channels, img_width, img_height))
