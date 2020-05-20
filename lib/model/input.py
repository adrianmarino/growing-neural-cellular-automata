import torch as t

from lib.util import img


def from_img(file_path, in_channels=16):
    image = get_image(file_path)
    alpha_channel = get_alpha_channel(image)
    hidden_channels = get_hidden_channels(image, in_channels)
    return t.cat((image, alpha_channel, hidden_channels), 0)


def get_image(file_path):
    # Load image as (3, w, h) tensor...
    image = img.load_as_tensor(file_path)
    # change range 0..255 to 0..1...
    image = img.normalize_tensor(image)
    return image


def get_alpha_channel(image):
    _, img_width, img_height = image.size()
    return t.zeros((1, img_width, img_height))


def get_hidden_channels(rgb_channels, in_channels):
    img_channels_count, img_width, img_height = rgb_channels.size()
    return t.rand((in_channels - img_channels_count - 1, img_width, img_height))
