from lib.util import img


def from_img(file_path, in_channels=3):
    img_tensor = img.load_as_tensor(file_path)

    width, height = img_tensor.size()[1], img_tensor.size()[2]

    # alpha_channel = t.zeros((1, width, height))
    # rand_channel = t.rand((1, width, height))

    # extra_channels = t.stack(alpha_channel + rand_channel * 12, 0)

    # input_tensor = t.stack([img_tensor, extra_channels], 0)

    normalized_tensor = img.normalize_tensor(img_tensor, from_channel=0, to_channel=3)
    return normalized_tensor
