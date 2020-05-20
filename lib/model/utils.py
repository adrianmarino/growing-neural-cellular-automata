import cv2
from torchviz import make_dot

from lib.util import img


def show_model_graph(output, filename='temp/model_graph', extension='png'):
    graph = make_dot(output)
    graph.render(filename, format=extension)
    image = cv2.imread(f'{filename}.{extension}')
    img.show_array(image)
