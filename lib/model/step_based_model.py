import abc

import torch as t

from lib.util import img


def batch_map(input_batch, function=lambda index, element: element):
    output_batch = None
    for index, _input in enumerate(input_batch):
        output = function(index, _input)
        output = output.unsqueeze(0)
        output_batch = output if output_batch is None else t.cat([output_batch, output], 0)
    return output_batch


class ModelStep:
    @abc.abstractmethod
    def perform(self, input_batch, current_batch):
        pass

    def name(self):
        return self.__class__.__name__


class StepBasedModel:
    def __init__(self, steps, preview_size=(1500, 1500)):
        self.steps = steps
        self.__preview_size = preview_size

    def forward(self, input_batch, show_step_output=False, show_output=False):
        current_batch = input_batch

        for step in self.steps:
            current_batch = step.perform(input_batch, current_batch)
            if show_step_output:
                self.__show_preview(current_batch, step.name())

        if show_output:
            self.__show_preview(current_batch)

        return current_batch

    def __show_preview(self, current_batch, step_name=''):
        img.show_tensor(
            current_batch[0],
            title=f'{step_name} output',
            size=self.__preview_size
        )
