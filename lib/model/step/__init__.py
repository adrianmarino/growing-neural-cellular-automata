import abc


def batch_map(input_batch, function=lambda index, element: element):
    output_batch = None
    for index, input in enumerate(input_batch):
        output = function(index, input)
        output = output[None, :]
        output_batch = output if output_batch is None else t.cat([output_batch, output], 0)
    return output_batch


class ModelStep:
    @abc.abstractmethod
    def perform(self, model, input_batch, current_batch):
        pass
