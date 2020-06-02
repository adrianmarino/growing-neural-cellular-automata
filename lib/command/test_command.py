from lib.command.command import Command
from lib.model.callback.plot_output import PlotOutput
from lib.model.cell_growth.cell_growth_model_builder import build_model_from
from lib.model.cell_growth.data_tensor import DataTensor
from lib.script_utils import get_target


class TestCommand(Command):
    def exec(self, cfg, args):
        model = build_model_from(cfg)
        model.load(cfg['model.weights.path'])

        target = get_target(cfg, args.config_name())
        initial = DataTensor.initial(target)

        callbacks = []
        if args.show_loss_graph():
            callbacks.append(PlotOutput(
                target=target,
                init=1,
                every=1,
                window_size=(cfg['model.preview.width'] * 2, cfg['model.preview.height']),
                close_key=0
            ))
        model.predict(initial, steps=cfg['model.forward.steps'], callbacks=callbacks)
