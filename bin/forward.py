from lib.model.cell_growth.cell_growth_model_builder import build_model_from
from lib.model.cell_growth.data_tensor import DataTensor
from lib.script_utils import load_config, init_logger, get_target

if __name__ == "__main__":
    model_name = 'lizard-16x16'

    cfg = load_config(model_name)
    init_logger(cfg)

    model = build_model_from(cfg)
    model.load(cfg['model.weights.path'])

    target = get_target(cfg, model_name)
    initial = DataTensor.initial(target)

    model.predict(initial, steps=cfg['model.forward.steps'], show_output=True)
