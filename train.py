import torch.optim as optim
from torch.nn import MSELoss

from lib.model.callback.plot_metrics import PlotMetrics
from lib.model.callback.plot_output import PlotOutput
from lib.model.cell_growth.cell_growth_model_builder import CellGrowthModelBuilder
from lib.model.cell_growth.data_tensor import DataTensor
from lib.util import img
from lib.util.config import Config
from lib.util.logger_factory import LoggerFactory

if __name__ == "__main__":
    cfg = Config('config.yaml')
    LoggerFactory(cfg['logger']).create()

    # ---------------------------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------------------------
    model = CellGrowthModelBuilder().perception(
        filters=cfg['model.step.perception.filters'],
        in_channels=cfg['model.step.perception.in_channels'],
        out_channels_per_filter=cfg['model.step.perception.out_channels_per_filter']
    ).update_rule(
        in_channels=cfg['model.step.update_rule.in_channels'],
        hidden_channels=cfg['model.step.update_rule.hidden_channels'],
        out_channels=cfg['model.step.update_rule.out_channels']
    ).stochastic_cell_update(
        threshold=cfg['model.step.stochastic_cell_update.threshold']
    ).living_cell_masking(
        threshold=cfg['model.step.living_cell_masking.threshold']
    ).build()

    target = DataTensor.target(
        image_array=img.load_as_tensor('data/lizard-32x32.png'),
        in_channels=cfg['model.step.perception.in_channels']
    )

    # ---------------------------------------------------------------------------------
    # Train model
    # ---------------------------------------------------------------------------------
    # img.show_tensor(target)
    # img.show_tensor(initial)

    optimizer = optim.Adam(model.steps[1].parameters(), lr=cfg['train.lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['train.scheduler.step_size'],
        gamma=cfg['train.scheduler.gamma']
    )

    model.train(
        epochs=cfg['train.epochs'],
        steps=cfg['train.steps'],
        initial=DataTensor.initial(target),
        target=target,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=MSELoss(reduction='sum'),
        callbacks=[
            PlotMetrics(reset_every=cfg['train.metrics.reset_every']),
            PlotOutput(every=cfg['train.preview.every'])
        ]
    )
