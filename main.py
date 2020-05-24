import logging

import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.optim import Adam

from lib.model.cell_growth.cell_growth_model_builder import CellGrowthModelBuilder
from lib.model.cell_growth.data_tensor import DataTensor
from lib.util import img
from lib.util.config import Config
from lib.util.logger_factory import LoggerFactory
import torch as t


def build_model(cfg):
    return CellGrowthModelBuilder().perception(
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


def plot_loss(epochs, losses):
    plt.clf()
    plt.plot(epochs, losses, label='Loss')
    plt.legend()
    plt.grid()
    plt.draw()
    plt.pause(0.001)


def train(model):
    image = img.load_as_tensor('data/lizard-32x32.png')

    target = DataTensor.target(
        image,
        in_channels=cfg['model.step.perception.in_channels']
    )
    initial = DataTensor.initial(target)

    # img.show_tensor(target)
    #  img.show_tensor(initial)
    optimizer = Adam(
        model.steps[1].parameters(),
        lr=cfg['train.lr']
    )
    criterion = MSELoss(reduction='sum')

    epochs = []
    losses = []
    for epoch in range(0, cfg['train.epochs']):
        optimizer.zero_grad()
        output_batch = initial.clone().unsqueeze(0)
        for step in range(0, cfg['train.steps']):
            output_batch = model.forward(output_batch, show_output=False)

        loss = criterion(output_batch[0, 0:3, :], target[0:3, :])
        loss.backward()
        optimizer.step()

        if epoch > 10:
            losses.append(loss.item())
            epochs.append(epoch)

        if epoch > 10 and epoch % 10 == 0:
            # plot_loss(epochs, losses)
            img.show_tensor(output_batch[0])

        logging.info(f'Epoch: {epoch} - Loss: {loss.item()}')


if __name__ == "__main__":
    cfg = Config('config.yaml')
    LoggerFactory(cfg['logger']).create()

    model = build_model(cfg)

    train(model)
