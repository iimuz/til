import pathlib
import pickle
import random
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# from simpleautoencoder import SimpleAutoencoder
from simpledeepautoencoder import SimpleDeepAutoencoder
from vwapdataset import VwapDataset

logger = getLogger(__name__)


class PyTMinMaxScalerVectorized:
    def __init__(self, min_val, max_val):
        self.min_val = torch.from_numpy(min_val)
        self.scale = torch.from_numpy(1.0 / (max_val - min_val))

    def __call__(self, tensor):
        result = (tensor - self.min_val) * self.scale
        return result


def _init_rand_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _init_loader(train, scaler):
    with open(scaler, "rb") as f:
        sc = pickle.load(f)
    transform = transforms.Compose(
        [transforms.ToTensor(), PyTMinMaxScalerVectorized(sc.data_min_, sc.data_max_)]
    )
    trainset = VwapDataset(pathlib.Path(train), 10, transform=transform)
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=0
    )

    return loader


def _save_diff(inputs, outputs, filepath):
    batch_size = inputs.shape[0]
    display_num = batch_size if batch_size < 10 else 10

    rows, cols = 1, display_num
    plt.figure(figsize=(18, 4))
    for idx in range(display_num):
        plt.subplot(rows, cols, idx + 1)
        plt.plot(inputs.cpu().detach().numpy()[idx].reshape((-1,)), label="input")
        plt.plot(outputs.cpu().detach().numpy()[idx].reshape((-1,)), label="output")
    plt.legend()

    plt.show()
    plt.savefig(filepath)
    plt.cla()
    plt.clf()
    plt.close()


def _save_loss(loss_list, filepath):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, "b")
    plt.show()
    plt.savefig(filepath)
    plt.cla()
    plt.clf()
    plt.close()


def _main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"use device: {device}")

    _init_rand_seed(0)
    loader = _init_loader(
        "_data/interim/dataset/train.pkl", "_data/interim/dataset/scaler.pkl"
    )
    # model = SimpleAutoencoder(10, 5).to(device)
    model = SimpleDeepAutoencoder(10, 6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    loss_list = []
    for epoch in range(100):
        running_loss = 0.0
        for idx, data in enumerate(loader):
            data = data.float().to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_list.append(running_loss / idx)
        running_loss = 0.0

        logger.info(f"loss[{epoch}]: {loss_list[-1]}")
        _save_diff(data, outputs, f"_data/interim/reconstruct/fig_{epoch:05}.png")
        _save_loss(loss_list, "_data/interim/stats/fig_loss.png")


if __name__ == "__main__":
    _main()
