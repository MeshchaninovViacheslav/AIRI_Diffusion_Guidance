import torch
from torchvision.datasets import MNIST
from torchvision.transforms import (
    Resize,
    Normalize,
    Compose,
    ToTensor
)
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.mnist_transforms = Compose(
            [
                Resize((config.data.image_size, config.data.image_size)),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )
        self.train_loader = DataLoader(
            MNIST(root='../data', download=True, train=True, transform=self.mnist_transforms),
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.valid_loader = DataLoader(
            MNIST(root='../data', download=True, train=False, transform=self.mnist_transforms),
            batch_size=5 * config.training.batch_size,
            shuffle=False,
            drop_last=True
        )

    def sample_train(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def sample_val(self):
        while True:
            for batch in self.valid_loader:
                yield batch
