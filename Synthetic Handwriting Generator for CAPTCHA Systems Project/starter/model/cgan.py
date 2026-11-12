"""Convolutional class-conditional GAN for normalized MNIST images."""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        if tuple(img_shape) != (1, 28, 28):
            raise ValueError("This generator supports 1x28x28 MNIST images.")
        if z_dim < 1 or num_classes < 1:
            raise ValueError("z_dim and num_classes must be positive.")
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.img_shape = tuple(img_shape)
        self.label_emb = nn.Embedding(num_classes, 32)
        self.project = nn.Sequential(
            nn.Linear(z_dim + 32, 128 * 7 * 7),
            nn.Unflatten(1, (128, 7, 7)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), dim=1)
        return self.model(self.project(x))


class Discriminator(nn.Module):
    """Return logits for numerically stable BCEWithLogitsLoss."""

    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        if tuple(img_shape) != (1, 28, 28):
            raise ValueError("This discriminator supports 1x28x28 MNIST images.")
        self.label_emb = nn.Embedding(num_classes, 28 * 28)
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, 28, 28)
        return self.model(torch.cat((img, label_map), dim=1))
