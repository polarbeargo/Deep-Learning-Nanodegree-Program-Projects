"""
dataloader.py
--------------
Helper functions to load and prepare handwriting datasets.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch

def get_mnist_loaders(batch_size=128, transform=None, *, root=None, num_workers=0,
                      pin_memory=False, seed=42):
    """
    Returns DataLoaders for MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.
        transform (torchvision.transforms): Transformations to apply.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    if root is None:
        root = Path(__file__).resolve().parents[2] / "data"
    train_data = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers:
        options.update(multiprocessing_context="spawn", prefetch_factor=1,
                       persistent_workers=True)
    train_loader = DataLoader(train_data, shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **options)
    test_loader = DataLoader(test_data, shuffle=False, **options)
    
    return train_loader, test_loader
