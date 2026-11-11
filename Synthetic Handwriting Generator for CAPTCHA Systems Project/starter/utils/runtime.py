"""Reproducible runtime configuration shared by notebooks and scripts."""

import random

import numpy as np
import torch


def setup(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Fixed MNIST sizes benefit from autotuning; not bitwise deterministic.
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def grad_scaler(device):
    return torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda")


def autocast(device):
    device = torch.device(device)
    return torch.autocast(device_type=device.type, enabled=device.type == "cuda")
