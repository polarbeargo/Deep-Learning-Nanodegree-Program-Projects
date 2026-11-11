"""Portable, atomically written model checkpoints and JSON histories."""

import json
import os
from pathlib import Path
import tempfile

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def save_checkpoint(model, optimizer=None, epoch=None, loss=None, name="model",
                    path=CHECKPOINT_DIR, *, config=None, ema=None, scaler=None):
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}_epoch{epoch if epoch is not None else 'final'}.pt"
    state = {"model_state": model.state_dict(), "config": config or {}}
    for key, value in (("epoch", epoch), ("loss", loss)):
        if value is not None:
            state[key] = value
    for key, obj in (("optimizer_state", optimizer), ("ema_state", ema),
                     ("scaler_state", scaler)):
        if obj is not None:
            state[key] = obj.state_dict()
    fd, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def load_checkpoint(model, optimizer=None, path=None, map_location="cpu", *, use_ema=False):
    if path is None:
        raise ValueError("Provide the checkpoint path.")
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    key = "ema_state" if use_ema else "model_state"
    if key not in checkpoint:
        raise ValueError(f"{path} does not contain {key}.")
    model.load_state_dict(checkpoint[key])
    config = checkpoint.get("config", {})
    if "schedule" in config and "timesteps" in config:
        model.diffusion_config = config
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
