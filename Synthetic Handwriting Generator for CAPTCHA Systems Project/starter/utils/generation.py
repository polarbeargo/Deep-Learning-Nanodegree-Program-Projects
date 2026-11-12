"""Metadata-aware model loading and balanced, batched digit generation."""

import hashlib
from pathlib import Path

import torch

from model.cgan import Generator
from model.diffusion import ConditionalUNet
from training.train_diffusion import sample_images
from utils.runtime import autocast


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_generator(path, kind, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Train the {kind} model first; checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config")
    if not config:
        raise ValueError(f"{path} lacks architecture/schedule metadata; retrain with this implementation.")
    if kind == "cgan":
        model = Generator(**config)
        model.load_state_dict(checkpoint["model_state"])
    elif kind == "diffusion":
        model = ConditionalUNet(**config["model_config"])
        if "ema_state" not in checkpoint:
            raise ValueError("Diffusion checkpoint has no EMA state.")
        model.load_state_dict(checkpoint["ema_state"])
        model.diffusion_config = config
    else:
        raise ValueError("kind must be 'cgan' or 'diffusion'.")
    return model.to(device).eval()


@torch.inference_mode()
def generate_digits(model, kind, labels, device="cpu", batch_size=128, seed=42,
                    sampling_steps=50, sampler="ddim"):
    labels = torch.as_tensor(labels, device="cpu")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("Digit labels must be integers, not floating-point or boolean values.")
    labels = labels.long()
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Provide a non-empty vector of digit labels.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if torch.any(labels < 0) or torch.any(labels > 9):
        raise ValueError("Digit labels must be between zero and nine.")
    if kind not in ("cgan", "diffusion"):
        raise ValueError("kind must be 'cgan' or 'diffusion'.")
    rng = torch.Generator().manual_seed(seed)
    images = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(labels), batch_size):
            batch_labels = labels[start:start + batch_size].to(device)
            if kind == "cgan":
                noise = torch.randn(len(batch_labels), model.z_dim, generator=rng).to(device)
                with autocast(device):
                    batch = model(noise, batch_labels)
            else:
                # Seeds are identical across step-count experiments for paired comparisons.
                batch, _ = sample_images(
                    model, device, num_samples=len(batch_labels), class_labels=batch_labels,
                    sampling_steps=sampling_steps, sampler=sampler, seed=seed + start,
                )
            if not torch.isfinite(batch).all():
                raise FloatingPointError("Generator returned non-finite pixels.")
            images.append(batch.detach().float().cpu().clamp(-1, 1))
    finally:
        model.train(was_training)
    return torch.cat(images), labels


def balanced_labels(per_class):
    if not isinstance(per_class, int) or isinstance(per_class, bool) or per_class < 1:
        raise ValueError("per_class must be a positive integer.")
    return torch.arange(10).repeat_interleave(per_class)


def real_examples(dataset, per_class):
    """Select a reproducible balanced reference set without touching test labels."""
    targets = torch.as_tensor(dataset.targets)
    indices = []
    for digit in range(10):
        candidates = torch.where(targets == digit)[0]
        if len(candidates) < per_class:
            raise ValueError(f"Not enough real examples for digit {digit}.")
        indices.extend(candidates[:per_class].tolist())
    return torch.stack([dataset[index][0] for index in indices]), balanced_labels(per_class)
