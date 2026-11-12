"""DDPM training, cosine/linear schedules and checkpoint-consistent sampling."""

import copy
import logging
import math
from numbers import Integral
from pathlib import Path

import torch
from torch.nn import functional as F

from utils.checkpoint import CHECKPOINT_DIR, save_checkpoint, save_json
from utils.runtime import autocast, grad_scaler

logger = logging.getLogger(__name__)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def linear_beta_schedule(timesteps):
    _positive_integer(timesteps, "timesteps")
    # Scale the standard 1000-step endpoints so smaller T still reaches noise.
    scale = 1000 / timesteps
    return torch.linspace(scale * 1e-4, scale * 0.02, timesteps).clamp(max=0.999)


def cosine_beta_schedule(timesteps, offset=0.008):
    _positive_integer(timesteps, "timesteps")
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps + offset) / (1 + offset)) * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999).float()


class DiffusionSchedule:
    def __init__(self, timesteps=1000, schedule="linear", device="cpu"):
        builders = {"linear": linear_beta_schedule, "cosine": cosine_beta_schedule}
        if schedule not in builders:
            raise ValueError("schedule must be 'linear' or 'cosine'.")
        self.betas = builders[schedule](timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = self.alphas.cumprod(0)
        self.alpha_bar_previous = F.pad(self.alpha_bar[:-1], (1, 0), value=1)
        self.posterior_variance = (
            self.betas * (1 - self.alpha_bar_previous) / (1 - self.alpha_bar)
        )
        self.timesteps = timesteps
        self.name = schedule

    def q_sample(self, clean, t, noise=None):
        if t.shape != (len(clean),) or t.dtype != torch.long:
            raise ValueError("t must be a length-batch torch.long tensor.")
        if torch.any(t < 0) or torch.any(t >= self.timesteps):
            raise ValueError("Timestep outside the training schedule.")
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("Noise and clean image shapes must match.")
        shape = (-1,) + (1,) * (clean.ndim - 1)
        alpha_bar = self.alpha_bar[t].view(shape)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise


def _noise(shape, device, generator):
    if generator is None:
        return torch.randn(shape, device=device)
    return torch.randn(shape, generator=generator, device="cpu").to(device)


@torch.inference_mode()
def sample_images(model, device, num_samples=16, num_classes=10,
                  img_size=(1, 28, 28), timesteps=None, class_labels=None, *,
                  schedule=None, sampling_steps=None, sampler="ddpm", seed=None,
                  initial_noise=None):
    """Full DDPM or respaced deterministic DDIM on the *trained* noise schedule.

    Lower sampling_steps changes the number of model calls, never the training
    beta schedule. DDPM requires every training step; DDIM can skip steps.
    """
    _positive_integer(num_samples, "num_samples")
    config = getattr(model, "diffusion_config", {})
    train_steps = config.get("timesteps", 1000) if timesteps is None else timesteps
    schedule = config.get("schedule", "linear") if schedule is None else schedule
    if config and (train_steps != config["timesteps"] or schedule != config["schedule"]):
        raise ValueError("Sampling schedule/timesteps must match the trained checkpoint.")
    noise_schedule = DiffusionSchedule(train_steps, schedule, device)
    sampling_steps = train_steps if sampling_steps is None else sampling_steps
    _positive_integer(sampling_steps, "sampling_steps")
    if sampling_steps > train_steps:
        raise ValueError("sampling_steps cannot exceed training timesteps.")
    if sampler not in ("ddpm", "ddim"):
        raise ValueError("sampler must be 'ddpm' or 'ddim'.")
    if sampler == "ddpm" and sampling_steps != train_steps:
        raise ValueError("DDPM requires all training steps. Use DDIM for fewer steps.")
    device = torch.device(device)
    rng = torch.Generator().manual_seed(seed) if seed is not None else None
    if class_labels is None:
        labels = torch.arange(num_samples, device=device) % num_classes
    else:
        labels = torch.as_tensor(class_labels, device=device)
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("Class labels must be integers.")
        labels = labels.long()
        if labels.shape != (num_samples,):
            raise ValueError("Provide one class label per sample.")
        if torch.any(labels < 0) or torch.any(labels >= num_classes):
            raise ValueError("Class label outside the configured digit range.")
    shape = (num_samples, *img_size)
    if initial_noise is not None and tuple(initial_noise.shape) != shape:
        raise ValueError("initial_noise has the wrong shape.")
    images = (_noise(shape, device, rng) if initial_noise is None
              else initial_noise.to(device).clone())
    indices = torch.linspace(train_steps - 1, 0, sampling_steps).long().tolist()
    was_training = model.training
    model.eval()
    try:
        for i, t in enumerate(indices):
            batch_t = torch.full((num_samples,), t, dtype=torch.long, device=device)
            with autocast(device):
                predicted_noise = model(images, batch_t, labels)
            predicted_noise = predicted_noise.float()
            alpha_bar = noise_schedule.alpha_bar[t]
            clean = ((images - (1 - alpha_bar).sqrt() * predicted_noise) /
                     alpha_bar.sqrt()).clamp(-1, 1)
            if sampler == "ddim":
                previous = indices[i + 1] if i + 1 < len(indices) else -1
                previous_bar = noise_schedule.alpha_bar[previous] if previous >= 0 else images.new_tensor(1)
                images = previous_bar.sqrt() * clean + (1 - previous_bar).sqrt() * predicted_noise
            else:
                previous_bar = noise_schedule.alpha_bar_previous[t]
                beta = noise_schedule.betas[t]
                mean = (
                    beta * previous_bar.sqrt() / (1 - alpha_bar) * clean
                    + (1 - previous_bar) * noise_schedule.alphas[t].sqrt()
                    / (1 - alpha_bar) * images
                )
                images = mean
                if t > 0:
                    images = images + noise_schedule.posterior_variance[t].sqrt() * _noise(shape, device, rng)
        if not torch.isfinite(images).all():
            raise FloatingPointError("Reverse diffusion produced non-finite images.")
        return images.clamp(-1, 1), labels
    finally:
        model.train(was_training)


def train_diffusion(model, dataloader, device, num_classes, timesteps=1000,
                    epochs=40, lr=2e-4, *, schedule="cosine",
                    checkpoint_dir=CHECKPOINT_DIR, name=None, save_every=5,
                    ema_decay=0.995, max_batches=None):
    _positive_integer(epochs, "epochs")
    _positive_integer(save_every, "save_every")
    if max_batches is not None:
        _positive_integer(max_batches, "max_batches")
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must lie in [0, 1).")
    name = name or f"diffusion_{schedule}"
    checkpoint_dir = Path(checkpoint_dir)
    model.to(device).train()
    config = dict(model_config=model.config, num_classes=num_classes,
                  timesteps=timesteps, schedule=schedule, ema_decay=ema_decay)
    model.diffusion_config = config
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    noise_schedule = DiffusionSchedule(timesteps, schedule, device)
    scaler = grad_scaler(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    history = []
    consecutive_overflows = 0
    for epoch in range(1, epochs + 1):
        total_loss = torch.zeros((), device=device)
        count = 0
        skipped_batches = 0
        optimizer_steps = 0
        for batch, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t = torch.randint(timesteps, (len(images),), device=device)
            noise = torch.randn_like(images)
            noisy_images = noise_schedule.q_sample(images, t, noise)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                predicted_noise = model(noisy_images, t, labels)
                loss = F.mse_loss(predicted_noise, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite diffusion loss in epoch {epoch}.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradients_finite = torch.stack([
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters() if parameter.grad is not None
            ]).all()
            if not gradients_finite:
                if not scaler.is_enabled():
                    raise FloatingPointError(
                        f"Non-finite diffusion gradients without AMP in epoch {epoch}, "
                        f"batch {batch + 1}; refusing an invalid optimizer update."
                    )
                previous_scale = scaler.get_scale()
                # unscale_ recorded the overflow: step skips AdamW, update backs off.
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                consecutive_overflows += 1
                logger.warning(
                    "%s epoch %d batch %d: AMP gradient overflow; skipped optimizer "
                    "and EMA updates, loss scale %g -> %g.",
                    name, epoch, batch + 1, previous_scale, scaler.get_scale(),
                )
                if consecutive_overflows >= 8:
                    raise FloatingPointError(
                        "Eight consecutive AMP gradient overflows; training is unstable. "
                        "Inspect the data/model and rerun in float32."
                    )
                total_loss += loss.detach() * len(images)
                count += len(images)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            consecutive_overflows = 0
            optimizer_steps += 1
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1 - ema_decay)
            total_loss += loss.detach() * len(images)
            count += len(images)
        if not count:
            raise ValueError("Training loader contains no images.")
        if optimizer_steps == 0:
            raise FloatingPointError(
                f"Epoch {epoch} completed without a valid optimizer step; "
                "refusing to save a success-shaped checkpoint."
            )
        row = dict(epoch=epoch, mse=(total_loss / count).item(), examples=count,
                   optimizer_steps=optimizer_steps, skipped_batches=skipped_batches)
        history.append(row)
        print(f"{name}: epoch {epoch}/{epochs}, MSE={row['mse']:.5f}, "
              f"updates={optimizer_steps}, AMP skips={skipped_batches}", flush=True)
        save_json(history, checkpoint_dir / f"{name}_history.json")
        if epoch % save_every == 0 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, row["mse"], name, checkpoint_dir,
                            config=config, ema=ema, scaler=scaler)
    save_checkpoint(model, optimizer, "final", history[-1]["mse"], name, checkpoint_dir,
                    config=config, ema=ema, scaler=scaler)
    return history
