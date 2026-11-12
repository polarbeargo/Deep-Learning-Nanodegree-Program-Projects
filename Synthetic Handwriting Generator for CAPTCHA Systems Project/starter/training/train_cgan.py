"""Alternating conditional GAN updates with stable binary cross entropy."""

import torch
from torch import nn
from pathlib import Path

from utils.checkpoint import CHECKPOINT_DIR, save_checkpoint, save_json
from utils.runtime import autocast, grad_scaler


def train_cgan(generator, discriminator, dataloader, optimizer_G, optimizer_D,
               criterion, device, z_dim, num_classes, epochs=50, *,
               checkpoint_dir=CHECKPOINT_DIR, save_every=10, max_batches=None):
    if epochs < 1 or save_every < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("epochs, save_every and max_batches must be positive.")
    if not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise ValueError("Use BCEWithLogitsLoss with the discriminator's raw logits.")
    generator.to(device).train()
    discriminator.to(device).train()
    checkpoint_dir = Path(checkpoint_dir)
    scaler_g, scaler_d = grad_scaler(device), grad_scaler(device)
    history = []
    config = dict(z_dim=z_dim, num_classes=num_classes, img_shape=list(generator.img_shape))
    for epoch in range(1, epochs + 1):
        total_g = torch.zeros((), device=device)
        total_d = torch.zeros((), device=device)
        count = 0
        for batch, (images, labels) in enumerate(dataloader):
            if max_batches is not None and batch >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            size = len(images)
            positive = torch.ones(size, 1, device=device)
            negative = torch.zeros_like(positive)
            z = torch.randn(size, z_dim, device=device)
            generated_labels = torch.randint(num_classes, (size,), device=device)

            optimizer_D.zero_grad(set_to_none=True)
            # Reuse the same generated batch for G; detach only for D.
            with autocast(device):
                generated = generator(z, generated_labels)
                real_loss = criterion(discriminator(images, labels), positive)
                fake_loss = criterion(discriminator(generated.detach(), generated_labels), negative)
                d_loss = (real_loss + fake_loss) / 2
            if not torch.isfinite(d_loss):
                raise FloatingPointError(f"Non-finite discriminator loss in epoch {epoch}.")
            scaler_d.scale(d_loss).backward()
            scaler_d.step(optimizer_D)
            scaler_d.update()

            optimizer_G.zero_grad(set_to_none=True)
            discriminator.requires_grad_(False)
            try:
                with autocast(device):
                    g_loss = criterion(discriminator(generated, generated_labels), positive)
                if not torch.isfinite(g_loss):
                    raise FloatingPointError(f"Non-finite generator loss in epoch {epoch}.")
                scaler_g.scale(g_loss).backward()
                scaler_g.step(optimizer_G)
                scaler_g.update()
            finally:
                discriminator.requires_grad_(True)
            total_g += g_loss.detach() * size
            total_d += d_loss.detach() * size
            count += size
        if not count:
            raise ValueError("Training loader contains no images.")
        row = dict(epoch=epoch, g_loss=(total_g / count).item(),
                   d_loss=(total_d / count).item(), examples=count)
        history.append(row)
        print(f"Epoch {epoch}/{epochs}: D={row['d_loss']:.4f}, G={row['g_loss']:.4f}", flush=True)
        save_json(history, checkpoint_dir / "cgan_history.json")
        if epoch % save_every == 0 or epoch == epochs:
            for name, model, optimizer, loss, scaler in (
                ("cgan_generator", generator, optimizer_G, row["g_loss"], scaler_g),
                ("cgan_discriminator", discriminator, optimizer_D, row["d_loss"], scaler_d),
            ):
                save_checkpoint(model, optimizer, epoch, loss, name, checkpoint_dir,
                                config=config, scaler=scaler)
    save_checkpoint(generator, optimizer_G, "final", history[-1]["g_loss"],
                    "cgan_generator", checkpoint_dir, config=config, scaler=scaler_g)
    save_checkpoint(discriminator, optimizer_D, "final", history[-1]["d_loss"],
                    "cgan_discriminator", checkpoint_dir, config=config, scaler=scaler_d)
    return history
