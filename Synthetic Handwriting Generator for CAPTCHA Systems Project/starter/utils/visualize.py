"""Fixed-scale plots; labels always reflect the conditioning, not recognition."""

import math

import matplotlib.pyplot as plt
import torch


def show_batch(images, labels, n=16, columns=8):
    n = min(n, len(images))
    if n < 1 or columns < 1 or len(labels) < n:
        raise ValueError("Provide images, matching labels and positive columns.")
    columns = min(columns, n)
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.4, rows * 1.6), squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(images[i].detach().cpu().squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
            ax.set_title(f"Label: {int(labels[i])}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig


def compare_samples(real, cgan, diffusion):
    if any(len(images) < 10 for images in (real, cgan, diffusion)):
        raise ValueError("Provide one image of each digit in class order.")
    fig, axes = plt.subplots(3, 10, figsize=(14, 5))
    for row, (name, images) in enumerate(zip(("Real", "cGAN", "Diffusion"), (real, cgan, diffusion))):
        for digit, ax in enumerate(axes[row]):
            ax.imshow(images[digit].detach().cpu().squeeze().numpy(), cmap="gray", vmin=-1, vmax=1)
            if row == 0:
                ax.set_title(str(digit))
            ax.set_xticks([])
            ax.set_yticks([])
        axes[row, 0].set_ylabel(name)
    fig.suptitle("Real vs. cGAN vs. Diffusion (conditioning labels 0-9)")
    fig.tight_layout()
    plt.show()
    return fig


@torch.inference_mode()
def latent_walk(generator, digit, device, steps=10, seed=42):
    if steps < 2 or not 0 <= digit < 10:
        raise ValueError("Use steps >= 2 and a digit between zero and nine.")
    rng = torch.Generator().manual_seed(seed)
    endpoints = torch.randn(2, generator.z_dim, generator=rng).to(device)
    weights = torch.linspace(0, 1, steps, device=device).unsqueeze(1)
    noise = (1 - weights) * endpoints[0] + weights * endpoints[1]
    labels = torch.full((steps,), digit, device=device, dtype=torch.long)
    was_training = generator.training
    generator.eval()
    try:
        return generator(noise, labels).cpu(), labels.cpu()
    finally:
        generator.train(was_training)
