"""Streaming Inception FID and descriptive within-class pixel diversity."""

import warnings

import torch


class FIDEvaluator:
    """Cache real Inception statistics and stream one generated set at a time."""

    def __init__(self, real_images, device="cpu", batch_size=64):
        from torchmetrics.image.fid import FrechetInceptionDistance

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if len(real_images) < 2:
            raise ValueError("FID requires at least two real images.")
        if len(real_images) <= 2048:
            warnings.warn("FID sample count <= feature dimension; covariance is singular "
                          "and estimates are highly biased. Use at least 5000 images.",
                          stacklevel=2)
        self.device = torch.device(device)
        if self.device.type == "mps":
            raise ValueError("Use CPU or CUDA for float64 Inception FID, not MPS.")
        self.batch_size = batch_size
        self.real_count = len(real_images)
        self.metric = FrechetInceptionDistance(
            feature=2048, normalize=True, reset_real_features=False,
        ).to(device)
        self._update(real_images, real=True)

    @torch.inference_mode()
    def _update(self, images, *, real):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("FID expects N x 1 x 28 x 28 normalized MNIST tensors.")
        if not torch.isfinite(images).all() or images.min() < -1.001 or images.max() > 1.001:
            raise ValueError("FID inputs must be finite and normalized to [-1, 1].")
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size].to(self.device)
            rgb = ((batch.float() + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)
            # Torch-fidelity handles its own standard Inception resize to 299x299.
            self.metric.update(rgb, real=real)

    def compute(self, fake_images):
        if len(fake_images) != self.real_count:
            raise ValueError("Use equal real/fake sample counts for comparable FID.")
        self.metric.reset()
        self._update(fake_images, real=False)
        result = float(self.metric.compute().item())
        if not torch.isfinite(torch.tensor(result)):
            raise FloatingPointError("FID computation produced a non-finite score.")
        if result < -0.01:
            raise FloatingPointError(f"FID is unexpectedly negative ({result}); check feature statistics.")
        if result < 0:
            warnings.warn(f"FID {result:.6g} is slightly negative from numerical roundoff; "
                          "reporting zero.", stacklevel=2)
            result = 0.0
        return result


def pixel_diversity(images, labels):
    """Mean within-class pixel standard deviation; noise can inflate this value."""
    values = []
    for digit in range(10):
        batch = images[labels == digit]
        if len(batch) < 2:
            raise ValueError("Diversity requires at least two images per class.")
        values.append(batch.std(dim=0).mean().item())
    return sum(values) / len(values)
