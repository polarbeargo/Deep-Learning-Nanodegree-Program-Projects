"""Train exclusively on generated digits; evaluate once on real MNIST test."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.runtime import autocast, grad_scaler


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.fc(self.conv(images))


def train_classifier(images, labels, device, epochs=5, batch_size=128, seed=42):
    """No real loader or pretrained weights enter this function."""
    if len(images) != len(labels) or not len(images):
        raise ValueError("Supply equal, non-empty synthetic image and label tensors.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if images.device.type != "cpu" or labels.device.type != "cpu":
        raise ValueError("Synthetic datasets must be CPU tensors.")
    torch.manual_seed(seed)
    model = SimpleCNN().to(device)
    loader = DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.device(device).type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = grad_scaler(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for batch, targets in loader:
            batch, targets = batch.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device):
                loss = criterion(model(batch), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite synthetic classifier loss.")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach() * len(batch)
        row = dict(epoch=epoch, loss=(total_loss / len(images)).item())
        history.append(row)
        print(f"Synthetic-only classifier epoch {epoch}/{epochs}: loss={row['loss']:.4f}",
              flush=True)
    return model, history


@torch.inference_mode()
def test_accuracy(model, loader, device):
    model.eval()
    correct = torch.zeros((), device=device, dtype=torch.long)
    per_class_correct = torch.zeros(10, device=device, dtype=torch.long)
    per_class_count = torch.zeros_like(per_class_correct)
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(1)
        correct += (predictions == labels).sum()
        per_class_count += torch.bincount(labels, minlength=10)
        per_class_correct += torch.bincount(labels[predictions == labels], minlength=10)
        total += len(labels)
    if total == 0 or torch.any(per_class_count == 0):
        raise ValueError("Evaluation loader must contain real examples of all ten digits.")
    return {
        "accuracy": correct.item() / total, "correct": correct.item(), "total": total,
        "per_class_accuracy": (per_class_correct.float() / per_class_count).cpu().tolist(),
    }
