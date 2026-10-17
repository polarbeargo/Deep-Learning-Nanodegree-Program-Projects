import tempfile
from pathlib import Path

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=device.type == "cuda")


def train_one_epoch(train_dataloader, model, optimizer, loss, scaler=None):
    """
    Performs one train_one_epoch epoch
    """

    device = _device()
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_loss = 0.0
    num_samples = 0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        if device.type == "cuda":
            data = data.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            output = model(data)
            loss_value = loss(output, target)
        scaler.scale(loss_value).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = data.size(0)
        train_loss += loss_value.detach().item() * batch_size
        num_samples += batch_size

    return train_loss / max(num_samples, 1)


def valid_one_epoch(valid_dataloader, model, loss, return_accuracy=False):
    """
    Validate at the end of one epoch
    """

    device = _device()
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    model.eval()
    valid_loss = 0.0
    num_samples = 0
    correct = 0
    with torch.inference_mode():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if device.type == "cuda":
                data = data.contiguous(memory_format=torch.channels_last)

            with _autocast(device):
                output = model(data)
                loss_value = loss(output, target)
            batch_size = data.size(0)
            valid_loss += loss_value.item() * batch_size
            num_samples += batch_size
            correct += output.argmax(dim=1).eq(target).sum().item()

    average_loss = valid_loss / max(num_samples, 1)
    if return_accuracy:
        return average_loss, 100.0 * correct / max(num_samples, 1)
    return average_loss


def optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs,
    save_path,
    interactive_tracking=False,
    early_stopping_patience=12,
    min_delta=1e-4,
):
    if n_epochs < 1:
        raise ValueError("n_epochs must be a positive integer")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be a positive integer")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")

    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    epochs_without_improvement = 0
    logs = {}
    device = _device()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # HINT: look here: 
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        threshold=1e-3,
        threshold_mode="rel",
        cooldown=1,
        min_lr=1e-6,
    )

    for epoch in range(1, n_epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss, scaler=scaler
        )

        valid_loss, valid_accuracy = valid_one_epoch(
            data_loaders["valid"], model, loss, return_accuracy=True
        )

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} "
            "\tValidation Accuracy: {:.2f}%".format(
                epoch, train_loss, valid_loss, valid_accuracy
            )
        )

        # Save checkpoints only when validation loss improves beyond min_delta.
        if valid_loss_min is None or valid_loss < valid_loss_min - min_delta:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            valid_loss_min = valid_loss
            epochs_without_improvement = 0
            checkpoint_path = Path(save_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["val_accuracy"] = valid_accuracy
            logs["lr"] = epoch_lr

            liveloss.update(logs)
            liveloss.send()

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping after {epoch} epochs: validation loss did not "
                f"improve for {early_stopping_patience} consecutive epochs."
            )
            break

    if valid_loss_min is not None:
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )


def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    device = _device()
    model.to(device)
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    model.eval()
    num_samples = 0
    with torch.inference_mode():
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if device.type == "cuda":
                data = data.contiguous(memory_format=torch.channels_last)

            with _autocast(device):
                logits = model(data)
                loss_value = loss(logits, target)
            batch_size = data.size(0)
            test_loss += loss_value.item() * batch_size
            num_samples += batch_size
            pred = logits.argmax(dim=1)

            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += batch_size

    test_loss /= max(num_samples, 1)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / max(total, 1), correct, total))

    return test_loss


    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
