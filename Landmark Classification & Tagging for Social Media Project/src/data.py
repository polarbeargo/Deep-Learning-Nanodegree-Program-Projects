import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing

from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = -1,
    limit: int = -1,
    *,
    multiprocessing_context=None,
    prefetch_factor=None,
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :param multiprocessing_context: start method for worker processes (for example
                        "spawn"). None keeps PyTorch's default (fork on Linux). "spawn"
                        avoids forking a large, multi-threaded notebook kernel.
    :param prefetch_factor: batches queued per worker. None keeps PyTorch's
                        default (2); use 1 to bound shared-memory usage.
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Bound worker count to avoid CPU and memory oversubscription.
        num_workers = min(8, multiprocessing.cpu_count())
    elif num_workers < 0:
        raise ValueError("num_workers must be -1 or a non-negative integer")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not 0 <= valid_size < 1:
        raise ValueError("valid_size must be in the range [0, 1)")
    if prefetch_factor is not None and (
        not isinstance(prefetch_factor, int) or prefetch_factor < 1
    ):
        raise ValueError("prefetch_factor must be a positive integer or None")

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean.tolist(), std.tolist()),
            ]
        ),
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["train"],
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "train",
        transform=data_transforms["valid"],
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        n_tot = min(limit, n_tot)
        indices = indices[:n_tot]

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    # prepare data loaders
    loader_options = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0 and multiprocessing_context is not None:
        loader_options["multiprocessing_context"] = multiprocessing_context
    if num_workers > 0 and prefetch_factor is not None:
        loader_options["prefetch_factor"] = prefetch_factor
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        **loader_options,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        sampler=valid_sampler,
        **loader_options,
    )

    # Now create the test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        transform=data_transforms["test"],
    )

    if limit > 0:
        indices = torch.arange(min(limit, len(test_data)))
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        **loader_options,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    class_names = data_loaders["train"].dataset.classes

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    num_images = min(max_n, len(images))
    fig = plt.figure(figsize=(5 * num_images, 4))
    for idx in range(num_images):
        ax = fig.add_subplot(1, num_images, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):

    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):

    visualize_one_batch(data_loaders, max_n=2)


@pytest.mark.parametrize("num_workers,prefetch_factor", [(0, 1), (2, 1), (2, None)])
def test_loader_prefetch_options(tmp_path, monkeypatch, num_workers, prefetch_factor):
    from PIL import Image
    from unittest.mock import Mock

    for split in ("train", "test"):
        folder = tmp_path / split / "landmark"
        folder.mkdir(parents=True)
        for i in range(3):
            Image.new("RGB", (256, 256), color=(128, 128, 128)).save(folder / f"{i}.png")

    monkeypatch.setattr(
        "src.data.compute_mean_and_std",
        lambda: (torch.tensor([0.5] * 3), torch.tensor([0.25] * 3)),
    )
    monkeypatch.setattr("src.data.get_data_location", lambda: str(tmp_path))
    loader = Mock()
    monkeypatch.setattr(torch.utils.data, "DataLoader", loader)

    get_data_loaders(
        batch_size=2, num_workers=num_workers,
        multiprocessing_context="spawn", prefetch_factor=prefetch_factor,
    )

    assert loader.call_count == 3
    for call in loader.call_args_list:
        options = call.kwargs
        assert options["persistent_workers"] == (num_workers > 0)
        if num_workers > 0:
            assert options["multiprocessing_context"] == "spawn"
        else:
            assert "multiprocessing_context" not in options
        if num_workers > 0 and prefetch_factor is not None:
            assert options["prefetch_factor"] == prefetch_factor
        else:
            assert "prefetch_factor" not in options
