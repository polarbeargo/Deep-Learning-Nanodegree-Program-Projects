import os

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets
import torchvision.transforms as T
from .helpers import get_data_location


class Predictor(nn.Module):

    def __init__(self, model, class_names, mean, std):
        super().__init__()

        self.model = model.eval()
        self.class_names = class_names

        # We use nn.Sequential and not nn.Compose because the former
        # is compatible with torch.script, while the latter isn't
        self.transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean.tolist(), std.tolist())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.transforms(x)
            logits = self.model(x)
            return F.softmax(logits, dim=1)


def predictor_test(test_dataloader, model_reloaded):

    folder = get_data_location()
    test_data = datasets.ImageFolder(os.path.join(folder, "test"), transform=T.ToTensor())

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=test_dataloader.batch_size or 32,
        num_workers=test_dataloader.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    try:
        device = next(model_reloaded.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    predicted, actual = [], []
    model_reloaded.eval()
    with torch.inference_mode():
        for images, labels in tqdm(
            test_loader, total=len(test_loader), leave=True, ncols=80
        ):
            probabilities = model_reloaded(images.to(device, non_blocking=True))
            predicted.extend(probabilities.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())

    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    print(f"Accuracy: {(predicted == actual).mean():.4f}")
    return predicted, actual


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    from .model import MyModel
    from .helpers import compute_mean_and_std

    mean, std = compute_mean_and_std()

    model = MyModel(num_classes=3, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    predictor = Predictor(model, class_names=['a', 'b', 'c'], mean=mean, std=std)

    out = predictor(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 3]
    ), f"Expected an output tensor of size (2, 3), got {out.shape}"

    assert torch.isclose(
        out[0].sum(),
        torch.Tensor([1]).squeeze()
    ), "The output of the .forward method should be a softmax vector with sum = 1"
