import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def get_model_transfer_learning(model_name="resnet18", n_classes=50, pretrained=True):

    if n_classes < 1:
        raise ValueError("n_classes must be a positive integer")

    # Get the requested architecture
    if hasattr(models, model_name):

        # "DEFAULT" selects the best available ImageNet weights for this
        # architecture and replaces the deprecated pretrained=True flag
        weights = "DEFAULT" if pretrained else None
        model_transfer = getattr(models, model_name)(weights=weights)

    else:

        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(f"Model {model_name} is not known. List of available models: "
                         f"https://pytorch.org/vision/{torchvision_major_minor}/models.html")

    if not isinstance(getattr(model_transfer, "fc", None), nn.Linear):
        raise ValueError(
            f"Model {model_name} has no final 'fc' linear layer; use a ResNet-style "
            "architecture such as resnet18, resnet50 or resnet152"
        )

    # Freeze all parameters in the model
    # HINT: loop over all parameters. If "param" is one parameter,
    # "param.requires_grad = False" freezes it
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes
    # 1. get numbers of features extracted by the backbone
    num_ftrs = model_transfer.fc.in_features

    # 2. Create a new linear layer with the appropriate number of inputs and
    #    outputs
    model_transfer.fc = nn.Linear(num_ftrs, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"


def test_only_new_head_is_trainable():

    model = get_model_transfer_learning(n_classes=7, pretrained=False)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}

    assert trainable == {"fc.weight", "fc.bias"}, (
        f"Only the new head should be trainable, got {sorted(trainable)}"
    )
    assert model.fc.in_features == 512 and model.fc.out_features == 7
