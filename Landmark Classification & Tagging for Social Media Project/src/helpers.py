from io import BytesIO
import urllib.request
from zipfile import ZipFile
import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt


# Let's see if we have an available GPU
import numpy as np
import random


def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")

    # Seed random generator for repeatibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Download data if not present already
    download_and_extract()
    compute_mean_and_std()

    # Make checkpoints subdir if not existing
    os.makedirs("checkpoints", exist_ok=True)


def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("landmark_images"):
        data_folder = "landmark_images"
    elif os.path.exists("/data/DLND/C2/landmark_images"):
        data_folder = "/data/DLND/C2/landmark_images"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    
    try:
        
        location = get_data_location()
    
    except IOError:
        # Dataset does not exist
        print(f"Downloading and unzipping {url}. This will take a while...")

        with urllib.request.urlopen(url) as resp:

            with ZipFile(BytesIO(resp.read())) as fp:

                fp.extractall(".")

        print("done")
                
    else:
        
        print(
            "Dataset already downloaded. If you need to re-download, "
            f"please delete the directory {location}"
        )
        return None


# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Style live metrics and overlay an EMA to clarify noisy loss histories."""
    metric = group_name.lower()
    titles = {
        "loss": "Training and validation loss",
        "lr": "Learning rate",
        "val_accuracy": "Validation accuracy",
    }
    ax.set_title(titles.get(metric, group_name.replace("_", " ").title()))
    ax.set_xlabel(x_label)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.03, y=0.08)

    if metric == "loss":
        ax.set_ylabel("Cross-entropy")
        ax.set_ylim(bottom=0)
        for line in tuple(ax.lines):
            if line.get_gid() == "loss-ema":
                line.remove()
        for line in tuple(ax.lines):
            if line.get_gid() not in (None, "loss-raw"):
                continue
            values = line.get_ydata()
            if len(values) < 2:
                continue

            if line.get_gid() is None:
                line.set_gid("loss-raw")
                line.set_label(f"{line.get_label()} (raw)")
            smoothed = [values[0]]
            for value in values[1:]:
                smoothed.append(0.3 * value + 0.7 * smoothed[-1])

            label = line.get_label()
            if label.endswith(" (raw)"):
                label = label[:-6]
            line.set_alpha(0.25)
            ema_line, = ax.plot(
                line.get_xdata(),
                smoothed,
                color=line.get_color(),
                linewidth=2.2,
                label=f"{label} (EMA)",
                zorder=line.get_zorder() + 1,
            )
            ema_line.set_gid("loss-ema")
    elif metric == "lr":
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
    elif metric == "val_accuracy":
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=False)


def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)
