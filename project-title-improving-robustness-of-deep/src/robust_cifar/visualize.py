from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torchvision.utils import make_grid

from .data import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD

sns.set_theme(style="whitegrid", context="notebook")


def load_history(path: str | Path) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def plot_training_history(history: pd.DataFrame, title: str, output_path: str | Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title(f"{title}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].set_title(f"{title}: accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_corruption_heatmap(df: pd.DataFrame, metric: str = "accuracy", title: str = "CIFAR-10-C", output_path: str | Path | None = None):
    table = df.pivot(index="corruption", columns="severity", values=metric)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(table, annot=True, fmt=".2f", cmap="viridis" if metric == "accuracy" else "magma_r", ax=ax)
    ax.set_title(f"{title}: {metric} by corruption and severity")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption")
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def plot_model_comparison(summary: pd.DataFrame, output_path: str | Path | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(summary, x="model", y="clean_accuracy", ax=axes[0], palette="Set2")
    axes[0].set_title("Clean CIFAR-10 accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(summary, x="model", y="mean_corrupted_accuracy", ax=axes[1], palette="Set2")
    axes[1].set_title("Mean CIFAR-10-C accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1).to(images.device)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1).to(images.device)
    return torch.clamp(images * std + mean, 0, 1)


def show_batch(images: torch.Tensor, labels: torch.Tensor | None = None, nrow: int = 8, title: str = "CIFAR-10 batch"):
    grid = make_grid(denormalize(images.cpu()), nrow=nrow)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    ax.set_title(title)
    if labels is not None:
        label_names = [CIFAR10_CLASSES[int(x)] for x in labels[: min(len(labels), nrow)].cpu()]
        ax.set_xlabel(", ".join(label_names))
    fig.tight_layout()
    return fig
