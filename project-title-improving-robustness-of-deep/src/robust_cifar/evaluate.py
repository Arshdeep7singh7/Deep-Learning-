from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .data import CIFAR10C_CORRUPTIONS, make_cifar10c_loader
from .train import evaluate_clean, get_device


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    accuracy = correct / total
    return {"accuracy": accuracy, "error": 1.0 - accuracy}


def evaluate_cifar10c(
    model: nn.Module,
    cifar10c_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 2,
    device: torch.device | None = None,
    corruptions: tuple[str, ...] = CIFAR10C_CORRUPTIONS,
    severities: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    device = device or get_device()
    model.to(device)
    rows = []
    for corruption in corruptions:
        for severity in severities:
            loader = make_cifar10c_loader(cifar10c_dir, corruption, severity, batch_size=batch_size, num_workers=num_workers)
            metrics = evaluate_loader(model, loader, device)
            rows.append({"corruption": corruption, "severity": severity, **metrics})
    return pd.DataFrame(rows)


def summarize_corruption_results(df: pd.DataFrame, baseline_df: pd.DataFrame | None = None) -> dict[str, float]:
    mean_error = float(df["error"].mean())
    mean_accuracy = float(df["accuracy"].mean())
    summary = {
        "mean_corrupted_accuracy": mean_accuracy,
        "mean_corruption_error": mean_error,
    }
    if baseline_df is not None:
        baseline_error = float(baseline_df["error"].mean())
        summary["relative_mce"] = mean_error / baseline_error if baseline_error > 0 else float("nan")
    return summary


def save_cifar10c_results(df: pd.DataFrame, output_path: str | Path, summary: dict[str, float] | None = None) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if summary is not None:
        with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


def evaluate_full_report(
    model: nn.Module,
    clean_loader,
    cifar10c_dir: str | Path,
    output_csv: str | Path,
    baseline_df: pd.DataFrame | None = None,
    batch_size: int = 128,
    num_workers: int = 2,
    device: torch.device | None = None,
) -> dict[str, float]:
    device = device or get_device()
    clean = evaluate_clean(model, clean_loader, device)
    cifar10c = evaluate_cifar10c(model, cifar10c_dir, batch_size=batch_size, num_workers=num_workers, device=device)
    summary = summarize_corruption_results(cifar10c, baseline_df=baseline_df)
    summary["clean_accuracy"] = float(clean["accuracy"])
    summary["clean_error"] = float(1.0 - clean["accuracy"])
    save_cifar10c_results(cifar10c, output_csv, summary)
    return summary
