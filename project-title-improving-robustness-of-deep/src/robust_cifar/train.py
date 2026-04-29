from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .losses import jsd_loss, kl_consistency_loss, linear_warmup


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def evaluate_clean(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            total_loss += loss.item() * targets.size(0)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def _save_history(history: list[dict], output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _save_checkpoint(model: nn.Module, output_dir: Path, name: str, epoch: int, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, output_dir / f"{name}.pt")


def train_baseline(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 100,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: torch.device | None = None,
    output_dir: str | Path = "checkpoints",
    run_name: str = "resnet18_baseline",
) -> list[dict]:
    device = device or get_device()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict] = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        for images, targets in tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}/{epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            running_correct += (logits.argmax(1) == targets).sum().item()
            seen += targets.size(0)

        scheduler.step()
        val = evaluate_clean(model, test_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / seen,
            "train_accuracy": running_correct / seen,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            _save_checkpoint(model, Path(output_dir), run_name, epoch + 1, val)
        _save_history(history, Path(output_dir), run_name)
    return history


def train_consistency(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 100,
    lr: float = 0.1,
    lambda_kl: float = 2.0,
    warmup_epochs: int = 10,
    device: torch.device | None = None,
    output_dir: str | Path = "checkpoints",
    run_name: str = "resnet18_consistency",
) -> list[dict]:
    device = device or get_device()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict] = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        warm_lambda = linear_warmup(epoch, warmup_epochs, lambda_kl)
        totals = {"loss": 0.0, "ce_clean": 0.0, "ce_corr": 0.0, "kl": 0.0, "correct": 0.0, "seen": 0.0}
        for clean, corrupted, targets in tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}/{epochs}", leave=False):
            clean = clean.to(device, non_blocking=True)
            corrupted = corrupted.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_clean = model(clean)
            logits_corr = model(corrupted)
            ce_clean = F.cross_entropy(logits_clean, targets)
            ce_corr = F.cross_entropy(logits_corr, targets)
            kl = kl_consistency_loss(logits_clean, logits_corr)
            loss = ce_clean + ce_corr + warm_lambda * kl
            loss.backward()
            optimizer.step()
            batch = targets.size(0)
            totals["loss"] += loss.item() * batch
            totals["ce_clean"] += ce_clean.item() * batch
            totals["ce_corr"] += ce_corr.item() * batch
            totals["kl"] += kl.item() * batch
            totals["correct"] += (logits_clean.argmax(1) == targets).sum().item()
            totals["seen"] += batch

        scheduler.step()
        val = evaluate_clean(model, test_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / totals["seen"],
            "train_accuracy": totals["correct"] / totals["seen"],
            "ce_clean": totals["ce_clean"] / totals["seen"],
            "ce_corrupted": totals["ce_corr"] / totals["seen"],
            "kl": totals["kl"] / totals["seen"],
            "lambda_kl": warm_lambda,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            _save_checkpoint(model, Path(output_dir), run_name, epoch + 1, val)
        _save_history(history, Path(output_dir), run_name)
    return history


def train_augmix(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int = 100,
    lr: float = 0.1,
    jsd_weight: float = 12.0,
    device: torch.device | None = None,
    output_dir: str | Path = "checkpoints",
    run_name: str = "resnet18_augmix",
) -> list[dict]:
    device = device or get_device()
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict] = []
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "jsd": 0.0, "correct": 0.0, "seen": 0.0}
        for clean, aug1, aug2, targets in tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}/{epochs}", leave=False):
            clean = clean.to(device, non_blocking=True)
            aug1 = aug1.to(device, non_blocking=True)
            aug2 = aug2.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_clean = model(clean)
            logits_aug1 = model(aug1)
            logits_aug2 = model(aug2)
            ce = F.cross_entropy(logits_clean, targets)
            jsd = jsd_loss(logits_clean, logits_aug1, logits_aug2)
            loss = ce + jsd_weight * jsd
            loss.backward()
            optimizer.step()
            batch = targets.size(0)
            totals["loss"] += loss.item() * batch
            totals["ce"] += ce.item() * batch
            totals["jsd"] += jsd.item() * batch
            totals["correct"] += (logits_clean.argmax(1) == targets).sum().item()
            totals["seen"] += batch

        scheduler.step()
        val = evaluate_clean(model, test_loader, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / totals["seen"],
            "train_accuracy": totals["correct"] / totals["seen"],
            "ce": totals["ce"] / totals["seen"],
            "jsd": totals["jsd"] / totals["seen"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            _save_checkpoint(model, Path(output_dir), run_name, epoch + 1, val)
        _save_history(history, Path(output_dir), run_name)
    return history
