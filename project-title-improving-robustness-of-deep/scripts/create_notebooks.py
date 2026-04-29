from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.strip().splitlines(True)}


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


common_setup = """
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import torch
import matplotlib.pyplot as plt

from robust_cifar.data import make_cifar10_loaders, CIFAR10_CLASSES
from robust_cifar.models import build_resnet18_cifar, count_parameters
from robust_cifar.train import get_device, seed_everything

seed_everything(42)
device = get_device()
device
"""


baseline = nb(
    [
        md(
            """
# Notebook 01: Baseline ResNet-18 on CIFAR-10

This notebook trains a CIFAR-adapted ResNet-18 using standard cross-entropy on clean CIFAR-10. It is the reference model for clean accuracy and corruption robustness comparisons.

Expected role in the project:

- Establish the clean-data baseline.
- Save training curves and best checkpoint.
- Provide the reference error used for relative mCE-style comparison.
"""
        ),
        code(common_setup),
        md("## Dataset and Loader\n\nCIFAR-10 has 50,000 training images and 10,000 test images across 10 classes."),
        code(
            """
train_loader, test_loader = make_cifar10_loaders(
    data_dir=PROJECT_ROOT / "data",
    batch_size=64,
    num_workers=2,
    mode="baseline",
    download=True,
)
len(train_loader.dataset), len(test_loader.dataset), CIFAR10_CLASSES
"""
        ),
        md("## Model\n\nThe first ResNet convolution is changed to a 3x3 stride-1 convolution and max-pooling is removed, which is standard for 32x32 CIFAR images."),
        code(
            """
model = build_resnet18_cifar()
print(model)
print(f"Trainable parameters: {count_parameters(model):,}")
"""
        ),
        md("## Training\n\nFor a quick smoke test, set `EPOCHS = 1`. For final project numbers use 50-100 epochs."),
        code(
            """
from robust_cifar.train import train_baseline

EPOCHS = 100
history = train_baseline(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    lr=0.1,
    device=device,
    output_dir=PROJECT_ROOT / "checkpoints",
    run_name="resnet18_baseline",
)
pd.DataFrame(history).tail()
"""
        ),
        md("## Training Curves"),
        code(
            """
from robust_cifar.visualize import plot_training_history

history_df = pd.DataFrame(history)
plot_training_history(
    history_df,
    title="Baseline ResNet-18",
    output_path=PROJECT_ROOT / "reports" / "figures" / "baseline_training.png",
)
plt.show()
"""
        ),
    ]
)


augmix = nb(
    [
        md(
            """
# Notebook 02: AugMix Robustness Baseline

AugMix creates multiple stochastic augmentation chains, mixes them, and adds a Jensen-Shannon consistency penalty. This is a strong published baseline for common corruption robustness.
"""
        ),
        code(common_setup),
        md("## AugMix Loader\n\nEach batch returns a clean image plus two AugMix views used for JSD consistency training."),
        code(
            """
train_loader, test_loader = make_cifar10_loaders(
    data_dir=PROJECT_ROOT / "data",
    batch_size=64,
    num_workers=2,
    mode="augmix",
    download=True,
)
batch = next(iter(train_loader))
[x.shape if hasattr(x, "shape") else len(x) for x in batch]
"""
        ),
        md("## Train AugMix Model"),
        code(
            """
from robust_cifar.train import train_augmix

model = build_resnet18_cifar()
EPOCHS = 100
history = train_augmix(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    lr=0.1,
    jsd_weight=12.0,
    device=device,
    output_dir=PROJECT_ROOT / "checkpoints",
    run_name="resnet18_augmix",
)
pd.DataFrame(history).tail()
"""
        ),
        md("## AugMix Curves"),
        code(
            """
from robust_cifar.visualize import plot_training_history

history_df = pd.DataFrame(history)
plot_training_history(
    history_df,
    title="AugMix ResNet-18",
    output_path=PROJECT_ROOT / "reports" / "figures" / "augmix_training.png",
)
plt.show()
"""
        ),
    ]
)


consistency = nb(
    [
        md(
            """
# Notebook 03: Proposed Consistency-Based Training

The proposed method trains the model to keep predictions stable between clean inputs and lightly corrupted inputs. The objective is:

`L = CE(x) + CE(x') + lambda * KL(p(x) || p(x'))`

The KL coefficient is warmed up during early epochs so the network first learns class-discriminative features before the consistency term becomes strong.
"""
        ),
        code(common_setup),
        md("## Consistency Loader\n\nEach training item returns `(clean_image, corrupted_image, label)` using controlled random corruptions: Gaussian noise, blur, JPEG compression, brightness, and contrast."),
        code(
            """
train_loader, test_loader = make_cifar10_loaders(
    data_dir=PROJECT_ROOT / "data",
    batch_size=64,
    num_workers=2,
    mode="consistency",
    corruption_severity=2,
    download=True,
)
clean, corrupted, labels = next(iter(train_loader))
clean.shape, corrupted.shape, labels.shape
"""
        ),
        md("## Visual Check of Clean and Corrupted Views"),
        code(
            """
from robust_cifar.visualize import show_batch

show_batch(clean[:16], labels[:16], title="Clean training views")
plt.show()
show_batch(corrupted[:16], labels[:16], title="Corrupted consistency views")
plt.show()
"""
        ),
        md("## Train Proposed Model"),
        code(
            """
from robust_cifar.train import train_consistency

model = build_resnet18_cifar()
EPOCHS = 100
history = train_consistency(
    model,
    train_loader,
    test_loader,
    epochs=EPOCHS,
    lr=0.1,
    lambda_kl=2.0,
    warmup_epochs=10,
    device=device,
    output_dir=PROJECT_ROOT / "checkpoints",
    run_name="resnet18_consistency",
)
pd.DataFrame(history).tail()
"""
        ),
        md("## Accuracy, KL, and Warm-Up Trends"),
        code(
            """
from robust_cifar.visualize import plot_training_history

history_df = pd.DataFrame(history)
plot_training_history(
    history_df,
    title="Consistency ResNet-18",
    output_path=PROJECT_ROOT / "reports" / "figures" / "consistency_training.png",
)
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(history_df["epoch"], history_df["kl"], label="KL loss", color="tab:blue")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("KL loss")
ax2 = ax1.twinx()
ax2.plot(history_df["epoch"], history_df["lambda_kl"], label="lambda", color="tab:orange")
ax2.set_ylabel("KL weight")
fig.suptitle("Consistency warm-up behavior")
fig.tight_layout()
fig.savefig(PROJECT_ROOT / "reports" / "figures" / "consistency_warmup.png", dpi=180, bbox_inches="tight")
plt.show()
"""
        ),
    ]
)


evaluation = nb(
    [
        md(
            """
# Notebook 04: Evaluation on CIFAR-10 and CIFAR-10-C

This notebook evaluates trained checkpoints on clean CIFAR-10 and CIFAR-10-C. CIFAR-10-C must be downloaded separately and extracted as `data/CIFAR-10-C/*.npy`.

Outputs:

- Per-corruption and per-severity accuracy/error CSV files.
- Heatmaps for corruption trends.
- A model comparison chart for clean and corrupted accuracy.
"""
        ),
        code(common_setup),
        md("## Load Checkpoints\n\nRun the training notebooks first, or change these paths to your checkpoint locations."),
        code(
            """
from robust_cifar.models import load_checkpoint

checkpoint_paths = {
    "Baseline": PROJECT_ROOT / "checkpoints" / "resnet18_baseline.pt",
    "AugMix": PROJECT_ROOT / "checkpoints" / "resnet18_augmix.pt",
    "Proposed": PROJECT_ROOT / "checkpoints" / "resnet18_consistency.pt",
}

models = {}
for name, path in checkpoint_paths.items():
    model = build_resnet18_cifar()
    if path.exists():
        load_checkpoint(model, path, device=device)
        models[name] = model.to(device)
    else:
        print(f"Missing checkpoint for {name}: {path}")
models.keys()
"""
        ),
        md("## Evaluate Clean Accuracy"),
        code(
            """
_, test_loader = make_cifar10_loaders(
    data_dir=PROJECT_ROOT / "data",
    batch_size=128,
    num_workers=2,
    mode="baseline",
    download=True,
)

from robust_cifar.train import evaluate_clean

clean_rows = []
for name, model in models.items():
    metrics = evaluate_clean(model, test_loader, device)
    clean_rows.append({"model": name, "clean_accuracy": metrics["accuracy"], "clean_error": 1 - metrics["accuracy"]})
clean_df = pd.DataFrame(clean_rows)
clean_df
"""
        ),
        md("## Evaluate CIFAR-10-C\n\nThis cell can take time because it evaluates 15 corruptions x 5 severity levels for each model."),
        code(
            """
from robust_cifar.evaluate import evaluate_cifar10c, summarize_corruption_results, save_cifar10c_results

cifar10c_dir = PROJECT_ROOT / "data" / "CIFAR-10-C"
cifar10c_results = {}
summary_rows = []

baseline_df = None
for name, model in models.items():
    df = evaluate_cifar10c(model, cifar10c_dir, batch_size=128, num_workers=2, device=device)
    summary = summarize_corruption_results(df, baseline_df=baseline_df)
    if name == "Baseline":
        baseline_df = df
        summary = summarize_corruption_results(df)
    clean_row = clean_df.loc[clean_df["model"] == name].iloc[0].to_dict()
    summary_rows.append({"model": name, **clean_row, **summary})
    cifar10c_results[name] = df
    save_cifar10c_results(df, PROJECT_ROOT / "reports" / f"{name.lower()}_cifar10c.csv", summary)

summary_df = pd.DataFrame(summary_rows)
summary_df
"""
        ),
        md("## Corruption Heatmaps and Model Comparison"),
        code(
            """
from robust_cifar.visualize import plot_corruption_heatmap, plot_model_comparison

for name, df in cifar10c_results.items():
    plot_corruption_heatmap(
        df,
        metric="accuracy",
        title=name,
        output_path=PROJECT_ROOT / "reports" / "figures" / f"{name.lower()}_cifar10c_heatmap.png",
    )
    plt.show()

plot_model_comparison(
    summary_df,
    output_path=PROJECT_ROOT / "reports" / "figures" / "model_comparison.png",
)
plt.show()
"""
        ),
        md("## Expected Interpretation\n\nThe baseline should usually keep high clean accuracy but drop sharply on noise and blur corruptions. AugMix should improve mean corrupted accuracy. The proposed consistency method should improve corrupted accuracy while preserving clean accuracy, especially on corruptions similar to the controlled training corruptions."),
    ]
)


def write_notebook(name: str, notebook: dict) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        f.write("\n")


def main() -> None:
    write_notebook("01_baseline_resnet18.ipynb", baseline)
    write_notebook("02_augmix_resnet18.ipynb", augmix)
    write_notebook("03_consistency_training.ipynb", consistency)
    write_notebook("04_evaluation_cifar10c.ipynb", evaluation)


if __name__ == "__main__":
    main()
