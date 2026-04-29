# Improving Robustness of Deep Learning Models using Consistency-Based Training on CIFAR-10

This project compares three CIFAR-10 robustness approaches:

1. **Baseline ResNet-18** trained on clean CIFAR-10 with cross-entropy.
2. **AugMix ResNet-18** using AugMix views and Jensen-Shannon consistency.
3. **Proposed consistency-based ResNet-18** trained with clean and corrupted views:

```text
L = CE(x) + CE(x') + lambda * KL(p(x) || p(x'))
```

The code is organized so you can run experiments from notebooks for presentation or from scripts for repeatable training.

## Project Structure

```text
configs/                 Hyperparameter configs for the three methods
data/                    CIFAR-10 and CIFAR-10-C data
notebooks/               Presentation-ready experiment notebooks
reports/                 Generated CSV files and figures
scripts/                 CLI training/evaluation helpers
src/robust_cifar/        Reusable project package
checkpoints/             Best model checkpoints
```

## Setup

```bash
pip install -r requirements.txt
```

If you run scripts directly, expose the local package:

```bash
set PYTHONPATH=%CD%\src
```

On PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

## Dataset

The notebooks and scripts can download CIFAR-10 automatically through `torchvision`.

CIFAR-10-C must be downloaded separately from the official benchmark release and extracted into:

```text
data/CIFAR-10-C/
  gaussian_noise.npy
  shot_noise.npy
  ...
  jpeg_compression.npy
  labels.npy
```

## Notebooks

Run these in order:

1. `notebooks/01_baseline_resnet18.ipynb`
2. `notebooks/02_augmix_resnet18.ipynb`
3. `notebooks/03_consistency_training.ipynb`
4. `notebooks/04_evaluation_cifar10c.ipynb`

For quick testing, change `EPOCHS = 100` to `EPOCHS = 1`. For final project results, train for 50-100 epochs.

## Script Usage

Train the baseline:

```bash
python scripts/train_experiment.py --method baseline --epochs 100 --batch-size 64
```

Train AugMix:

```bash
python scripts/train_experiment.py --method augmix --epochs 100 --batch-size 64
```

Train the proposed consistency model:

```bash
python scripts/train_experiment.py --method consistency --epochs 100 --batch-size 64 --lambda-kl 2.0 --warmup-epochs 10
```

Evaluate a checkpoint:

```bash
python scripts/evaluate_checkpoint.py ^
  --checkpoint checkpoints/resnet18_consistency.pt ^
  --model-name proposed ^
  --cifar10c-dir data/CIFAR-10-C
```

## Metrics and Visualizations

The project generates:

- Clean CIFAR-10 accuracy and loss curves.
- CIFAR-10-C accuracy/error per corruption and severity.
- Mean corrupted accuracy.
- Mean corruption error.
- Relative mCE-style score against the project baseline when baseline results are supplied.
- Heatmaps for corruption robustness trends.
- Clean vs corrupted model comparison charts.

## Expected Trend

| Model | Clean Accuracy | Corrupted Accuracy |
|---|---:|---:|
| ResNet-18 baseline | High | Low |
| AugMix | High | Medium |
| Proposed consistency method | High | Higher |

Actual numbers depend on epoch count, hardware, random seed, and whether the full CIFAR-10-C benchmark is evaluated.
