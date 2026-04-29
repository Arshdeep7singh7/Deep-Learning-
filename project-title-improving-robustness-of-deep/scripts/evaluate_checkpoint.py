from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robust_cifar.data import make_cifar10_loaders
from robust_cifar.evaluate import evaluate_full_report
from robust_cifar.models import build_resnet18_cifar, load_checkpoint
from robust_cifar.train import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on CIFAR-10 and CIFAR-10-C.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cifar10c-dir", default="data/CIFAR-10-C")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--baseline-results", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    _, test_loader = make_cifar10_loaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, mode="baseline")
    model = build_resnet18_cifar()
    load_checkpoint(model, args.checkpoint, device=device)
    baseline_df = pd.read_csv(args.baseline_results) if args.baseline_results else None
    output_csv = Path(args.output_dir) / f"{args.model_name}_cifar10c.csv"
    summary = evaluate_full_report(
        model,
        test_loader,
        args.cifar10c_dir,
        output_csv,
        baseline_df=baseline_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    print(summary)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
