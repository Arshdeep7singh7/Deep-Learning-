from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robust_cifar.data import make_cifar10_loaders
from robust_cifar.models import build_resnet18_cifar
from robust_cifar.train import get_device, seed_everything, train_augmix, train_baseline, train_consistency


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 robustness experiments.")
    parser.add_argument("--method", choices=["baseline", "augmix", "consistency"], required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-kl", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    loaders = make_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mode=args.method,
    )
    model = build_resnet18_cifar()
    run_name = f"resnet18_{args.method}"
    if args.method == "baseline":
        train_baseline(model, *loaders, epochs=args.epochs, lr=args.lr, device=device, output_dir=Path(args.output_dir), run_name=run_name)
    elif args.method == "augmix":
        train_augmix(model, *loaders, epochs=args.epochs, lr=args.lr, device=device, output_dir=Path(args.output_dir), run_name=run_name)
    else:
        train_consistency(
            model,
            *loaders,
            epochs=args.epochs,
            lr=args.lr,
            lambda_kl=args.lambda_kl,
            warmup_epochs=args.warmup_epochs,
            device=device,
            output_dir=Path(args.output_dir),
            run_name=run_name,
        )


if __name__ == "__main__":
    main()
