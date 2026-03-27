"""Fine-tune MegaDescriptor backbone with ArcFace loss for wildlife re-identification.

Uses wildlife_tools.train.BasicTrainer with ArcFaceLoss to learn identity-discriminative
embeddings from the labeled training split.

Usage:
  # SeaTurtleID2022 (~30-45 min on RTX 3050)
  uv run python train_embedding.py --root data --dataset SeaTurtleID2022 \\
    --backbone megadescriptor-l --input-size 224 --epochs 10 --batch-size 4 \\
    --lr 1e-5 --device cuda --output-dir models/finetune/turtle

  # LynxID2025 (~15-20 min)
  uv run python train_embedding.py --root data --dataset LynxID2025 \\
    --backbone megadescriptor-l --input-size 224 --epochs 15 --batch-size 4 \\
    --lr 1e-5 --device cuda --output-dir models/finetune/lynx

  # SalamanderID2025 (~10 min)
  uv run python train_embedding.py --root data --dataset SalamanderID2025 \\
    --backbone megadescriptor-l --input-size 224 --epochs 20 --batch-size 8 \\
    --lr 2e-5 --device cuda --output-dir models/finetune/salamander
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.optim as optim
import torchvision.transforms as T

from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.data import ImageDataset
from wildlife_tools.train import ArcFaceLoss, BasicTrainer, set_seed
from wildlife_tools.train.callbacks import EpochCheckpoint

BACKBONES = {
    "megadescriptor-l": ("hf-hub:BVRA/MegaDescriptor-L-384", 1536),
    "megadescriptor-s": ("hf-hub:BVRA/MegaDescriptor-S-224", 768),
}


def build_model(backbone: str, input_size: int) -> tuple[torch.nn.Module, int]:
    """Return (model, embedding_dim)."""
    hub_id, embed_dim = BACKBONES[backbone]
    model = timm.create_model(hub_id, pretrained=True, img_size=input_size)
    return model, embed_dim


def build_transforms(input_size: int) -> T.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize((input_size + 32, input_size + 32)),
        T.RandomCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def load_train_dataset(root: Path, dataset_name: str, transform: T.Compose, min_samples: int = 1) -> ImageDataset:
    """Load the labeled training split for one dataset.

    min_samples: minimum images per identity to include.
    Set to 2 to filter out singleton identities (improves ArcFace training).
    """
    dataset_full = AnimalCLEF2026(
        str(root),
        transform=None,
        load_label=True,
        factorize_label=False,
        check_files=False,
    )
    df = dataset_full.metadata
    mask = (df["dataset"] == dataset_name) & (df["split"] == "train") & df["identity"].notna()
    train_meta = df[mask].reset_index(drop=True)

    if min_samples > 1:
        counts = train_meta["identity"].value_counts()
        keep = counts[counts >= min_samples].index
        train_meta = train_meta[train_meta["identity"].isin(keep)].reset_index(drop=True)

    n_classes = train_meta["identity"].nunique()
    print(f"[{dataset_name}] train images: {len(train_meta)}, identities: {n_classes}")

    return ImageDataset(
        metadata=train_meta,
        root=str(root),
        transform=transform,
        col_path="path",
        col_label="identity",
        load_label=True,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(42, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, embed_dim = build_model(args.backbone, args.input_size)
    transform = build_transforms(args.input_size)

    # SalamanderID2025: filter singletons (47% of classes have only 1 image)
    min_samples = 2 if args.dataset == "SalamanderID2025" else 1
    dataset = load_train_dataset(args.root, args.dataset, transform, min_samples=min_samples)

    num_classes = dataset.num_classes
    print(f"num_classes: {num_classes}, embed_dim: {embed_dim}")

    objective = ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embed_dim,
        margin=0.5,
        scale=64,
    )

    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": args.lr},
        {"params": objective.parameters(), "lr": args.lr * 10},
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    callbacks = EpochCheckpoint(folder=str(args.output_dir), save_step=5)

    trainer = BasicTrainer(
        dataset=dataset,
        model=model,
        objective=objective,
        optimizer=optimizer,
        epochs=args.epochs,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=0,  # Windows: must be 0
        epoch_callback=callbacks,
    )

    print(
        f"\nStarting: {args.dataset} | {args.backbone} | "
        f"input={args.input_size}px | epochs={args.epochs} | "
        f"batch={args.batch_size} | lr={args.lr}"
    )
    trainer.train()

    # Save backbone weights only (not the ArcFace head)
    final_path = args.output_dir / "backbone_final.pth"
    torch.save({"model": model.state_dict(), "embed_dim": embed_dim}, final_path)
    print(f"Saved backbone to: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune embedding backbone with ArcFace loss")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["SeaTurtleID2022", "LynxID2025", "SalamanderID2025"],
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--backbone",
        default="megadescriptor-l",
        choices=list(BACKBONES),
    )
    parser.add_argument("--input-size", type=int, default=224, help="Input image size (px)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("models/finetune"))
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    train(args)


if __name__ == "__main__":
    main()
