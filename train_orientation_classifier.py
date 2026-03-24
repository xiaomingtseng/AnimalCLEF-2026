"""Train orientation classifier for SeaTurtleID2022.

Strategy:
1. Train lightweight CNN classifier on train set (8526 images with orientation labels)
2. Predict orientation for test set (500 images)
3. Use predicted orientations for orientation-aware clustering
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from wildlife_datasets.datasets import AnimalCLEF2026


class OrientationDataset(Dataset):
    """Dataset for orientation classification."""

    def __init__(self, wildlife_dataset, transform=None):
        self.dataset = wildlife_dataset
        self.transform = transform

        # Filter out images without orientation labels
        valid_mask = wildlife_dataset.metadata["orientation"].notna()
        self.indices = np.where(valid_mask)[0]

        # Get orientation labels
        orientations = wildlife_dataset.metadata.iloc[self.indices]["orientation"].values

        # Map orientations to class indices
        self.classes = sorted(set(orientations))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels = [self.class_to_idx[o] for o in orientations]

        print(f"Dataset size: {len(self.indices)}")
        print(f"Classes ({len(self.classes)}): {self.classes}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.dataset[real_idx][0]  # Get image (already PIL)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def train_orientation_classifier(
    train_dataset,
    val_dataset,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    model_path: Path,
) -> tuple[nn.Module, list[str]]:
    """Train orientation classifier."""

    print("\n" + "=" * 80)
    print("Training Orientation Classifier")
    print("=" * 80)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model: EfficientNet-B0 (lightweight and accurate)
    num_classes = len(train_dataset.classes)
    print(f"\nModel: EfficientNet-B0")
    print(f"Number of classes: {num_classes}")

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_dataset.classes,
                    "class_to_idx": train_dataset.class_to_idx,
                    "val_acc": val_acc,
                },
                model_path,
            )
            print(f"  -> Saved best model (val_acc: {val_acc:.2f}%)")

        scheduler.step()

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}%")

    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, train_dataset.classes


def evaluate_classifier(model, dataset, device: str, classes: list[str]) -> None:
    """Evaluate classifier and print metrics."""

    print("\n" + "=" * 80)
    print("Evaluation on Validation Set")
    print("=" * 80)

    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


def predict_test_orientations(
    model,
    test_dataset_wildlife,
    classes: list[str],
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    """Predict orientations for test set."""

    print("\n" + "=" * 80)
    print("Predicting Test Set Orientations")
    print("=" * 80)

    model.eval()

    # Create transform
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_dataset_wildlife.set_transform(transform)

    # Create dataloader
    loader = DataLoader(test_dataset_wildlife, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images in loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to orientation labels
    predicted_orientations = [classes[p] for p in all_preds]

    # Get max probability (confidence)
    confidences = [probs[p] for p, probs in zip(all_preds, all_probs)]

    # Create result dataframe
    result = pd.DataFrame({
        "image_id": test_dataset_wildlife.metadata["image_id"].values,
        "predicted_orientation": predicted_orientations,
        "confidence": confidences,
    })

    print(f"\nPredicted orientations for {len(result)} test images")
    print("\nOrientation distribution:")
    print(result["predicted_orientation"].value_counts())

    print(f"\nMean confidence: {result['confidence'].mean():.4f}")
    print(f"Min confidence: {result['confidence'].min():.4f}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train orientation classifier")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/orientation_classifier.pth"),
        help="Model save path",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/orientation_analysis"),
        help="Output directory",
    )
    args = parser.parse_args()

    # Create output directories
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset_full = AnimalCLEF2026(
        str(args.root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    metadata = dataset_full.metadata

    # Filter: SeaTurtleID2022 train set
    turtle_train_mask = (metadata["dataset"] == "SeaTurtleID2022") & (metadata["split"] == "train")
    turtle_train_dataset = dataset_full.get_subset(turtle_train_mask)

    # Split train into train/val (80/20)
    n_train = len(turtle_train_dataset)
    indices = np.arange(n_train)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(0.8 * n_train)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Create train/val subsets
    train_subset = turtle_train_dataset.get_subset(train_indices)
    val_subset = turtle_train_dataset.get_subset(val_indices)

    # Data transforms
    train_transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    val_transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Create datasets
    print("\n[Train Dataset]")
    train_dataset = OrientationDataset(train_subset, transform=train_transform)

    print("\n[Validation Dataset]")
    val_dataset = OrientationDataset(val_subset, transform=val_transform)

    # Train classifier
    model, classes = train_orientation_classifier(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_path=args.model_path,
    )

    # Evaluate
    evaluate_classifier(model, val_dataset, args.device, classes)

    # Predict test set orientations
    turtle_test_mask = (metadata["dataset"] == "SeaTurtleID2022") & (metadata["split"] == "test")
    turtle_test_dataset = dataset_full.get_subset(turtle_test_mask)

    predictions = predict_test_orientations(
        model, turtle_test_dataset, classes, args.device, args.batch_size
    )

    # Save predictions
    predictions.to_csv(args.outdir / "test_orientation_predictions.csv", index=False)
    print(f"\n[OK] Saved predictions: {args.outdir / 'test_orientation_predictions.csv'}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model saved: {args.model_path}")
    print(f"Predictions saved: {args.outdir / 'test_orientation_predictions.csv'}")
    print("\nNext step: Run orientation-aware clustering on test set using predicted orientations")


if __name__ == "__main__":
    main()
