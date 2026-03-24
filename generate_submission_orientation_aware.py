"""Generate submission with orientation-group-aware clustering for SeaTurtleID2022.

Strategy:
- SeaTurtleID2022: Orientation-group-aware clustering with predicted orientations
  - side_view (left + right): DBSCAN(eps=0.55)
  - top_view (top + topleft + topright): DBSCAN(eps=0.55)
  - front: Single-image clusters (too few samples)
- Other datasets: Use baseline DBSCAN with dataset-specific eps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures


def relabel_negatives(labels: np.ndarray) -> np.ndarray:
    """Turn DBSCAN noise points (-1) into single-image clusters."""
    labels = np.array(labels, dtype=np.int64)
    neg_indices = np.where(labels == -1)[0]
    if len(neg_indices) == 0:
        return labels

    start = labels[labels >= 0].max() + 1 if np.any(labels >= 0) else 0
    labels[neg_indices] = np.arange(start, start + len(neg_indices), dtype=np.int64)
    return labels


def run_dbscan(features: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Cluster features using DBSCAN."""
    similarity = cosine_similarity(features, features)
    max_sim = float(np.max(similarity))
    if max_sim <= 0:
        distance = np.ones_like(similarity, dtype=np.float32)
    else:
        distance = (max_sim - np.maximum(similarity, 0.0)) / max_sim

    clustering = DBSCAN(eps=eps, metric="precomputed", min_samples=min_samples)
    labels = clustering.fit(distance).labels_
    return relabel_negatives(labels)


def cluster_turtle_orientation_aware(
    dataset,
    features: np.ndarray,
    predictions_df: pd.DataFrame,
    eps_config: dict[str, float],
) -> pd.DataFrame:
    """Cluster SeaTurtleID2022 with orientation-group-aware strategy."""

    # Apply orientation mapping
    orientation_mapping = {
        'top': 'top_view',
        'topleft': 'top_view',
        'topright': 'top_view',
        'left': 'side_view',
        'right': 'side_view',
        'front': 'front',
        'down': 'down',
    }

    predictions_df = predictions_df.copy()
    predictions_df['orientation_group'] = predictions_df['predicted_orientation'].map(orientation_mapping)

    # Merge with metadata
    metadata = dataset.metadata.copy()
    metadata = metadata.merge(
        predictions_df[["image_id", "orientation_group"]],
        on="image_id",
        how="left"
    )

    results = []
    cluster_counter = 0  # Track global cluster ID across orientation groups

    for orientation_group in sorted(predictions_df['orientation_group'].unique()):
        mask = metadata["orientation_group"] == orientation_group
        indices = np.where(mask)[0]

        group_features = features[indices]
        group_metadata = metadata.iloc[indices].copy()

        eps = eps_config.get(orientation_group, 0.55)

        # Cluster this orientation group
        if len(group_features) >= 2:
            labels = run_dbscan(group_features, eps=eps)
        else:
            # Single image or too few: assign unique clusters
            labels = np.arange(len(group_features))

        # Create unique cluster names with sequential global IDs
        cluster_names = [
            f"cluster_SeaTurtleID2022_{cluster_counter + int(label)}"
            for label in labels
        ]

        # Update global counter for next group
        cluster_counter += len(set(labels))

        group_result = pd.DataFrame({
            "image_id": group_metadata["image_id"].astype(int),
            "cluster": cluster_names,
        })

        results.append(group_result)

        n_clusters = len(set(labels))
        print(f"  [{orientation_group:>10}] images={len(indices):<3} eps={eps:.2f} clusters={n_clusters}")

    return pd.concat(results, axis=0, ignore_index=True)


def cluster_baseline_dataset(
    dataset,
    features: np.ndarray,
    eps: float,
    dataset_name: str,
) -> pd.DataFrame:
    """Cluster dataset using baseline DBSCAN."""

    labels = run_dbscan(features, eps=eps)

    cluster_names = [f"cluster_{dataset_name}_{int(label)}" for label in labels]

    result = pd.DataFrame({
        "image_id": dataset.metadata["image_id"].astype(int),
        "cluster": cluster_names,
    })

    n_clusters = len(set(labels))
    print(f"  [{dataset_name}] images={len(dataset):<4} eps={eps:.2f} clusters={n_clusters}")

    return result


def build_submission(
    root: Path,
    output_path: Path,
    predictions_path: Path,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """Build submission with orientation-aware clustering."""

    print("\n" + "=" * 80)
    print("Building Orientation-Aware Submission")
    print("=" * 80)

    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU")
        device = "cpu"

    # Load dataset
    dataset_full = AnimalCLEF2026(
        str(root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    metadata = dataset_full.metadata
    dataset_full = dataset_full.get_subset(metadata["split"] == "test")

    # Split by dataset
    datasets = {}
    for name in dataset_full.metadata["dataset"].unique():
        datasets[name] = dataset_full.get_subset(dataset_full.metadata["dataset"] == name)

    # Load turtle orientation predictions
    predictions_df = pd.read_csv(predictions_path)
    print(f"\nLoaded orientation predictions: {len(predictions_df)} images")

    # Eps configuration
    eps_baseline = {
        "LynxID2025": 0.30,
        "SalamanderID2025": 0.20,
        "TexasHornedLizards": 0.24,
    }

    eps_turtle = {
        "side_view": 0.55,
        "top_view": 0.55,
        "front": 0.75,  # Will likely be all single-image clusters anyway
    }

    results = []

    for name, dataset in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing: {name}")
        print(f"{'='*80}")

        # Choose model
        if name in ["SalamanderID2025", "SeaTurtleID2022"]:
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
            size = 384
            model_name = "MegaDescriptor-L-384"
        elif name in ["LynxID2025", "TexasHornedLizards"]:
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    "conservationxlabs/miewid-msv3",
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                ).eval()
                size = 512
                model_name = "miewid-msv3"
            except AttributeError:
                print(f"  [WARN] MiewID incompatible, falling back to MegaDescriptor")
                model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
                size = 384
                model_name = "MegaDescriptor-L-384"
        else:
            raise ValueError(f"Unknown dataset: {name}")

        # Extract features
        extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)
        transform = T.Compose([
            T.Resize(size=(size, size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        dataset.set_transform(transform)
        features = extractor(dataset).features

        print(f"Model: {model_name}, Features: {features.shape}")

        # Cluster
        if name == "SeaTurtleID2022":
            # Orientation-aware clustering
            result = cluster_turtle_orientation_aware(
                dataset, features, predictions_df, eps_turtle
            )
        else:
            # Baseline clustering
            eps = eps_baseline.get(name, 0.30)
            result = cluster_baseline_dataset(dataset, features, eps, name)

        results.append(result)

    # Combine and save
    submission = pd.concat(results, axis=0, ignore_index=True).sort_values("image_id")
    submission.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("Submission Generated!")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Total images: {len(submission)}")
    print(f"Total clusters: {submission['cluster'].nunique()}")

    return submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission with orientation-aware clustering"
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_orientation_aware.csv"),
        help="Output submission path",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/orientation_analysis/test_orientation_predictions.csv"),
        help="Orientation predictions CSV",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions not found: {args.predictions}")

    build_submission(
        root=args.root,
        output_path=args.output,
        predictions_path=args.predictions,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
