"""Run orientation-aware clustering on test set using predicted orientations.

This script uses the orientation predictions from the trained classifier
to split test set into orientation groups and cluster each group separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from sklearn.cluster import DBSCAN
from umap import UMAP
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


def run_dbscan_clustering(features: np.ndarray, eps: float) -> tuple[np.ndarray, int, int, float]:
    """Run DBSCAN clustering and return labels, n_clusters, n_noise, noise_ratio."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute cosine similarity directly
    similarity = cosine_similarity(features, features)
    max_sim = float(np.max(similarity))
    if max_sim <= 0:
        distance = np.ones_like(similarity, dtype=np.float32)
    else:
        distance = (max_sim - np.maximum(similarity, 0.0)) / max_sim

    clustering = DBSCAN(eps=eps, metric="precomputed", min_samples=2)
    labels = clustering.fit(distance).labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels) * 100 if len(labels) > 0 else 0

    return labels, n_clusters, n_noise, noise_ratio


def analyze_orientation_clustering(
    test_dataset,
    predictions_df: pd.DataFrame,
    batch_size: int,
    device: str,
    eps_range: tuple[float, float, float],
    outdir: Path,
) -> pd.DataFrame:
    """Analyze clustering quality for each predicted orientation group."""

    print("\n" + "=" * 80)
    print("Orientation-Aware Clustering Analysis (Test Set)")
    print("=" * 80)

    # Apply orientation mapping (merge similar views)
    print("\n[0] Applying Orientation Mapping")
    print("-" * 80)

    orientation_mapping = {
        'top': 'top_view',
        'topleft': 'top_view',
        'topright': 'top_view',
        'left': 'side_view',
        'right': 'side_view',
        'front': 'front',
        'down': 'down',
    }

    predictions_df['orientation_group'] = predictions_df['predicted_orientation'].map(orientation_mapping)

    print("Original orientations -> Merged groups:")
    for orig, merged in orientation_mapping.items():
        count = (predictions_df['predicted_orientation'] == orig).sum()
        print(f"  {orig:>10} -> {merged:<12} ({count} images)")

    print(f"\nMerged orientation distribution:")
    print(predictions_df['orientation_group'].value_counts())

    # Extract features
    print("\n[1] Feature Extraction (MegaDescriptor-L-384)")
    print("-" * 80)

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
    extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)

    transform = T.Compose([
        T.Resize(size=(384, 384)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_dataset.set_transform(transform)
    features = extractor(test_dataset).features
    print(f"[OK] Extracted features: {features.shape}")

    # Merge predictions with metadata
    metadata = test_dataset.metadata.copy()
    metadata = metadata.merge(
        predictions_df[["image_id", "predicted_orientation", "confidence", "orientation_group"]],
        on="image_id",
        how="left"
    )

    # Get orientation groups
    orientation_groups = sorted(predictions_df["orientation_group"].unique())
    print(f"\n[2] Orientation Groups: {orientation_groups}")
    print("-" * 80)

    matcher = CosineSimilarity()

    # Eps grid search
    eps_start, eps_end, eps_step = eps_range
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step)

    print(f"Eps grid search range: {eps_start} to {eps_end} (step {eps_step})")

    all_results = []

    for orientation_group in orientation_groups:
        print(f"\n{'='*80}")
        print(f"Orientation Group: {orientation_group}")
        print(f"{'='*80}")

        # Filter by orientation group
        mask = metadata["orientation_group"] == orientation_group
        indices = np.where(mask)[0]

        orientation_features = features[indices]
        orientation_metadata = metadata.iloc[indices].copy()

        n_images = len(indices)
        avg_confidence = orientation_metadata["confidence"].mean()

        print(f"Images: {n_images}")
        print(f"Avg confidence: {avg_confidence:.4f}")

        # Show original orientations in this group
        orig_dist = orientation_metadata["predicted_orientation"].value_counts()
        print(f"Original orientations: {dict(orig_dist)}")

        # Eps grid search
        print(f"\n[3] Eps Grid Search for '{orientation_group}'")
        print("-" * 80)
        print(f"{'Eps':<8} {'Clusters':<10} {'Noise':<8} {'Noise %':<10}")
        print("-" * 80)

        orientation_results = []

        for eps in eps_values:
            labels, n_clusters, n_noise, noise_ratio = run_dbscan_clustering(
                orientation_features, eps
            )

            print(f"{eps:<8.2f} {n_clusters:<10} {n_noise:<8} {noise_ratio:<10.1f}%")

            orientation_results.append({
                "orientation_group": orientation_group,
                "eps": eps,
                "n_images": n_images,
                "avg_confidence": avg_confidence,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": noise_ratio,
            })

        all_results.extend(orientation_results)

        # UMAP visualization
        optimal_eps = 0.40  # Default eps
        labels, n_clusters, n_noise, noise_ratio = run_dbscan_clustering(
            orientation_features, optimal_eps
        )

        print(f"\n[4] UMAP Visualization (eps={optimal_eps:.2f})")
        print("-" * 80)

        # Compute distance for UMAP
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(orientation_features, orientation_features)
        max_sim = float(np.max(similarity))
        distance = (max_sim - np.maximum(similarity, 0.0)) / max_sim if max_sim > 0 else np.ones_like(similarity)

        umap = UMAP(n_components=2, random_state=42, metric="precomputed")
        embedding_2d = umap.fit_transform(distance)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap="tab20",
            s=30,
            alpha=0.6,
        )

        ax.set_title(
            f"SeaTurtleID2022 Test - Orientation Group: {orientation_group}\n"
            f"n={n_images}, eps={optimal_eps:.2f}, clusters={n_clusters}, noise={noise_ratio:.1f}%\n"
            f"Avg classifier confidence: {avg_confidence:.2f}",
            fontsize=11
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        fig.tight_layout()
        fig.savefig(outdir / f"test_group_{orientation_group}_umap.png", dpi=150)
        plt.close(fig)

        print(f"[OK] Saved: {outdir / f'test_group_{orientation_group}_umap.png'}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / "test_orientation_group_clustering.csv", index=False)
    print(f"\n[OK] Saved: {outdir / 'test_orientation_group_clustering.csv'}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: Recommended Eps per Orientation Group")
    print("=" * 80)

    for orientation_group in orientation_groups:
        orient_df = results_df[results_df["orientation_group"] == orientation_group]
        candidates = orient_df[orient_df["noise_ratio"] < 80.0]

        if len(candidates) > 0:
            best = candidates.loc[candidates["n_clusters"].idxmax()]
            print(
                f"{orientation_group:>12}: eps={best['eps']:.2f}, "
                f"clusters={int(best['n_clusters'])}, "
                f"noise={best['noise_ratio']:.1f}%"
            )
        else:
            best = orient_df.loc[orient_df["noise_ratio"].idxmin()]
            print(
                f"{orientation_group:>12}: eps={best['eps']:.2f}, "
                f"clusters={int(best['n_clusters'])}, "
                f"noise={best['noise_ratio']:.1f}% (best available, all >80% noise)"
            )

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orientation-aware clustering on test set with predicted orientations"
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("reports/orientation_analysis/test_orientation_predictions.csv"),
        help="Path to orientation predictions CSV",
    )
    parser.add_argument(
        "--eps-start", type=float, default=0.30, help="Eps grid search start"
    )
    parser.add_argument(
        "--eps-end", type=float, default=0.80, help="Eps grid search end"
    )
    parser.add_argument(
        "--eps-step", type=float, default=0.05, help="Eps grid search step"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("reports/orientation_analysis"),
        help="Output directory",
    )
    args = parser.parse_args()

    # Load predictions
    predictions_df = pd.read_csv(args.predictions)
    print(f"Loaded orientation predictions: {len(predictions_df)} images")

    # Load dataset
    dataset_full = AnimalCLEF2026(
        str(args.root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    metadata = dataset_full.metadata

    # Filter: SeaTurtleID2022 test set
    turtle_test_mask = (metadata["dataset"] == "SeaTurtleID2022") & (metadata["split"] == "test")
    turtle_test_dataset = dataset_full.get_subset(turtle_test_mask)

    print(f"SeaTurtleID2022 Test Set: {len(turtle_test_dataset)} images")

    # Run analysis
    results_df = analyze_orientation_clustering(
        turtle_test_dataset,
        predictions_df,
        batch_size=args.batch_size,
        device=args.device,
        eps_range=(args.eps_start, args.eps_end, args.eps_step),
        outdir=args.outdir,
    )

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {args.outdir}")
    print("\nNext steps:")
    print("  1. Review UMAP plots for each orientation group")
    print("  2. Compare clustering quality to baseline (no orientation split)")
    print("  3. If improved, generate submission using orientation-group-aware clustering")


if __name__ == "__main__":
    main()
