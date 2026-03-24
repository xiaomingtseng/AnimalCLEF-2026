"""SeaTurtleID2022 orientation-aware clustering analysis.

Step 1: Analyze train set orientation distribution
Step 2: Test set clustering analysis by orientation with eps grid search
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
from sklearn.metrics import silhouette_score
from umap import UMAP
from wildlife_datasets.datasets import AnimalCLEF2026
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity


def analyze_train_orientation_distribution(dataset, outdir: Path) -> None:
    """Step 1: Analyze orientation distribution in train set."""
    print("\n" + "=" * 80)
    print("STEP 1: Train Set Orientation Distribution Analysis")
    print("=" * 80)

    # Train set orientation counts (including NaN)
    print("\n[1.1] Orientation Value Counts (Train Set)")
    print("-" * 80)
    orientation_counts = dataset.metadata["orientation"].value_counts(dropna=False)
    print(orientation_counts)
    total = len(dataset)
    print(f"\nTotal train images: {total}")
    nan_count = dataset.metadata["orientation"].isna().sum()
    if nan_count > 0:
        print(f"Missing orientation: {nan_count} ({100*nan_count/total:.1f}%)")

    # Average images per identity per orientation
    print("\n[1.2] Images per Identity by Orientation")
    print("-" * 80)

    identity_orientation = dataset.metadata.groupby(["identity", "orientation"]).size().reset_index(name="count")

    # Average images per identity for each orientation
    avg_per_orientation = identity_orientation.groupby("orientation")["count"].agg(["mean", "std", "min", "max"])
    print("\nAverage images per identity (by orientation):")
    print(avg_per_orientation)

    # How many orientations does each identity have?
    print("\n[1.3] Orientation Diversity per Identity")
    print("-" * 80)

    identities_with_orientation = dataset.metadata[dataset.metadata["orientation"].notna()]
    orientations_per_identity = identities_with_orientation.groupby("identity")["orientation"].nunique()

    print("\nNumber of orientations per identity:")
    print(orientations_per_identity.value_counts().sort_index())
    print(f"\nMean orientations per identity: {orientations_per_identity.mean():.2f}")
    print(f"Identities with only 1 orientation: {(orientations_per_identity == 1).sum()} / {len(orientations_per_identity)}")
    print(f"Identities with 2+ orientations: {(orientations_per_identity >= 2).sum()} / {len(orientations_per_identity)}")

    # Detail: which orientations appear together?
    print("\n[1.4] Orientation Co-occurrence (same identity)")
    print("-" * 80)

    identity_orientation_sets = identities_with_orientation.groupby("identity")["orientation"].apply(set)

    # Count orientation combinations
    from collections import Counter
    orientation_combos = Counter([tuple(sorted(s)) for s in identity_orientation_sets if len(s) > 1])

    if orientation_combos:
        print("\nTop 10 orientation combinations for multi-orientation identities:")
        for combo, count in orientation_combos.most_common(10):
            print(f"  {combo}: {count} identities")
    else:
        print("\nNo identities with multiple orientations found.")

    # Save detailed analysis
    outdir.mkdir(parents=True, exist_ok=True)

    # Save identity-orientation mapping
    identity_summary = dataset.metadata[dataset.metadata["orientation"].notna()].groupby("identity").agg({
        "orientation": lambda x: ", ".join(sorted(set(x))),
        "image_id": "count"
    }).rename(columns={"image_id": "num_images", "orientation": "orientations"})
    identity_summary["num_orientations"] = identity_summary["orientations"].str.count(",") + 1
    identity_summary.to_csv(outdir / "turtle_train_identity_orientation.csv")
    print(f"\n[OK] Saved: {outdir / 'turtle_train_identity_orientation.csv'}")


def run_dbscan_clustering(features: np.ndarray, eps: float, matcher) -> tuple[np.ndarray, int, int, float]:
    """Run DBSCAN clustering and return labels, n_clusters, n_noise, noise_ratio."""
    similarity = matcher(features, features)
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


def analyze_test_orientation_clustering(
    dataset,
    batch_size: int,
    device: str,
    eps_range: tuple[float, float, float],
    outdir: Path,
) -> None:
    """Step 2: Test set clustering analysis by orientation with eps grid search."""
    print("\n" + "=" * 80)
    print("STEP 2: Test Set Clustering Analysis by Orientation")
    print("=" * 80)

    # Load model and extract features
    print("\n[2.1] Feature Extraction (MegaDescriptor-L-384)")
    print("-" * 80)

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
    extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)

    transform = T.Compose([
        T.Resize(size=(384, 384)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    dataset.set_transform(transform)
    features = extractor(dataset).features
    print(f"[OK] Extracted features: {features.shape}")

    # Get orientations (handle NaN)
    orientations = dataset.metadata["orientation"].dropna().unique()
    orientations = sorted(orientations)

    print(f"\n[2.2] Orientations in Test Set: {orientations}")
    print("-" * 80)

    matcher = CosineSimilarity()

    # Eps grid search range
    eps_start, eps_end, eps_step = eps_range
    eps_values = np.arange(eps_start, eps_end + eps_step, eps_step)

    print(f"Eps grid search range: {eps_start} to {eps_end} (step {eps_step})")
    print(f"Total eps values to test: {len(eps_values)}")

    all_results = []

    for orientation in orientations:
        print(f"\n{'='*80}")
        print(f"Analyzing Orientation: {orientation}")
        print(f"{'='*80}")

        # Filter by orientation
        mask = dataset.metadata["orientation"] == orientation
        indices = np.where(mask)[0]

        orientation_features = features[indices]
        orientation_metadata = dataset.metadata.iloc[indices].copy()

        n_images = len(indices)
        print(f"Images: {n_images}")

        # Eps grid search
        print(f"\n[2.3] Eps Grid Search for '{orientation}'")
        print("-" * 80)
        print(f"{'Eps':<8} {'Clusters':<10} {'Noise':<8} {'Noise %':<10}")
        print("-" * 80)

        orientation_results = []

        for eps in eps_values:
            labels, n_clusters, n_noise, noise_ratio = run_dbscan_clustering(
                orientation_features, eps, matcher
            )

            print(f"{eps:<8.2f} {n_clusters:<10} {n_noise:<8} {noise_ratio:<10.1f}%")

            orientation_results.append({
                "orientation": orientation,
                "eps": eps,
                "n_images": n_images,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": noise_ratio,
            })

        all_results.extend(orientation_results)

        # UMAP visualization at optimal eps (middle of range)
        optimal_eps = (eps_start + eps_end) / 2
        labels, n_clusters, n_noise, noise_ratio = run_dbscan_clustering(
            orientation_features, optimal_eps, matcher
        )

        print(f"\n[2.4] UMAP Visualization (eps={optimal_eps:.2f})")
        print("-" * 80)

        # Compute similarity for UMAP
        similarity = matcher(orientation_features, orientation_features)
        max_sim = float(np.max(similarity))
        distance = (max_sim - np.maximum(similarity, 0.0)) / max_sim if max_sim > 0 else np.ones_like(similarity)

        umap = UMAP(n_components=2, random_state=42, metric="precomputed")
        embedding_2d = umap.fit_transform(distance)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        # Color by cluster
        scatter = ax.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=labels,
            cmap="tab20",
            s=30,
            alpha=0.6,
        )

        ax.set_title(
            f"SeaTurtleID2022 Test Set - Orientation: {orientation}\n"
            f"n={n_images}, eps={optimal_eps:.2f}, clusters={n_clusters}, noise={noise_ratio:.1f}%",
            fontsize=12
        )
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

        fig.tight_layout()
        fig.savefig(outdir / f"turtle_test_{orientation}_umap.png", dpi=150)
        plt.close(fig)

        print(f"[OK] Saved: {outdir / f'turtle_test_{orientation}_umap.png'}")

    # Save eps grid search results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / "turtle_test_eps_grid_search.csv", index=False)
    print(f"\n[OK] Saved: {outdir / 'turtle_test_eps_grid_search.csv'}")

    # Summary table
    print("\n" + "=" * 80)
    print("STEP 2 Summary: Eps Grid Search Results")
    print("=" * 80)

    # For each orientation, show best eps (max clusters with reasonable noise)
    print("\nRecommended eps per orientation (heuristic: max clusters with <80% noise):")
    print("-" * 80)

    for orientation in orientations:
        orient_df = results_df[results_df["orientation"] == orientation]
        # Filter: noise < 80%
        candidates = orient_df[orient_df["noise_ratio"] < 80.0]

        if len(candidates) > 0:
            best = candidates.loc[candidates["n_clusters"].idxmax()]
            print(f"{orientation:>10}: eps={best['eps']:.2f}, clusters={int(best['n_clusters'])}, noise={best['noise_ratio']:.1f}%")
        else:
            print(f"{orientation:>10}: No good eps found (all have >80% noise)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SeaTurtleID2022 orientation-aware clustering analysis"
    )
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
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

    # Load dataset
    dataset_full = AnimalCLEF2026(
        str(args.root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    metadata = dataset_full.metadata

    # Filter: SeaTurtleID2022 only
    turtle_mask = metadata["dataset"] == "SeaTurtleID2022"

    # Step 1: Analyze train set
    train_mask = turtle_mask & (metadata["split"] == "train")
    train_dataset = dataset_full.get_subset(train_mask)

    print(f"\nSeaTurtleID2022 Train Set: {len(train_dataset)} images")
    analyze_train_orientation_distribution(train_dataset, args.outdir)

    # Step 2: Analyze test set clustering
    test_mask = turtle_mask & (metadata["split"] == "test")
    test_dataset = dataset_full.get_subset(test_mask)

    print(f"\nSeaTurtleID2022 Test Set: {len(test_dataset)} images")
    analyze_test_orientation_clustering(
        test_dataset,
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
    print("  1. Review UMAP plots to check if orientation-based splitting improves clustering")
    print("  2. Check eps grid search results to find optimal eps per orientation")
    print("  3. If clustering looks good, proceed to generate submission")


if __name__ == "__main__":
    main()
