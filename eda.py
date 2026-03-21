"""EDA utilities for AnimalCLEF 2026.

This script provides:
1) Dataset/split/species distribution summaries
2) Image size/aspect-ratio exploration
3) UMAP visualization + clustering diagnostics for embedding quality

Usage example:
    uv run python eda.py --root data --outdir reports/eda --run-umap --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def plot_bar_from_series(series: pd.Series, title: str, out_path: Path, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def distribution_eda(metadata: pd.DataFrame, outdir: Path) -> None:
    ensure_dir(outdir)

    by_dataset = metadata["dataset"].value_counts().sort_index().rename_axis("dataset").reset_index(name="count")
    by_species = metadata["species"].value_counts().sort_index().rename_axis("species").reset_index(name="count")
    by_split = metadata["split"].value_counts().sort_index().rename_axis("split").reset_index(name="count")
    by_dataset_split = (
        metadata.groupby(["dataset", "split"]).size().rename("count").reset_index().sort_values(["dataset", "split"])
    )

    save_table(by_dataset, outdir / "counts_by_dataset.csv")
    save_table(by_species, outdir / "counts_by_species.csv")
    save_table(by_split, outdir / "counts_by_split.csv")
    save_table(by_dataset_split, outdir / "counts_by_dataset_split.csv")

    plot_bar_from_series(
        metadata["dataset"].value_counts().sort_index(),
        title="Image Count by Dataset",
        out_path=outdir / "counts_by_dataset.png",
        xlabel="dataset",
        ylabel="count",
    )
    plot_bar_from_series(
        metadata["species"].value_counts().sort_index(),
        title="Image Count by Species",
        out_path=outdir / "counts_by_species.png",
        xlabel="species",
        ylabel="count",
    )
    plot_bar_from_series(
        metadata["split"].value_counts().sort_index(),
        title="Image Count by Split",
        out_path=outdir / "counts_by_split.png",
        xlabel="split",
        ylabel="count",
    )


def sample_metadata(metadata: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(metadata) <= max_rows:
        return metadata.copy()
    return metadata.sample(n=max_rows, random_state=seed).copy()


def image_size_eda(metadata: pd.DataFrame, root: Path, outdir: Path, size_sample: int, seed: int) -> pd.DataFrame:
    ensure_dir(outdir)
    meta = sample_metadata(metadata, size_sample, seed)

    rows = []
    errors = 0
    for row in meta.itertuples(index=False):
        img_path = root / row.path
        try:
            with Image.open(img_path) as img:
                width, height = img.size
            rows.append(
                {
                    "image_id": int(row.image_id),
                    "dataset": row.dataset,
                    "species": row.species,
                    "split": row.split,
                    "width": int(width),
                    "height": int(height),
                    "aspect_ratio": float(width / max(height, 1)),
                    "area": int(width * height),
                }
            )
        except Exception:
            errors += 1

    size_df = pd.DataFrame(rows)
    save_table(size_df, outdir / "image_size_stats.csv")
    if errors:
        print(f"[WARN] Failed to read {errors} images.")

    if size_df.empty:
        print("[WARN] No image size stats generated.")
        return size_df

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(size_df["width"], bins=40)
    axes[0].set_title("Width Distribution")
    axes[0].set_xlabel("width")

    axes[1].hist(size_df["height"], bins=40)
    axes[1].set_title("Height Distribution")
    axes[1].set_xlabel("height")

    axes[2].hist(size_df["aspect_ratio"], bins=40)
    axes[2].set_title("Aspect Ratio Distribution")
    axes[2].set_xlabel("width / height")

    fig.tight_layout()
    fig.savefig(outdir / "image_size_histograms.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / 'image_size_histograms.png'}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for dataset_name, ddf in size_df.groupby("dataset"):
        ax.scatter(ddf["width"], ddf["height"], s=6, alpha=0.35, label=dataset_name)
    ax.set_title("Width vs Height by Dataset")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "image_size_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {outdir / 'image_size_scatter.png'}")

    return size_df


def select_model_for_dataset(name: str):
    import timm
    from transformers import AutoModel

    if name in ["SalamanderID2025", "SeaTurtleID2022"]:
        return timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval(), 384, "MegaDescriptor-L-384"

    if name in ["LynxID2025", "TexasHornedLizards"]:
        try:
            model = AutoModel.from_pretrained(
                "conservationxlabs/miewid-msv3",
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            ).eval()
            return model, 512, "miewid-msv3"
        except AttributeError as exc:
            if "all_tied_weights_keys" not in str(exc):
                raise
            print("[WARN] MiewID incompatible with current transformers; fallback to MegaDescriptor-L-384")
            return timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval(), 384, "MegaDescriptor-L-384 (fallback)"

    raise ValueError(f"Unsupported dataset name: {name}")


def cluster_with_optional_hdbscan(features_l2: np.ndarray, min_cluster_size: int, min_samples: int) -> tuple[np.ndarray, str]:
    try:
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(features_l2)
        return labels.astype(int), "hdbscan"
    except Exception:
        labels = DBSCAN(eps=0.3, min_samples=max(min_samples, 2), metric="euclidean").fit_predict(features_l2)
        return labels.astype(int), "dbscan_fallback"


def run_umap_diagnostics(
    root: Path,
    outdir: Path,
    device: str,
    batch_size: int,
    split: str,
    max_per_dataset: int,
    umap_neighbors: int,
    umap_min_dist: float,
    hdbscan_min_cluster_size: int,
    hdbscan_min_samples: int,
    seed: int,
) -> None:
    import torch
    import torchvision.transforms as T
    import umap
    from wildlife_datasets.datasets import AnimalCLEF2026
    from wildlife_tools.features import DeepFeatures

    ensure_dir(outdir)

    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU.")
        device = "cpu"

    dataset_full = AnimalCLEF2026(
        str(root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    metadata = dataset_full.metadata
    metadata = metadata[metadata["split"] == split].copy()
    if metadata.empty:
        print(f"[WARN] No rows found for split={split}; skip UMAP diagnostics.")
        return

    all_summaries = []

    for name in sorted(metadata["dataset"].unique()):
        sub_meta = metadata[metadata["dataset"] == name].copy()
        if max_per_dataset > 0 and len(sub_meta) > max_per_dataset:
            sub_meta = sub_meta.sample(n=max_per_dataset, random_state=seed)

        dataset = dataset_full.get_subset(dataset_full.metadata["image_id"].isin(sub_meta["image_id"]))
        model, img_size, model_name = select_model_for_dataset(name)

        transform = T.Compose(
            [
                T.Resize(size=(img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        dataset.set_transform(transform)

        extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)
        features = extractor(dataset).features
        feats = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        feats_l2 = feats / norms

        reducer = umap.UMAP(
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=seed,
        )
        umap_xy = reducer.fit_transform(feats_l2)

        labels, algo = cluster_with_optional_hdbscan(
            feats_l2,
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
        )

        n_clusters = len(set(labels) - {-1})
        noise_ratio = float(np.mean(labels == -1))
        sil = np.nan
        if n_clusters >= 2:
            try:
                sil = float(silhouette_score(feats_l2, labels, metric="euclidean"))
            except Exception:
                sil = np.nan

        summary = {
            "dataset": name,
            "model": model_name,
            "samples": int(len(feats_l2)),
            "cluster_algo": algo,
            "n_clusters_excluding_noise": int(n_clusters),
            "noise_ratio": noise_ratio,
            "silhouette": sil,
        }
        all_summaries.append(summary)
        print(summary)

        cluster_df = pd.DataFrame(
            {
                "image_id": dataset.metadata["image_id"].astype(int).values,
                "umap_x": umap_xy[:, 0],
                "umap_y": umap_xy[:, 1],
                "cluster": labels,
                "dataset": name,
            }
        )
        save_table(cluster_df, outdir / f"umap_{name}.csv")

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_df = cluster_df.copy()
        noise_df = plot_df[plot_df["cluster"] == -1]
        non_noise_df = plot_df[plot_df["cluster"] != -1]

        if not non_noise_df.empty:
            sc = ax.scatter(
                non_noise_df["umap_x"],
                non_noise_df["umap_y"],
                c=non_noise_df["cluster"],
                s=12,
                alpha=0.8,
                cmap="tab20",
            )
            fig.colorbar(sc, ax=ax, label="cluster")

        if not noise_df.empty:
            ax.scatter(noise_df["umap_x"], noise_df["umap_y"], s=10, alpha=0.5, c="lightgray", label="noise")
            ax.legend()

        ax.set_title(f"UMAP + {algo} clusters ({name})")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(outdir / f"umap_{name}.png", dpi=170)
        plt.close(fig)
        print(f"Saved: {outdir / f'umap_{name}.png'}")

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        save_table(summary_df, outdir / "umap_cluster_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnimalCLEF 2026 EDA script")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root directory")
    parser.add_argument("--outdir", type=Path, default=Path("reports/eda"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument(
        "--size-sample",
        type=int,
        default=0,
        help="How many rows to sample for image-size EDA (0 means all)",
    )

    parser.add_argument("--run-umap", action="store_true", help="Run embedding extraction + UMAP diagnostics")
    parser.add_argument("--umap-split", type=str, default="test", help="Split for UMAP diagnostics")
    parser.add_argument(
        "--umap-max-per-dataset",
        type=int,
        default=600,
        help="Max samples per dataset for UMAP diagnostics (0 means all)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="Feature extraction batch size")
    parser.add_argument("--umap-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=10, help="HDBSCAN min_cluster_size")
    parser.add_argument("--hdbscan-min-samples", type=int, default=2, help="HDBSCAN min_samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    outdir = args.outdir
    ensure_dir(outdir)

    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {metadata_path}, rows={len(metadata)}")

    distribution_eda(metadata, outdir / "distribution")
    image_size_eda(metadata, root, outdir / "image_size", size_sample=args.size_sample, seed=args.seed)

    if args.run_umap:
        run_umap_diagnostics(
            root=root,
            outdir=outdir / "umap",
            device=args.device,
            batch_size=args.batch_size,
            split=args.umap_split,
            max_per_dataset=args.umap_max_per_dataset,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            seed=args.seed,
        )

    print("EDA complete.")


if __name__ == "__main__":
    main()
