"""Find optimal DBSCAN eps using train split ground truth.

Computes within-class vs between-class distance distributions,
then sweeps eps values and reports ARI + cluster count per dataset.

Usage:
  uv run python eps_grid_search.py --backbone mixed --n 300
  uv run python eps_grid_search.py --finetuned-dir models/finetune --n 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN

from baseline_config import similarity_to_distance
from main import load_backbone, BACKBONE_CONFIGS


def relabel_negatives(labels: np.ndarray) -> np.ndarray:
    labels = np.array(labels)
    neg_indices = np.where(labels == -1)[0]
    new_labels = np.arange(labels.max() + 1, labels.max() + 1 + len(neg_indices))
    labels[neg_indices] = new_labels
    return labels


def analyze(args):
    import timm
    import torch
    import torchvision.transforms as T
    from transformers import AutoModel
    from wildlife_datasets.datasets import AnimalCLEF2026
    from wildlife_tools.features import DeepFeatures
    from wildlife_tools.similarity import CosineSimilarity

    # Read identity labels directly from CSV (avoids factorize renaming issues)
    raw_meta = pd.read_csv(args.root / "metadata.csv")
    train_meta = raw_meta[(raw_meta["split"] == "train") & raw_meta["identity"].notna()]

    # factorize_label=True: identity → int codes so DataLoader can collate
    dataset_full = AnimalCLEF2026(
        str(args.root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    matcher = CosineSimilarity()
    results = []

    for ds_name in train_meta["dataset"].unique():
        ds_meta = train_meta[train_meta["dataset"] == ds_name]

        # Stratified sample: up to args.n images, keep identity balance
        # At least 2 images per identity where possible
        counts = ds_meta["identity"].value_counts()
        multi = counts[counts >= 2].index
        ds_multi = ds_meta[ds_meta["identity"].isin(multi)]

        n = min(args.n, len(ds_multi))
        if n < 20:
            print(f"[{ds_name}] too few multi-image identities ({len(ds_multi)}), skipping")
            continue

        per_id = max(2, int(n / len(multi)) + 1)
        parts = [
            g.sample(min(len(g), per_id), random_state=42)
            for _, g in ds_multi.groupby("identity")
        ]
        sampled = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=42)
            .head(n)
            .reset_index(drop=True)
        )

        # Build dataset subset
        ds = dataset_full.get_subset(
            dataset_full.metadata["image_id"].isin(sampled["image_id"])
        )

        # Align true_labels to the order ds.metadata returns images
        id_to_identity = dict(zip(sampled["image_id"], sampled["identity"]))
        ordered_identities = ds.metadata["image_id"].map(id_to_identity)
        true_labels = pd.Categorical(ordered_identities).codes

        # Load model — fine-tuned checkpoint takes priority
        ckpt_path = None
        if args.finetuned_dir:
            candidate = args.finetuned_dir / ds_name / "backbone_final.pth"
            if candidate.exists():
                ckpt_path = candidate

        if ckpt_path:
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=False, img_size=224).eval()
            ckpt = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(ckpt.get("model", ckpt), strict=True)
            size = 224
            model_name = f"MegaDescriptor-L+finetuned({ds_name})"
            print(f"  Loaded fine-tuned checkpoint: {ckpt_path}")
        elif args.backbone == "mixed":
            if ds_name in ["SalamanderID2025", "SeaTurtleID2022"]:
                model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
                size = 384
                model_name = "MegaDescriptor-L-384"
            else:
                model = AutoModel.from_pretrained(
                    "conservationxlabs/miewid-msv3",
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                ).eval()
                size = 512
                model_name = "MiewID-MSV3"
        else:
            model, size, model_name = load_backbone(args.backbone, args.device)

        transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        ds.set_transform(transform)

        extractor = DeepFeatures(model=model, device=args.device, batch_size=args.batch_size)
        features = extractor(ds)
        sim = matcher(features, features)
        dist = similarity_to_distance(sim)

        n_imgs = len(dist)
        idx = np.triu_indices(n_imgs, k=1)
        upper_dist = dist[idx]

        # Within-class vs between-class
        same_mask = true_labels[idx[0]] == true_labels[idx[1]]
        within = upper_dist[same_mask]
        between = upper_dist[~same_mask]

        print(f"\n{'='*60}")
        print(f"[{ds_name}] backbone={model_name} n={n_imgs} identities={len(np.unique(true_labels))}")
        print(f"  Within-class  dist: mean={within.mean():.4f}  p50={np.percentile(within,50):.4f}  p90={np.percentile(within,90):.4f}  p95={np.percentile(within,95):.4f}")
        print(f"  Between-class dist: mean={between.mean():.4f}  p5={np.percentile(between,5):.4f}   p10={np.percentile(between,10):.4f}  p25={np.percentile(between,25):.4f}")
        print(f"  → Natural gap: within p95={np.percentile(within,95):.4f}  between p5={np.percentile(between,5):.4f}")

        # Sweep eps
        eps_candidates = np.round(np.arange(0.05, 0.55, 0.02), 3)
        print(f"\n  {'eps':>6}  {'clusters':>8}  {'noise%':>7}  {'ARI':>7}")
        print(f"  {'-'*38}")
        best_ari, best_eps = -1, None
        for eps in eps_candidates:
            clustering = DBSCAN(eps=eps, metric="precomputed", min_samples=2)
            raw_labels = clustering.fit(dist).labels_
            noise_pct = (raw_labels == -1).mean() * 100
            pred = relabel_negatives(raw_labels.copy())
            n_clusters = len(np.unique(pred))
            ari = adjusted_rand_score(true_labels, pred)
            marker = " ←" if ari > best_ari else ""
            print(f"  {eps:>6.2f}  {n_clusters:>8}  {noise_pct:>6.1f}%  {ari:>7.4f}{marker}")
            if ari > best_ari:
                best_ari = ari
                best_eps = eps
        print(f"\n  Best eps={best_eps:.2f}  ARI={best_ari:.4f}")
        results.append({"dataset": ds_name, "best_eps": best_eps, "best_ari": best_ari})

    print(f"\n{'='*60}")
    print("SUMMARY — recommended eps:")
    for r in results:
        print(f"  {r['dataset']}: eps={r['best_eps']:.2f}  ARI={r['best_ari']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--backbone", default="mixed",
                        help=f"'mixed' or one of {list(BACKBONE_CONFIGS)}")
    parser.add_argument("--finetuned-dir", type=Path, default=None,
                        dest="finetuned_dir",
                        help="Directory with per-dataset fine-tuned checkpoints (overrides --backbone)")
    parser.add_argument("--n", type=int, default=300, help="Max samples per dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
