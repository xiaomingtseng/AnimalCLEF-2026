
"""AnimalCLEF 2026 clustering baseline.

This script mirrors the official starter notebook as a CLI pipeline:
1) Load test images from AnimalCLEF2026 metadata
2) Extract deep features per dataset
3) Compute pairwise cosine similarity
4) Cluster with DBSCAN
5) Save Kaggle submission CSV
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from baseline_config import OFFICIAL_EPS, similarity_to_distance


def relabel_negatives(labels: np.ndarray) -> np.ndarray:
    """Turn DBSCAN noise points (-1) into single-image clusters."""
    labels = np.array(labels)
    neg_indices = np.where(labels == -1)[0]
    new_labels = np.arange(labels.max() + 1, labels.max() + 1 + len(neg_indices))
    labels[neg_indices] = new_labels
    return labels


def run_dbscan(similarity: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Cluster a similarity matrix by converting it to a precomputed distance matrix."""
    distance = similarity_to_distance(similarity)

    clustering = DBSCAN(eps=eps, metric="precomputed", min_samples=min_samples)
    labels = clustering.fit(distance).labels_
    return relabel_negatives(labels)


def build_submission(
    root: Path,
    output_path: Path,
    batch_size: int,
    device: str,
    eps_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run baseline pipeline and return the submission DataFrame."""
    import timm
    import torchvision.transforms as T
    from transformers import AutoModel
    from wildlife_datasets.datasets import AnimalCLEF2026
    from wildlife_tools.features import DeepFeatures
    from wildlife_tools.similarity import CosineSimilarity
    import torch

    if device.lower().startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available in current PyTorch build; using CPU instead.")
        device = "cpu"

    dataset_full = AnimalCLEF2026(
        str(root),
        transform=None,
        load_label=True,
        factorize_label=True,
        check_files=False,
    )

    metadata = dataset_full.metadata
    dataset_full = dataset_full.get_subset(metadata["split"] == "test")

    datasets = {}
    for name in dataset_full.metadata["dataset"].unique():
        datasets[name] = dataset_full.get_subset(dataset_full.metadata["dataset"] == name)

    eps_opt = dict(OFFICIAL_EPS)
    if eps_overrides:
        eps_opt.update(eps_overrides)

    matcher = CosineSimilarity()
    results = []

    def load_miewid_with_fallback():
        """Load MiewID - now compatible with transformers 4.x."""
        return AutoModel.from_pretrained(
            "conservationxlabs/miewid-msv3",
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).eval(), 512, "miewid-msv3"

    for name, dataset in datasets.items():
        if name in ["SalamanderID2025", "SeaTurtleID2022"]:
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
            size = 384
            model_name = "MegaDescriptor-L-384"
        elif name in ["LynxID2025", "TexasHornedLizards"]:
            model, size, model_name = load_miewid_with_fallback()
        else:
            raise ValueError(f"Unsupported dataset name: {name}")

        extractor = DeepFeatures(model=model, device=device, batch_size=batch_size)
        transform = T.Compose(
            [
                T.Resize(size=(size, size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        dataset.set_transform(transform)
        features = extractor(dataset)
        similarity = matcher(features, features)

        eps = eps_opt.get(name, 0.35)
        clusters = run_dbscan(similarity, eps=eps)
        result = pd.DataFrame(
            {
                "image_id": dataset.metadata["image_id"].astype(int),
                "cluster": [f"cluster_{name}_{int(c)}" for c in clusters],
            }
        )
        results.append(result)
        print(
            f"[{name}] model={model_name} images={len(dataset)} "
            f"eps={eps} clusters={result['cluster'].nunique()}"
        )

    submission = pd.concat(results, axis=0, ignore_index=True).sort_values("image_id")
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")
    return submission


def parse_eps_overrides(items: list[str] | None) -> dict[str, float]:
    """Parse --eps values in the format DATASET=VALUE."""
    overrides: dict[str, float] = {}
    if not items:
        return overrides

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --eps value '{item}'. Use DATASET=VALUE format.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = float(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="AnimalCLEF2026 baseline submission generator")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Dataset root path")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Feature extraction batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--eps",
        type=str,
        nargs="*",
        default=None,
        help="Optional DBSCAN eps override: DATASET=VALUE",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    eps_overrides = parse_eps_overrides(args.eps)
    build_submission(
        root=args.root,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        eps_overrides=eps_overrides,
    )


if __name__ == "__main__":
    main()

