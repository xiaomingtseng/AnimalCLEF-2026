
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


BACKBONE_CONFIGS = {
    "megadescriptor-l": ("hf-hub:BVRA/MegaDescriptor-L-384", 384),
    "megadescriptor-s": ("hf-hub:BVRA/MegaDescriptor-S-224", 224),
    "dinov2-b": ("vit_base_patch14_reg4_dinov2", 336),
    "dinov2-l": ("vit_large_patch14_reg4_dinov2", 224),
    "dinov2-l336": ("vit_large_patch14_reg4_dinov2", 336),
}


def load_backbone(name: str, device: str):
    """Load a named backbone. Returns (model, input_size, display_name)."""
    import timm
    import torch

    if name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(BACKBONE_CONFIGS)}")

    model_id, size = BACKBONE_CONFIGS[name]

    if name.startswith("dinov2"):
        model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=size).eval()
    else:
        model = timm.create_model(model_id, pretrained=True, img_size=size).eval()

    return model, size, name


def build_submission(
    root: Path,
    output_path: Path,
    batch_size: int,
    device: str,
    eps_overrides: dict[str, float] | None = None,
    backbone: str = "mixed",
    finetuned_checkpoints: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run baseline pipeline and return the submission DataFrame."""
    import timm
    import torch
    import torchvision.transforms as T
    from transformers import AutoModel
    from wildlife_datasets.datasets import AnimalCLEF2026
    from wildlife_tools.features import DeepFeatures
    from wildlife_tools.similarity import CosineSimilarity

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
        """Load MiewID with fallback to MegaDescriptor on any error."""
        try:
            return AutoModel.from_pretrained(
                "conservationxlabs/miewid-msv3",
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            ).eval(), 512, "miewid-msv3"
        except Exception as e:
            print(f"[WARN] MiewID failed ({type(e).__name__}: {e}); falling back to MegaDescriptor-L-384")
            return timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval(), 384, "MegaDescriptor-L-384(fallback)"

    for name, dataset in datasets.items():
        if backbone == "mixed":
            if name in ["SalamanderID2025", "SeaTurtleID2022"]:
                model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
                size = 384
                model_name = "MegaDescriptor-L-384"
            elif name in ["LynxID2025", "TexasHornedLizards"]:
                model, size, model_name = load_miewid_with_fallback()
            else:
                raise ValueError(f"Unsupported dataset name: {name}")
        else:
            model, size, model_name = load_backbone(backbone, device)

        # Load fine-tuned checkpoint if provided for this dataset
        if finetuned_checkpoints and name in finetuned_checkpoints:
            ckpt_path = finetuned_checkpoints[name]
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model", ckpt)
            # Rebuild model at training size (224) to match checkpoint
            model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=False, img_size=224).eval()
            model.load_state_dict(state_dict, strict=True)
            size = 224
            model_name = f"MegaDescriptor-L+finetuned"
            print(f"  [fine-tuned] loaded {ckpt_path}")

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


def parse_kv_overrides(items: list[str] | None) -> dict[str, str]:
    """Parse KEY=VALUE pairs from a list of strings."""
    overrides: dict[str, str] = {}
    if not items:
        return overrides
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid value '{item}'. Use KEY=VALUE format.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
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
    parser.add_argument(
        "--backbone",
        type=str,
        default="mixed",
        help=(
            "Backbone for all datasets. 'mixed' uses MegaDescriptor+MiewID (default). "
            f"Other options: {list(BACKBONE_CONFIGS)}"
        ),
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        nargs="*",
        default=None,
        dest="finetuned_checkpoints",
        help="Load fine-tuned weights per dataset: DATASET=PATH",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    eps_raw = parse_kv_overrides(args.eps)
    eps_overrides = {k: float(v) for k, v in eps_raw.items()}
    finetuned_checkpoints = parse_kv_overrides(args.finetuned_checkpoints)

    build_submission(
        root=args.root,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        eps_overrides=eps_overrides,
        backbone=args.backbone,
        finetuned_checkpoints=finetuned_checkpoints or None,
    )


if __name__ == "__main__":
    main()

