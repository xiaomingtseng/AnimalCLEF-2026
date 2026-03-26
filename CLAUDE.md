# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle competition ([AnimalCLEF 2026](https://www.kaggle.com/competitions/animal-clef-2026)): cluster test images by individual animal identity across 4 wildlife datasets. Output is `submission.csv` with `image_id,cluster` columns.

**Current best score**: 0.14602 (baseline). Reference target from starter notebook: 0.19401.

## Environment

Uses `uv` package manager with Python 3.12+. PyTorch is configured with CUDA 12.4 wheels in `pyproject.toml`.

```bash
uv sync                         # Install all dependencies
uv run python main.py ...       # Run any script
```

Verify GPU setup:
```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Running the Pipeline

```bash
# GPU (4GB VRAM — use batch-size 2)
uv run python main.py --root data --output submission.csv --device cuda --batch-size 2

# CPU
uv run python main.py --root data --output submission.csv --device cpu --batch-size 8

# Override DBSCAN eps per dataset
uv run python main.py --root data --device cuda --batch-size 2 \
  --eps LynxID2025=0.28 SalamanderID2025=0.20 SeaTurtleID2022=0.38 TexasHornedLizards=0.24
```

Submit to Kaggle:
```bash
kaggle competitions submit -c animal-clef-2026 -f submission.csv -m "Description"
```

## Architecture

### Core Pipeline (`main.py`)

1. Load `data/metadata.csv` via `wildlife_datasets.AnimalCLEF2026`, filter to `split == "test"`
2. Split test images by dataset (4 separate sub-pipelines)
3. For each dataset:
   - Extract features using a pretrained model (batch inference, images resized to model input size, ImageNet normalization)
   - Compute pairwise cosine similarity matrix
   - Convert to distance: `distance = (max_sim - similarity) / max_sim` (see `baseline_config.similarity_to_distance()`)
   - Run DBSCAN with dataset-specific `eps` from `baseline_config.OFFICIAL_EPS`
   - Relabel DBSCAN noise points (`-1`) as individual singleton clusters
4. Concatenate all results; cluster names are `cluster_{dataset}_{id}`

### Model Assignment

| Dataset | Model | Fallback |
|---|---|---|
| SalamanderID2025 | MegaDescriptor-L-384 | — |
| SeaTurtleID2022 | MegaDescriptor-L-384 | — |
| LynxID2025 | MiewID-MSV3 | MegaDescriptor-L-384 |
| TexasHornedLizards | MiewID-MSV3 | MegaDescriptor-L-384 |

MiewID has known compatibility issues with some `transformers` versions; `main.py` auto-falls back to MegaDescriptor without crashing.

### Default DBSCAN Eps (`baseline_config.OFFICIAL_EPS`)

```python
{"LynxID2025": 0.3, "SalamanderID2025": 0.2, "SeaTurtleID2022": 0.4, "TexasHornedLizards": 0.24}
```

### Secondary Scripts

- `eda.py` — distribution analysis, image size stats, UMAP diagnostics, silhouette scores
- `eps_grid_search.py` — sweep eps values per dataset, report cluster count and noise ratio
- `train_orientation_classifier.py` — train EfficientNet-B0 to classify image orientation
- `generate_submission_orientation_aware.py` — cluster SeaTurtleID2022 separately by orientation group
- `baseline_config.py` — shared `OFFICIAL_EPS` dict and `similarity_to_distance()` utility

### Data

- `data/metadata.csv` — `image_id, identity, path, date, orientation, species, split, dataset`
- `data/sample_submission.csv` — format reference
- `models/orientation_classifier.pth` — trained orientation classifier (EfficientNet-B0)
- `working_notes.md` — experiment log

## Known Issues & Current Diagnosis

**Root cause of low score**: Pretrained embeddings (MegaDescriptor-L-384) are viewpoint-sensitive. The same individual photographed from different angles produces very different embeddings, causing DBSCAN to treat them as separate identities.

UMAP analysis (300 test samples per dataset) showed 97–100% noise across all datasets — adjusting `eps` has minimal effect because the problem is feature representation, not clustering parameters.

**Priority next steps**:
1. Fine-tune embedding model on train split with contrastive/triplet loss
2. Try ViT backbone or larger embedding dimensions
3. Dataset-specific preprocessing (SeaTurtleID2022: separate head vs. full-body pipeline; underwater color normalization)
4. Ensemble multiple models
