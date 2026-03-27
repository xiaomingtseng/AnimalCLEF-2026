# Identity Counts vs Cluster Counts

Generated: 2026-03-26

## Train Split: Ground Truth Identities

| Dataset | Identities | Train images | Mean imgs/id | Median | Min | Max | Singletons |
|---|---|---|---|---|---|---|---|
| SeaTurtleID2022 | 438 | 8,729 | 19.9 | 13 | 1 | 190 | 1 (0.2%) |
| LynxID2025 | 77 | 2,957 | 38.4 | 17 | 1 | 353 | 6 (7.8%) |
| SalamanderID2025 | 587 | 1,388 | 2.4 | 1 | 1 | 12 | 310 (52.8%) |
| TexasHornedLizards | N/A | 0 | — | — | — | — | — |

## Test Split: Image Counts

| Dataset | Test images |
|---|---|
| LynxID2025 | 946 |
| SalamanderID2025 | 689 |
| SeaTurtleID2022 | 500 |
| TexasHornedLizards | 274 |

## Cluster Counts vs Submissions

| Dataset | Train IDs | Test imgs | mixed_baseline (0.19402) | dinov2b_v1 (bad) | dinov2b_v2 |
|---|---|---|---|---|---|
| LynxID2025 | 77 | 946 | 670 | 5 | 132 |
| SalamanderID2025 | 587 | 689 | 633 | 1 | 29 |
| SeaTurtleID2022 | 438 | 500 | 322 | 2 | 137 |
| TexasHornedLizards | — | 274 | 260 | 1 | 14 |

## Diagnosis

**mixed_baseline**: Clusters ≈ test image count → mostly singletons (over-splitting).
  - Good: SeaTurtleID2022 (322 clusters / 438 train IDs, reasonable range)
  - Bad: LynxID2025 (670 clusters vs 77 train IDs — 8.7× over-split)

**dinov2b_v1** (eps=0.35–0.5): Everything merged into 1–5 clusters → eps too large.

**dinov2b_v2** (eps=0.10–0.17, from p5–p15 of pairwise distances): Still too few clusters.
  - SalamanderID2025: 29 clusters for 689 test images (should be ~500+)
  - TexasHornedLizards: 14 clusters for 274 images
  - Cause: DBSCAN chain-linking at p5–p15 eps still merges most points.

## DINOv2-B Distance Distribution (200 samples per dataset)

| Dataset | Dist range | Mean | Suggested eps (p5–p15) |
|---|---|---|---|
| LynxID2025 | 0.033–0.801 | 0.320 | 0.151–0.197 |
| SalamanderID2025 | 0.019–0.422 | 0.185 | 0.082–0.114 |
| SeaTurtleID2022 | 0.018–1.000 | 0.252 | 0.081–0.110 |
| TexasHornedLizards | 0.048–0.538 | 0.222 | 0.132–0.159 |

## Next Step

Calibrate eps on train split where ground truth is known.
Use `eps_grid_search.py` or within-class vs between-class distance analysis
to find the natural gap separating same-individual from different-individual pairs.
