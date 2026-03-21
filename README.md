# AnimalCLEF 2026 Baseline (Local + GPU)

AnimalCLEF 2026 本地 baseline，目標是對測試圖片做個體分群並產生 Kaggle submission。

- Competition: https://www.kaggle.com/competitions/animal-clef-2026
- Task: individual re-identification clustering
- Output: `submission.csv` with `image_id,cluster`

## Known Issues & Limitations

### Embedding Quality Bottleneck 🚨

**Current Score**: 0.14602 (after eps grid search)

**Root Cause**: MegaDescriptor-L-384 embeddings are insufficiently discriminative for this task.

**Diagnosis Evidence** (from UMAP analysis on 300 test samples per dataset):

| Dataset | Model | Clusters | Noise % | Silhouette |
|---------|-------|----------|---------|------------|
| LynxID2025 | MegaDescriptor-L384 | 0 | 100% | N/A |
| SalamanderID2025 | MegaDescriptor-L384 | 0 | 100% | N/A |
| SeaTurtleID2022 | MegaDescriptor-L384 | 4 | 97.3% | -0.0017 |
| TexasHornedLizards | MegaDescriptor-L384 | 0 | 100% | N/A |

**Specific Issues**:

1. **Intra-identity Scatter**: Same animal appears vastly different across photos due to:
   - Viewing angle (head-on vs side vs dorsal view)
   - Lighting conditions (daylight vs underwater vs tank)
   - Image quality (sharp vs blurry)
   - Body position (curled vs stretched)

2. **SeaTurtleID2022** (Most problematic):
   - Head closeups vs full-body shots → completely different features
   - Underwater photos: unstable color/contrast
   - Quality variation: professional vs phone camera photos
   - Shell pattern obscured by water/algae

3. **Embedding 角度敏感性過高**: 
   - 同一隻動物的不同視角照片 → embedding 差異巨大
   - 例如：頭部特寫、全身照、側面、背部 → 都被視為不同特徵
   - MegaDescriptor 無法跨角度識別同一個體
   - 結果：DBSCAN 把同個體的不同照片當作不同個體

4. **DBSCAN Limitation**: 
   - Adjusting `eps` (±0.05) shows minimal effect on score
   - Problem is not clustering parameters but feature representation itself

**Recommended Next Steps** (Priority Order):

1. **Fine-tune embedding model** on training split with:
   - Contrastive loss (triplet or SimCLR)
   - Identity-aware training signal
   
2. **Try alternative architectures**:
   - Vision Transformer backbone (e.g., ViT-B/16)
   - Larger feature dimensions (512D vs 384D)

3. **Dataset-specific adaptations**:
   - SeaTurtles: separate head vs body pipeline
   - Color normalization for underwater photos

4. **Ensemble approach**: Combine multiple models

---

## Current Pipeline

`main.py` 目前流程：

1. 載入 `AnimalCLEF2026` metadata，篩選 test split。
2. 依資料集抽特徵：
   - `SalamanderID2025`, `SeaTurtleID2022` -> MegaDescriptor-L-384
   - `LynxID2025`, `TexasHornedLizards` -> MiewID（若相容性問題自動 fallback 到 MegaDescriptor）
3. 計算 cosine similarity。
4. 用 DBSCAN 分群。
5. 將 DBSCAN 的 `-1` 噪音樣本重標成單張群。
6. 輸出 `submission.csv`。

## Environment Setup

使用 `uv` 管理環境。

```bash
uv sync
```

專案已在 `pyproject.toml` 設定 CUDA 版 PyTorch index（cu124）。

快速檢查：

```bash
uv run python -c "import torch; print('torch', torch.__version__); print('cuda_compiled', torch.version.cuda); print('cuda_available', torch.cuda.is_available())"
```

## Run Baseline

GPU：

```bash
uv run python main.py --root data --output submission.csv --device cuda
```

4GB VRAM 建議：

```bash
uv run python main.py --root data --output submission.csv --device cuda --batch-size 2
```

CPU：

```bash
uv run python main.py --root data --output submission.csv --device cpu --batch-size 8
```

## Submit to Kaggle

```bash
kaggle competitions submit -c animal-clef-2026 -f submission.csv -m "Baseline v1: DBSCAN eps default, CUDA run on RTX 3050 laptop, local starter reproduction"
```

## EDA Results

[詳見 `working_notes.md` → Session 2]

資料集特徵分析：

- **SeaTurtleID2022**: 資料集中最大（>50% of images），圖像尺寸變異大（寬 100-500px），品質不穩定
- **LynxID2025**: 中等規模，野外拍攝，光照條件變化
- **SalamanderID2025**: 中等規模，多種背景環境
- **TexasHornedLizards**: 最小資料集，野外環境，環境複雜

**重要發現**: 
- 圖像尺寸變異大 → 需要強大的尺寸不變性
- 不同環境/光照 → embedding 需要對視角/光照魯棒
- 樣本稀疏（train set 也多為長尾分布） → 難以用密度聚類

---

## Useful Options

`main.py` 支援參數：

- `--root`: dataset root（預設 `data`）
- `--output`: submission 路徑（預設 `submission.csv`）
- `--device`: `cuda` 或 `cpu`
- `--batch-size`: 特徵抽取 batch size（預設 `32`）
- `--eps`: 覆寫 DBSCAN 參數，例如：

```bash
uv run python main.py --root data --output submission.csv --device cuda --batch-size 2 --eps LynxID2025=0.28 SalamanderID2025=0.20 SeaTurtleID2022=0.38 TexasHornedLizards=0.24
```

## Known Notes

- MiewID 與部分 `transformers` 版本有相容性問題，程式已做 fallback，不會直接中止。
- 若指定 `--device cuda` 但環境不支援 CUDA，程式會自動降級到 CPU。
- 4GB VRAM 容易在大 batch OOM，建議優先用 `--batch-size 2` 或 `4`。

## Latest Result

- Baseline v1 score: `0.14602`
- Reference score (starter mention): `0.19401`

## Experiment Log Template

每次跑完一版，建議記一筆（可貼到 `working_notes.md`）：

```text
Date:
Submission message:
Public score:

Config:
- device:
- batch_size:
- model policy: (miewid / fallback)
- eps:
   - LynxID2025=
   - SalamanderID2025=
   - SeaTurtleID2022=
   - TexasHornedLizards=

Observations:
-

Next action:
-
```

## Next Step: EDA

建議先完成這 4 件事再進一步調參：

1. 各 dataset / split 影像數量統計。
2. 影像寬高分布與異常值檢查。
3. train identity 分布（長尾程度）。
4. test 影像基礎品質檢查（壓縮、模糊、方向）。

## Project Files

- `main.py`: baseline pipeline
- `pyproject.toml`: dependencies and CUDA wheel source
- `data/metadata.csv`: competition metadata
- `data/sample_submission.csv`: submission format example
- `working_notes.md`: experiment notes
