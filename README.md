# AnimalCLEF 2026 Baseline (Local + GPU)

AnimalCLEF 2026 本地 baseline，目標是對測試圖片做個體分群並產生 Kaggle submission。

- Competition: https://www.kaggle.com/competitions/animal-clef-2026
- Task: individual re-identification clustering
- Output: `submission.csv` with `image_id,cluster`

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
