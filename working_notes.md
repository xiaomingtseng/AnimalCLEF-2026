# Working Notes - AnimalCLEF 2026

## Session 1: 2026-03-21

### Setup
- ✅ Installed Kaggle API
- ✅ Downloaded `animal-clef-2026.zip` (2.0GB)
- ✅ Setup uv virtual environment with dependencies
- ✅ Synced CUDA PyTorch (`torch 2.6.0+cu124`, `torchvision 0.21.0+cu124`)

### Baseline Status

- ✅ Local baseline pipeline implemented in `main.py`
- ✅ Submission generated and uploaded
- ✅ Baseline v1 public score: `0.14602`
- ⚠️ MiewID compatibility issue handled by fallback to MegaDescriptor

---

## Session 2: EDA & Embedding Analysis (2026-03-21)

### 完成工作
✅ 實作 EDA pipeline (`eda.py`)：
- 數據集分布統計（dataset/species/split）
- 圖像尺寸分析（寬、高、長寬比）
- UMAP 降維 + HDBSCAN/DBSCAN 聚類診斷
- 沉默分值 (silhouette score) 評估

✅ Eps grid search：
- LynxID2025: 0.30
- SalamanderID2025: 0.20
- SeaTurtleID2022: 0.40
- TexasHornedLizards: 0.24

✅ UMAP 聚類診斷結果（test split, 300 samples/dataset）

---

### 根本問題診斷 ⚠️

**症狀**: 調整 DBSCAN eps 效果有限，分數卡在 **0.14602**

**根本原因**: **Embedding 品質不足**

具體表現：

1. **同個體散布過開**
   - 同一隻動物的不同照片在 embedding 空間裡散得很開
   - 無法靠空間密度有效分群

2. **海龜數據最嚴重**
   - 頭部特寫 vs 全身照：特徵完全不同
   - 水下拍攝：顏色、光照不穩定
   - 畫質差異大：清晰照 vs 模糊照

3. **Embedding 角度敏感性過高**
   - 同一隻動物的照片因視角、光線、拍攝距離不同，embedding 特徵差異巨大
   - MegaDescriptor 無法識別「這是同一個個體的不同角度」
   - DBSCAN 因此把同個體的不同照片當作不同個體，無法聚合

4. **聚類診斷結果**
   - LynxID2025: 100% noise（0 clusters）
   - SalamanderID2025: 100% noise（0 clusters）
   - SeaTurtleID2022: 4 clusters, 97.3% noise, silhouette = -0.0017（幾乎隨機）
   - TexasHornedLizards: 100% noise（0 clusters）

**結論**: embedding 對視角/光線/距離太敏感，同個體的不同照片被 DBSCAN 視為不同群。調整聚類參數無法修正根本問題，必須改善 embedding 的「視角魯棒性」

---

### 改進方向 (Future)

1. **強化 Embedding 品質**（高優先級）
   - 嘗試不同預訓練模型（vision transformer？）
   - 微調特徵提取器（fine-tune on training split）
   - 引入 metric learning（triplet loss, contrastive learning）
   - 集成多個模型

2. **針對數據特性優化**
   - 海龜：分離「頭部照」和「全身照」進行獨立聚類再融合
   - 處理光照/顏色差異（色彩正規化、augmentation）

3. **後處理優化**
   - 訓練集身體標記匹配（SIFT/ORB）輔助測試集
   - 多模態融合（embedding + visual feature）

---

---

## Session 3: Repo Cleanup & Next Phase Planning (2026-03-26)

### 完成工作
✅ 建立 `CLAUDE.md`（專案架構與常用指令文件）

✅ 重構共用邏輯至 `baseline_config.py`：
- `OFFICIAL_EPS` dict（各 dataset 的 DBSCAN eps）
- `similarity_to_distance()`（cosine similarity → DBSCAN distance）
- `main.py` 與 `generate_submission_orientation_aware.py` 改為 import 此模組

✅ 移除 MiewID fallback（已確認與 transformers 4.x 相容）

✅ 刪除不再需要的檔案（test scripts、診斷工具、多餘 CSV）

### 下一步方向（按優先順序）
1. **換更強的預訓練模型**（零訓練成本）
   - DINOv2（ViT backbone，視角魯棒性優於 ResNet 系）
   - MegaDescriptor-L-448（更大 input size）
2. **Fine-tune embedding**（最高優先，高效益）
   - 用 train split（有 identity label）做 metric learning
   - Triplet loss 或 ArcFace
   - `wildlife-tools` 有內建 `EmbeddingTrainer` 可直接使用
3. **SeaTurtle 專項優化**（佔 >50% 資料，改善影響最大）
   - 利用 `orientation` 欄位分離 head closeup vs full-body

---

## 實驗日誌 (Experiment Log)

### Exp 1: Official Baseline 復現（MegaDescriptor + MiewID）
**Status**: ✅ Done（確認與官方 starter notebook 同一條線，非真正實驗）

**Config**:
- Models（兩個模型）:
  - `MegaDescriptor-L-384`：SalamanderID2025、SeaTurtleID2022
  - `MiewID-MSV3`：LynxID2025、TexasHornedLizards
- ⚠️ **transformers 需降版**：MiewID 與最新版 transformers 不相容，需 pin 到舊版才能正常載入
- Clustering: DBSCAN with dataset-specific eps
- Device: CUDA (RTX 3050 Laptop, 4GB VRAM)
- Batch size: 2-4
- Eps: LynxID2025=0.30, SalamanderID2025=0.20, SeaTurtleID2022=0.40, TexasHornedLizards=0.24

**Results**:
- Public score: `0.19401`（= 官方 baseline，無改善）

**Notes**:
- 這個分數只是確認本地環境能正確復現官方結果，不算實驗進展
- Eps grid search 調整空間有限（±0.05 幾乎無效）
- UMAP 診斷：全部 dataset 幾乎 100% noise，根本問題是 embedding 品質而非聚類參數

---

> ⬆ 以上為 baseline 確認階段，以下才是真正的實驗

---

### Exp 2: Embedding 改善（進行中）
**Status**: ⏳ Planning

**Candidates**:
1. DINOv2 / ViT backbone（zero-shot，直接替換 MegaDescriptor）
2. Fine-tune on training split with triplet loss / ArcFace
3. Ensemble multiple models

---

## 参考资源 (References)

- MegaDescriptor: https://huggingface.co/models?search=megadescriptor
- AnimalCLEF Task: https://www.kaggle.com/competitions/animal-clef-2026
- Data: 2.0GB compressed ZIP

---

**Last Updated**: 2026-03-26
