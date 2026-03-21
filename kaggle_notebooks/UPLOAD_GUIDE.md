# 快速上傳 Kaggle Notebook 指南

## 步驟 1: 準備檔案
- 確保你已經有 `AnimalCLEF2026_L384_Baseline.ipynb` 檔案

## 步驟 2: 選擇上傳方式

### 方式 A: 直接複製到 Kaggle Notebook（推薦）
1. 登入 [Kaggle.com](https://kaggle.com)
2. 點擊 **Create → New Notebook**
3. 選擇 **Python** 環境
4. 點擊 **+ Add Input** 並新增 AnimalCLEF 2026 dataset:
   - 搜尋 "AnimalCLEF-2026-Animal-Re-Identification" 
   - 選擇官方競賽數據集
5. 點擊 **Import notebook**，選擇此 `.ipynb` 檔案上傳

### 方式 B: 從 GitHub/Repository 上傳
1. 如果你已上傳此 repo 到 GitHub:
   - Kaggle → Import Notebook → 貼上 GitHub URL
2. 或在 Kaggle 上建立新 notebook，複製代碼

## 步驟 3: 設定與執行

### 檢查數據集
- 確保右側 **Data** 面板中有 AnimalCLEF 2026 dataset
- 檢查路徑是否正確：`/kaggle/input/animalclef-2026-animal-re-identification`

### 修改配置（可選）
如果需要調整性能，編輯第 2 個 cell 中的 CONFIG：
```python
CONFIG = {
    'batch_size': 4,  # 改小以節省 VRAM（改為 2 或 1）
    # ... 其他設定
}
```

### 執行 Notebook
- 點擊 **▶ Run All** 或逐帳執行（**Ctrl + Enter**）
- 等待特徵提取和聚類完成（約 3-5 分鐘）

## 步驟 4: 提交結果

### 輸出文件位置
- 位於 `/kaggle/working/submission.csv`

### 提交方式
1. 點擊 **Commit & Run** 或 **Save Version**
2. 在筆記本右側點擊 **Publish** 發布（可選）
3. 下載 `submission.csv`：
   - 點擊右側 **Output** 面板
   - 選擇 `submission.csv` 並下載
4. 前往競賽頁面，點擊 **Make Submission**
5. 上傳 CSV 檔案並提交

## 檔案結構

```
kaggle_notebooks/
├── AnimalCLEF2026_L384_Baseline.ipynb  ← 主要 notebook
├── README.md                            ← 詳細使用說明
└── UPLOAD_GUIDE.md                      ← 本文件
```

## 注意事項

✓ **GPU 支持** — 自動使用 Kaggle 提供的 GPU  
✓ **自動兼容** — 所有依賴包在執行時自動安裝  
✓ **內存優化** — 預設 `batch_size=4` 適合大多數情況  
✓ **隨時調整** — 可修改 EPS 值以調整聚類敏感度  

## 常見問題

**Q: 執行時間太長？**
- A: 這是正常的，因為需要提取 ~50k 圖像的特徵。可以減少 batch_size 以加速（使用更多 GPU）

**Q: GPU 記憶體不足？**
- A: 將 `batch_size` 改為 `2` 或 `1`

**Q: 結果不好？**
- A: 嘗試修改 EPS_CONFIG 中的 eps 值：
  - 增大 eps → 更少、更大的聚類
  - 減小 eps → 更多、更小的聚類

**Q: 如何查看具體結果？**
- A: Notebook 最後會顯示聚類統計。下載 `submission.csv` 查看詳細結果。

---

**需要幫助？** 檢查 README.md 獲取更多詳情！
