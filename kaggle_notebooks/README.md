# AnimalCLEF 2026 - L-384 Kaggle Notebook

This folder contains a production-ready Jupyter notebook for the AnimalCLEF 2026 clustering task, specifically designed for Kaggle.

## 📋 Notebook Contents

**File:** `AnimalCLEF2026_L384_Baseline.ipynb`

This notebook implements a complete end-to-end clustering pipeline:

1. **Environment Setup** — Install and import required packages
2. **Configuration** — Set device, batch size, and DBSCAN eps values
3. **Data Loading** — Load AnimalCLEF2026 test dataset
4. **Model Setup** — Load MegaDescriptor-L-384 model
5. **Feature Extraction** — Extract 384-dimensional embeddings from test images
6. **Clustering** — Apply DBSCAN with optimized eps per dataset
7. **Submission Generation** — Create Kaggle submission CSV

## 🚀 How to Use on Kaggle

### Option 1: Create a New Kaggle Notebook

1. Go to [Kaggle.com](https://kaggle.com)
2. Navigate to **Create → Notebook**
3. Select **Python** as the language
4. Click **+ Add Input** and add the AnimalCLEF 2026 dataset:
   - Search for "AnimalCLEF 2026"
   - Add the official competition dataset
5. Copy and paste the notebook code cell by cell, or:
   - Click **File → Import notebook** and upload this `.ipynb` file

### Option 2: Use the Raw `.ipynb` File

1. Open this repository on Kaggle or GitHub
2. Copy the entire `.ipynb` file content
3. Paste it into a new Kaggle notebook

## ⚙️ Configuration

The notebook is pre-configured with:

```python
CONFIG = {
    'data_root': Path('/kaggle/input/animalclef-2026-animal-re-identification'),
    'output_dir': Path('/kaggle/working'),
    'device': 'cuda',  # Uses GPU if available
    'batch_size': 4,   # Adjust based on GPU memory
    'model_name': 'hf-hub:BVRA/MegaDescriptor-L-384',
}
```

### Adjust Batch Size

If you encounter GPU memory issues:
- Reduce `batch_size` to `2` or `1`
- The model will automatically fall back to CPU if GPU memory is exhausted

### Adjust DBSCAN Parameters

The `EPS_CONFIG` dictionary contains dataset-specific `eps` values:

```python
EPS_CONFIG = {
    'LynxID2025': 0.30,
    'SalamanderID2025': 0.20,
    'SeaTurtleID2022': 0.40,
    'TexasHornedLizards': 0.24,
}
```

These values are optimized from baseline experiments. You can modify them to tune clustering behavior.

## 📊 Output

After running the notebook, you'll get:

1. **submission.csv** — The main Kaggle submission file with columns:
   - `image_id` — Unique image identifier
   - `cluster` — Predicted cluster ID in format `cluster_{dataset}_{id}`

2. **Console output** — Detailed statistics:
   - Number of images per dataset
   - Number of clusters identified
   - Cluster size distribution

## 🔧 Model Details

**Model:** MegaDescriptor-L-384
- **Dimensions:** 384-D embeddings
- **Purpose:** Universal animal re-identification feature extraction
- **Similarity:** Cosine similarity
- **Clustering:** DBSCAN with precomputed distance matrix

### Why L-384?

- Balanced between expressiveness and computational efficiency
- Works well across diverse animal species
- Suitable for clustering-by-similarity approach
- Efficient on Kaggle GPU resources

## 📈 Performance Baseline

Expected performance metrics:
- **Total Clusters:** ~200-400 (depends on eps values)
- **Runtime:** ~3-5 minutes (GPU with 4GB+ VRAM)
- **Memory:** ~4-6GB GPU RAM

## 🛠️ Troubleshooting

### Package Installation Fails
- Some packages may already be installed on Kaggle
- The notebook gracefully handles this with `-q` (quiet) flag
- If specific packages fail, comment out that line and run manually

### GPU Out of Memory
- Reduce `batch_size` to `2` or `1`
- The model will automatically use CPU if needed

### Dataset Not Found
- Verify you've added the AnimalCLEF 2026 dataset as input
- Check the data path in CONFIG matches your dataset location
- Default path: `/kaggle/input/animalclef-2026-animal-re-identification`

## 📝 Notes

- This notebook uses only the **test split** for clustering predictions
- Each dataset gets its own clustering namespace (e.g., `cluster_LynxID2025_0`)
- Noise points from DBSCAN are converted to single-image clusters
- Results are sorted by `image_id` before submission

## 🔗 Related Files

Other files in this directory:
- `AnimalCLEF2026_L384_Baseline.ipynb` — This notebook
- `README.md` — This file

For local development, see the parent directory:
- `main.py` — CLI version of the baseline
- `eda.py` — Exploratory data analysis with UMAP diagnostics

---

**Last Updated:** 2026-03-21  
**Model:** MegaDescriptor-L-384  
**Framework:** PyTorch + wildlife-tools  
**Status:** Production Ready ✓
