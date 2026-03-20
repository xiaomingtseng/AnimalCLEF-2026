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

## Session 2 Plan: EDA (Next)

目标：先用 EDA 找到可解释的调参方向，再进行下一轮提交。

**Tasks**:
1. 统计每个 dataset 的 train/test 数量
2. train identity 分布（长尾、头部占比）
3. 图像尺寸分布（宽、高、长宽比）
4. 不同数据源的拍摄条件差异（水下/圈养/野外）
5. test set 基础质量检查（模糊、遮挡、方向）

---

## EDA Checklist

- [ ] 统计 train/test 样本数（按 dataset）
- [ ] 统计 train identity 数与每个 identity 样本数
- [ ] 检查图像格式和大小范围
- [ ] 可视化长尾分布（identity histogram）
- [ ] 抽样可视化异常图片

---

## 实验日志 (Experiment Log)

### Exp 1: Baseline v1 (Local Reproduction)
**Status**: ✅ Done

**Config**:
- device: cuda
- batch_size: default (32)
- clustering: DBSCAN with dataset-specific default eps
- model policy: MiewID for Lynx/Texas, fallback to MegaDescriptor if incompatible

**Result**:
- Public score: `0.14602`

**Message**:
- `Baseline v1: DBSCAN eps default, CUDA run on RTX 3050 laptop, local starter reproduction`

### Exp 2: EDA-driven tuning
**Status**: ⏳ Next

**Plan**:
1. 完成 EDA checklist
2. 基于 EDA 设定 eps 搜索范围
3. 生成 3-5 版 submission 比较

---

## 参考资源 (References)

- MegaDescriptor: https://huggingface.co/models?search=megadescriptor
- AnimalCLEF Task: https://www.kaggle.com/competitions/animal-clef-2026
- Data: 2.0GB compressed ZIP

---

**Last Updated**: 2026-03-21
