# HW1: Image Classification with Deep Learning

> Visual Recognition — 2026 Spring
> Student ID: `314551095` | Name: `沈蕎萱`
> GitHub: `[https://github.com/ShenChiaoXiuan/NYCU-CV-2026-HW1]`

---

## Introduction

100-class fine-grained image classification trained on 21,024 images, evaluated on 2,344 test images via [CodaBench](https://www.codabench.org/).

Built on **ResNet-101** with the following modifications:

- **GeM Pooling** — replaces average pooling for better spatial feature aggregation
- **Custom Head** — `BatchNorm1d → Dropout(0.3) → Linear(100)`
- **Label Smoothing CE** (ε = 0.1) + **Mixup** (α = 0.2)
- **TrivialAugmentWide** + **RandomErasing**
- **Backbone Warm-up** — frozen for epochs 1–3, then full fine-tuning
- **AdamW + Cosine Annealing** over 60 epochs
- **TTA** — 3-view ensemble at inference
- **~44.5M parameters** (within 100M limit)

---

## Environment Setup

```bash
conda create -n hw1 python=3.10
conda activate hw1

# PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install tqdm numpy pillow
```

---

## Usage

**Dataset structure:**
```
data/
├── train/   # one subfolder per class
├── val/
└── test/    # flat directory, no subfolders
```

**Training:**
```bash
python train.py
# Saves best checkpoint to checkpoints_resnet101_v1/best_model.pth
```

**Inference & Submission:**
```bash
python predict.py
# Saves submission/submission.zip -> upload to CodaBench -> My Submissions
```

---

## Performance Snapshot

![Performance Snapshot](performance_snapshot.png)