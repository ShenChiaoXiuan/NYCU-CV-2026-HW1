"""
HW1 - Inference Script
Compatible with train.py (ResNet-101, BN->Dropout->Linear head)
"""

import os
import zipfile
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
# Model — must match train.py exactly
# ─────────────────────────────────────────────
class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        return (
            x.clamp(min=self.eps)
            .pow(self.p)
            .mean(dim=[-2, -1], keepdim=True)
            .pow(1.0 / self.p)
        )


def build_model(num_classes=100):
    weights = models.ResNet101_Weights.IMAGENET1K_V2
    model = models.resnet101(weights=None)
    model.avgpool = GeMPooling(p=3.0)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# TTA transforms
# ─────────────────────────────────────────────
def get_tta_transforms(image_size=448):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    base = int(image_size * 1.1)
    return [
        transforms.Compose(
            [
                transforms.Resize(base),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(base),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.3)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    ]


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
@torch.no_grad()
def predict(model, img_paths, transforms_list, device, batch_size=16):
    model.eval()
    all_preds = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="Predicting"):
        batch_paths = img_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        logits_sum = None
        for tf in transforms_list:
            batch = torch.stack([tf(img) for img in images]).to(device)
            logits = model(batch)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        preds = logits_sum.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
    return all_preds


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    DATA_DIR = "data"
    CKPT_PATH = "checkpoints_resnet101_v1/best_model.pth"
    OUTPUT_DIR = "submission"
    IMAGE_SIZE = 448
    BATCH_SIZE = 16
    NUM_CLASSES = 100

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Class mapping (from train set folder names) ──
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    print(f"[INFO] Classes: {len(idx_to_class)}")

    # ── Load model ───────────────────────────
    model = build_model(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model = model.to(device)
    print(f"[INFO] Loaded: {CKPT_PATH}")

    # ── Test images (sorted, flat directory) ─
    test_dir = Path(DATA_DIR) / "test"
    img_paths = sorted(
        p
        for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"[INFO] Test images: {len(img_paths)}")

    # ── Predict with TTA ─────────────────────
    tfs = get_tta_transforms(IMAGE_SIZE)
    preds = predict(model, img_paths, tfs, device, BATCH_SIZE)

    # ── Write prediction.csv ─────────────────
    csv_path = os.path.join(OUTPUT_DIR, "prediction.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for path, pred_idx in zip(img_paths, preds):
            writer.writerow([path.stem, idx_to_class[pred_idx]])
    print(f"[INFO] Saved: {csv_path}")

    # ── Zip for submission ───────────────────
    zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, "prediction.csv")
    print(f"[DONE] Upload {zip_path} to CodaBench → My Submissions")


if __name__ == "__main__":
    main()
