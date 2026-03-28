"""
HW1 - Image Classification with ResNet-101 Optimized
1. 使用 ResNet-101 提升模型容量 (約 44.5M 參數，符合 < 100M 限制)
2. 將解析度調整為 448 (更平衡的資訊量與訓練效率)
3. 優化 Data Augmentation 強度
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────
def get_transforms(image_size: int = 448, augment: bool = True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose(
            [
                # 448 是高階視覺任務中非常強大的解析度
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),  # 自動化增強策略，通常比手動調有效
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.2),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    val_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_tf, val_tf


def build_dataloaders(
    data_dir: str, batch_size: int, image_size: int, num_workers: int = 8
):
    train_tf, val_tf = get_transforms(image_size, augment=True)
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_tf
    )
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset.class_to_idx


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x.clamp(min=self.eps)
            .pow(self.p)
            .mean(dim=[-2, -1], keepdim=True)
            .pow(1.0 / self.p)
        )


def build_model(
    num_classes: int, arch: str = "resnet101", pretrained: bool = True
) -> nn.Module:
    # 使用 ResNet-101 提供更深的特徵學習
    weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet101(weights=weights)

    # 替換為 GeM Pooling 增強特徵凝聚力
    model.avgpool = GeMPooling(p=3.0)

    in_features = model.fc.in_features

    # 去掉最後一層，加上三層
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),  # 加入 BN 穩定微調過程
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# Loss & Augmentation (Label Smoothing / Mixup)
# ─────────────────────────────────────────────
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(preds, dim=1)
        loss = -log_prob.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=1)
        return ((1 - self.smoothing) * loss + self.smoothing * smooth_loss).mean()


def mixup_data(x, y, alpha=0.2):  # 稍微調低 alpha 讓訓練更穩定
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────
# Training / Evaluation
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    total_loss, total = 0.0, 0
    for imgs, labels in tqdm(loader, desc="  Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            preds = model(imgs)
            loss = mixup_criterion(criterion, preds, y_a, y_b, lam)
        else:
            preds = model(imgs)
            loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total += labels.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in tqdm(loader, desc="  Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────
# Main Execution
# ─────────────────────────────────────────────
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建議參數
    data_dir = "data"
    save_dir = "checkpoints_resnet101_v1"
    os.makedirs(save_dir, exist_ok=True)

    # 建構 Data
    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_dir, batch_size=16, image_size=448
    )

    # 建構 Model
    model = build_model(num_classes=100).to(device)
    print(f"[INFO] Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.05
    )  # 使用較小的 LR 給深層模型
    scheduler = CosineAnnealingLR(optimizer, T_max=60)
    criterion = LabelSmoothingCrossEntropy()

    best_acc = 0.0
    for epoch in range(1, 61):
        # 前 3 epoch 凍結 Backbone 預熱分類器
        if epoch == 1:
            for name, p in model.named_parameters():
                if "fc" not in name:
                    p.requires_grad = False
        if epoch == 4:
            for p in model.parameters():
                p.requires_grad = True

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
