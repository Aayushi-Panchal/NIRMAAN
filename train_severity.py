"""
train_severity.py
-----------------
Trains a Category-Conditioned Multi-Head ResNet-18 Severity Assessment Model.

Architecture:
  - Shared Backbone: ResNet-18 (ImageNet pretrained weights) -> 512-dim visual features
  - 4 Dedicated Category-Specific Classification Heads:
      * potholes_head     -> [Low, Medium, High]
      * waterlogging_head -> [Low, Medium, High]
      * streetlights_head -> [Low, Medium, High]
      * garbage_head      -> [Low, Medium, High]

Training Details:
  - Hardware: NVIDIA GeForce RTX 2050 (CUDA)
  - Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
  - Scheduler: CosineAnnealingLR (T_max=12, eta_min=1e-6)
  - Epochs: 12
  - Augmentations: RandomResizedCrop (224), RandomHorizontalFlip, ColorJitter
  - Metric Tracking: Macro-F1 and accuracy on validation set
  - Checkpoint: nirmaan_severity_best.pth

Evaluation:
  - Full evaluation on untouched unseen test set (157 images)
  - Saves classification report, per-category metrics, and confusion matrix to nirmaan_severity_metrics.json
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import numpy as np

MANIFEST_PATH = r"c:\Users\LENOVO\OneDrive\Documents\live project\severity_dataset_manifest.json"
CHECKPOINT_PATH = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_severity_best.pth"
METRICS_PATH = r"c:\Users\LENOVO\OneDrive\Documents\live project\nirmaan_severity_metrics.json"

CATEGORIES = ["potholes", "waterlogging", "streetlights", "garbage"]
SEVERITY_CLASSES = ["Low", "Medium", "High"]
BATCH_SIZE = 32
NUM_EPOCHS = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

class CategoryConditionedSeverityNet(nn.Module):
    def __init__(self, categories=CATEGORIES, num_classes=3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features  # 512
        # Use backbone up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        # Dedicated head for each civic problem category
        self.heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25),
                nn.Linear(128, num_classes)
            ) for cat in categories
        })

    def forward(self, x, categories):
        # x: (B, 3, 224, 224)
        # categories: list of category strings of length B
        features = self.backbone(x).flatten(1)  # (B, 512)
        logits = torch.zeros(x.size(0), 3, device=x.device)
        for i, cat in enumerate(categories):
            head = self.heads[cat] if cat in self.heads else self.heads["potholes"]
            logits[i] = head(features[i:i+1])
        return logits

    def forward_single(self, x, category):
        # Single image forward pass for inference
        features = self.backbone(x).flatten(1)
        head = self.heads[category] if category in self.heads else self.heads["potholes"]
        return head(features)


class NirmaanManifestDataset(Dataset):
    def __init__(self, manifest_path, split="train", transform=None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.records = [r for r in data["records"] if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = rec["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        category = rec["category"]
        severity_id = rec["severity_id"]  # 0=Low, 1=Medium, 2=High
        return img, category, severity_id

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    categories = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, categories, labels

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"  TRAINING CATEGORY-CONDITIONED SEVERITY MODEL (ResNet-18)", flush=True)
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"=======================================================", flush=True)

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = NirmaanManifestDataset(MANIFEST_PATH, split="train", transform=train_transforms)
    val_ds = NirmaanManifestDataset(MANIFEST_PATH, split="validation", transform=val_test_transforms)
    test_ds = NirmaanManifestDataset(MANIFEST_PATH, split="test", transform=val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    print(f"Dataset splits: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}", flush=True)

    model = CategoryConditionedSeverityNet(categories=CATEGORIES, num_classes=3).to(device)

    # Class weighting across train set
    train_labels = [r["severity_id"] for r in train_ds.records]
    class_counts = np.bincount(train_labels, minlength=3)
    weights = len(train_labels) / (3.0 * class_counts.astype(np.float32))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Train severity distribution: Low={class_counts[0]}, Med={class_counts[1]}, High={class_counts[2]}", flush=True)
    print(f"Computed loss weights: {weights.tolist()}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_f1 = -1.0
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, categories, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, categories)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, categories, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, categories)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_ds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "categories": CATEGORIES,
                "classes": SEVERITY_CLASSES,
                "class_to_idx": {"Low": 0, "Medium": 1, "High": 2},
                "best_val_acc": float(best_val_acc),
                "best_val_f1": float(best_val_f1),
                "epoch": epoch,
                "architecture": "CategoryConditioned-ResNet18"
            }, CHECKPOINT_PATH)
            star = " * BEST"
        else:
            star = ""

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% Macro-F1: {val_f1*100:.1f}%{star}", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s.", flush=True)
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% | Best Macro-F1: {best_val_f1*100:.2f}%", flush=True)
    print(f"Saved best model checkpoint to: {CHECKPOINT_PATH}", flush=True)

    # ==========================================
    # --- Final Evaluation on UNSEEN Test Set ---
    # ==========================================
    print(f"\n=======================================================", flush=True)
    print(f"  EVALUATING ON UNTOUCHED UNSEEN TEST SET (N={len(test_ds)})", flush=True)
    print(f"=======================================================", flush=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds = []
    test_targets = []
    test_categories = []

    with torch.no_grad():
        for images, categories, labels in test_loader:
            images = images.to(device)
            outputs = model(images, categories)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())
            test_categories.extend(categories)

    test_acc = accuracy_score(test_targets, test_preds)
    macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])
    report_dict = classification_report(test_targets, test_preds, target_names=SEVERITY_CLASSES, output_dict=True, zero_division=0)
    report_str = classification_report(test_targets, test_preds, target_names=SEVERITY_CLASSES, zero_division=0)

    print("\n--- TEST SET CLASSIFICATION REPORT ---", flush=True)
    print(report_str, flush=True)
    print("\n--- TEST CONFUSION MATRIX ---", flush=True)
    print(f"Classes: {SEVERITY_CLASSES}", flush=True)
    print(cm, flush=True)
    print(f"\nOverall Test Accuracy: {test_acc*100:.2f}%", flush=True)
    print(f"Macro F1-Score:        {macro_f1*100:.2f}%", flush=True)

    # Per-category evaluation
    cat_metrics = {}
    print("\n--- PER-CATEGORY TEST PERFORMANCE ---", flush=True)
    for cat in CATEGORIES:
        cat_indices = [i for i, c in enumerate(test_categories) if c == cat]
        if cat_indices:
            cat_targets = [test_targets[i] for i in cat_indices]
            cat_preds = [test_preds[i] for i in cat_indices]
            cat_acc = accuracy_score(cat_targets, cat_preds)
            cat_f1 = f1_score(cat_targets, cat_preds, average="macro", zero_division=0)
            cat_metrics[cat] = {
                "support": len(cat_indices),
                "accuracy": float(round(cat_acc, 4)),
                "macro_f1": float(round(cat_f1, 4))
            }
            print(f"  {cat:14s}: Support={len(cat_indices):2d} | Accuracy={cat_acc*100:.1f}% | Macro-F1={cat_f1*100:.1f}%", flush=True)

    metrics_data = {
        "architecture": "CategoryConditioned-ResNet18",
        "backbone": "ResNet-18 (ImageNet pretrained)",
        "heads": [f"{c}_head" for c in CATEGORIES],
        "classes": SEVERITY_CLASSES,
        "class_to_idx": {"Low": 0, "Medium": 1, "High": 2},
        "test_accuracy": float(round(test_acc, 4)),
        "test_macro_f1": float(round(macro_f1, 4)),
        "test_weighted_f1": float(round(weighted_f1, 4)),
        "confusion_matrix": cm.tolist(),
        "per_category_metrics": cat_metrics,
        "classification_report": report_dict,
        "checkpoint_path": CHECKPOINT_PATH,
        "total_test_samples": len(test_targets)
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\nSaved evaluation metrics to: {METRICS_PATH}", flush=True)

if __name__ == "__main__":
    train_model()
