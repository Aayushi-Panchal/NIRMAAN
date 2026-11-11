import os
import sys
import json
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------------------------------
# 1. CONFIGURATION & DEVICE
# -------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[Device] Using CPU")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "nirmaan_recognition_final")

BATCH_SIZE = 16
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# -------------------------------------------------------------
# 2. TRANSFORMS & DATA LOADERS
# -------------------------------------------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "validation"), transform=val_test_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=val_test_transforms)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"[Dataset] Classes ({num_classes}): {class_names}")
print(f"[Dataset] Train samples: {len(train_dataset)}")
print(f"[Dataset] Validation samples: {len(val_dataset)}")
print(f"[Dataset] Test samples: {len(test_dataset)}")

# Save class mapping
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
mapping_path = os.path.join(PROJECT_DIR, "nirmaan_class_mapping.json")
with open(mapping_path, "w") as f:
    json.dump({
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "classes": class_names
    }, f, indent=4)
print(f"[Mapping] Saved class mapping to {mapping_path}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

# -------------------------------------------------------------
# 3. CLASS WEIGHTS FOR BALANCED LEARNING
# -------------------------------------------------------------
train_targets = [label for _, label in train_dataset.samples]
class_counts = [train_targets.count(i) for i in range(num_classes)]
total_samples = len(train_targets)
# Balanced class weights: total_samples / (num_classes * class_count)
class_weights = [total_samples / (num_classes * c) for c in class_counts]
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print("[Balancing] Train counts per class:", dict(zip(class_names, class_counts)))
print("[Balancing] Computed class weights:", {c: round(w, 3) for c, w in zip(class_names, class_weights)})

# -------------------------------------------------------------
# 4. MODEL ARCHITECTURE (ResNet-50)
# -------------------------------------------------------------
print("\n[Model] Initializing ResNet-50...")
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Freeze early feature layers (conv1, bn1, layer1, layer2)
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name and "layer3" not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Replace FC head with dropout for regularization
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)
model = model.to(DEVICE)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"[Model] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# Optimizer with differential learning rate (backbone lower, head higher)
optimizer = optim.AdamW([
    {"params": model.layer3.parameters(), "lr": LEARNING_RATE * 0.2},
    {"params": model.layer4.parameters(), "lr": LEARNING_RATE * 0.5},
    {"params": model.fc.parameters(), "lr": LEARNING_RATE}
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# -------------------------------------------------------------
# 5. TRAINING LOOP
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STARTING TRAINING: NIRMAAN FIRST RECOGNITION MODEL")
print("=" * 60)

best_model_path = os.path.join(PROJECT_DIR, "nirmaan_recognition_best.pth")
best_val_f1 = 0.0
best_model_weights = copy.deepcopy(model.state_dict())

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_f1": []
}

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    scheduler.step()
    epoch_train_loss = running_loss / total
    epoch_train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = accuracy_score(val_targets, val_preds)
    epoch_val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

    history["train_loss"].append(epoch_train_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_loss"].append(epoch_val_loss)
    history["val_acc"].append(epoch_val_acc)
    history["val_f1"].append(epoch_val_f1)

    print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS:02d}] "
          f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
          f"Val Macro-F1: {epoch_val_f1*100:.2f}%")

    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(best_model_weights, best_model_path)
        print(f"  --> Best model saved (Val Macro-F1: {best_val_f1*100:.2f}%)")
    sys.stdout.flush()

# Save Best Model
best_model_path = os.path.join(PROJECT_DIR, "nirmaan_recognition_best.pth")
torch.save(best_model_weights, best_model_path)
print(f"\n[Checkpoint] Saved best model weights to: {best_model_path}")

# Load best weights for testing
model.load_state_dict(best_model_weights)

# -------------------------------------------------------------
# 6. EVALUATION ON UNTOUCHED TEST SET
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

model.eval()
test_preds = []
test_targets = []
test_probs = []
test_paths = [s[0] for s in test_dataset.samples]

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.numpy())
        test_probs.extend(probs.cpu().numpy())

test_targets = np.array(test_targets)
test_preds = np.array(test_preds)
test_probs = np.array(test_probs)

test_accuracy = accuracy_score(test_targets, test_preds)
test_precision = precision_score(test_targets, test_preds, average="weighted", zero_division=0)
test_recall = recall_score(test_targets, test_preds, average="weighted", zero_division=0)
test_f1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)
macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)

print(f"\nOverall Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Weighted Precision     : {test_precision * 100:.2f}%")
print(f"Weighted Recall        : {test_recall * 100:.2f}%")
print(f"Weighted F1-Score      : {test_f1 * 100:.2f}%")
print(f"Macro F1-Score         : {macro_f1 * 100:.2f}%")

print("\n" + "-" * 50)
print("CLASSIFICATION REPORT:")
print("-" * 50)
clf_report_str = classification_report(test_targets, test_preds, target_names=class_names, digits=4, zero_division=0)
print(clf_report_str)

clf_dict = classification_report(test_targets, test_preds, target_names=class_names, output_dict=True, zero_division=0)

# Save text report
report_path = os.path.join(PROJECT_DIR, "nirmaan_recognition_results.txt")
with open(report_path, "w") as f:
    f.write("NIRMAAN CIVIC PROBLEM RECOGNITION MODEL - TEST RESULTS\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Architecture : ResNet-50\n")
    f.write(f"Checkpoint   : nirmaan_recognition_best.pth\n")
    f.write(f"Classes      : {', '.join(class_names)}\n\n")
    f.write(f"Test Accuracy     : {test_accuracy * 100:.2f}%\n")
    f.write(f"Weighted Precision: {test_precision * 100:.2f}%\n")
    f.write(f"Weighted Recall   : {test_recall * 100:.2f}%\n")
    f.write(f"Weighted F1 Score : {test_f1 * 100:.2f}%\n")
    f.write(f"Macro F1 Score    : {macro_f1 * 100:.2f}%\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-" * 50 + "\n")
    f.write(clf_report_str + "\n\n")

# Save json metrics
metrics_path = os.path.join(PROJECT_DIR, "nirmaan_recognition_results.json")
with open(metrics_path, "w") as f:
    json.dump({
        "accuracy": test_accuracy,
        "weighted_precision": test_precision,
        "weighted_recall": test_recall,
        "weighted_f1": test_f1,
        "macro_f1": macro_f1,
        "classification_report": clf_dict
    }, f, indent=4)

# -------------------------------------------------------------
# 7. GENERATE IMPORTANT GRAPHS
# -------------------------------------------------------------
# 1. Confusion Matrix
cm = confusion_matrix(test_targets, test_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
plt.ylabel("Actual Label", fontsize=11, fontweight="bold")
plt.title(f"Nirmaan Recognition Model - Confusion Matrix\n(Accuracy: {test_accuracy*100:.2f}%)", fontsize=12, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(PROJECT_DIR, "nirmaan_recognition_confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"[Plot] Saved confusion matrix to {cm_path}")

# 2. Training and Validation Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs_range = range(1, NUM_EPOCHS + 1)

# Loss curve
ax1.plot(epochs_range, history["train_loss"], marker="o", label="Train Loss", color="#1f77b4")
ax1.plot(epochs_range, history["val_loss"], marker="s", label="Val Loss", color="#ff7f0e")
ax1.set_title("Training & Validation Loss", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

# Accuracy curve
ax2.plot(epochs_range, np.array(history["train_acc"]) * 100, marker="o", label="Train Acc", color="#2ca02c")
ax2.plot(epochs_range, np.array(history["val_acc"]) * 100, marker="s", label="Val Acc", color="#d62728")
ax2.plot(epochs_range, np.array(history["val_f1"]) * 100, marker="^", label="Val Macro-F1", color="#9467bd")
ax2.set_title("Training & Validation Accuracy / F1", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Percentage (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
curves_path = os.path.join(PROJECT_DIR, "nirmaan_recognition_training_curves.png")
plt.savefig(curves_path, dpi=300)
plt.close()
print(f"[Plot] Saved training curves to {curves_path}")

# -------------------------------------------------------------
# 8. ERROR ANALYSIS
# -------------------------------------------------------------
misclassified = []
for idx in range(len(test_targets)):
    if test_targets[idx] != test_preds[idx]:
        actual_class = class_names[test_targets[idx]]
        pred_class = class_names[test_preds[idx]]
        conf = float(test_probs[idx][test_preds[idx]])
        img_name = os.path.basename(test_paths[idx])
        misclassified.append({
            "image": img_name,
            "actual": actual_class,
            "predicted": pred_class,
            "confidence": conf,
            "path": test_paths[idx]
        })

error_report_path = os.path.join(PROJECT_DIR, "nirmaan_error_analysis.txt")
with open(error_report_path, "w") as f:
    f.write("NIRMAAN RECOGNITION MODEL - ERROR ANALYSIS REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Test Samples: {len(test_targets)}\n")
    f.write(f"Total Misclassified: {len(misclassified)} ({len(misclassified)/len(test_targets)*100:.2f}% error rate)\n\n")
    if misclassified:
        f.write("Misclassified Samples Details:\n")
        f.write("-" * 60 + "\n")
        for m in misclassified:
            f.write(f"Image: {m['image']} | Actual: {m['actual']} -> Predicted: {m['predicted']} (Confidence: {m['confidence']*100:.1f}%)\n")
    else:
        f.write("No misclassifications observed on the test set!\n")

print(f"\n[Analysis] Total misclassified on test set: {len(misclassified)} / {len(test_targets)}")
print(f"[Analysis] Saved error analysis to {error_report_path}")
print("\n" + "=" * 60)
print("TRAINING & EVALUATION COMPLETE [SUCCESS]")
print("=" * 60)
