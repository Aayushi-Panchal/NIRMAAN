import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 1. SETTINGS
# =========================================================

DATASET_PATH = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("NIRMAAN BASELINE MODEL")
print("=" * 50)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================================================
# 3. TRANSFORMS
# =========================================================
# IMPORTANT:
# Baseline mein random augmentation nahi hai.

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. LOAD ORIGINAL DATASET
# =========================================================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

class_names = dataset.classes
targets = np.array(dataset.targets)

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nTotal images:", len(dataset))


# =========================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# =========================================================
# 70% Train
# 15% Validation
# 15% Test

indices = np.arange(len(dataset))

train_indices, temp_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=targets
)

temp_targets = targets[temp_indices]

val_indices, test_indices = train_test_split(
    temp_indices,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_targets
)

print("\nDataset Split:")
print("Train      :", len(train_indices))
print("Validation :", len(val_indices))
print("Test       :", len(test_indices))


# =========================================================
# 6. CREATE SUBSETS
# =========================================================

train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

test_dataset = Subset(
    dataset,
    test_indices
)


# =========================================================
# 7. DATA LOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# =========================================================
# 8. LOAD PRETRAINED RESNET-50
# =========================================================

print("\nLoading ResNet-50...")

weights = ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)


# Freeze pretrained layers
for param in model.parameters():
    param.requires_grad = False


# Replace final layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(class_names)
)

model = model.to(DEVICE)


print("ResNet-50 loaded successfully.")


# =========================================================
# 9. LOSS FUNCTION + OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 10. TRAINING
# =========================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


print("\n")
print("=" * 50)
print("STARTING BASELINE TRAINING")
print("=" * 50)


for epoch in range(NUM_EPOCHS):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = running_loss / total

    train_accuracy = correct / total


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    running_val_loss = 0.0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_val_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = running_val_loss / val_total

    val_accuracy = val_correct / val_total


    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # -----------------------------------------------------
    # SAVE BEST MODEL
    # -----------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )


# =========================================================
# 11. LOAD BEST MODEL
# =========================================================

model.load_state_dict(
    best_model_weights
)

print("\nBest Validation Accuracy:",
      f"{best_val_accuracy * 100:.2f}%")


# =========================================================
# 12. TEST SET PREDICTIONS
# =========================================================

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# =========================================================
# 13. FINAL TEST METRICS
# =========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


print("\n")
print("=" * 50)
print("BASELINE TEST RESULTS")
print("=" * 50)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title(
    "Nirmaan Baseline - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "baseline_confusion_matrix.png",
    dpi=300
)

plt.show()


# =========================================================
# 16. ACCURACY GRAPH
# =========================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs,
    np.array(train_accuracies) * 100,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    np.array(val_accuracies) * 100,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.title(
    "Nirmaan Baseline - Training vs Validation Accuracy"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "baseline_accuracy_graph.png",
    dpi=300
)

plt.show()


# =========================================================
# 17. LOSS GRAPH
# =========================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Nirmaan Baseline - Training vs Validation Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "baseline_loss_graph.png",
    dpi=300
)

plt.show()


# =========================================================
# 18. BASELINE METRICS BAR GRAPH
# =========================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_values = [
    accuracy * 100,
    precision * 100,
    recall * 100,
    f1 * 100
]

plt.figure(
    figsize=(9, 6)
)

bars = plt.bar(
    metric_names,
    metric_values
)

plt.ylim(0, 100)

plt.ylabel("Score (%)")

plt.xlabel("Metric")

plt.title(
    "Nirmaan Baseline Model Performance"
)

for bar, value in zip(
    bars,
    metric_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "baseline_metrics_graph.png",
    dpi=300
)

plt.show()


# =========================================================
# 19. SAVE RESULTS
# =========================================================

results = pd.DataFrame({

    "Model": [
        "ResNet-50 Baseline"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1_Score": [
        f1
    ]
})

results.to_csv(
    "baseline_results.csv",
    index=False
)


# =========================================================
# 20. SAVE TEXT REPORT
# =========================================================

with open(
    "baseline_results.txt",
    "w"
) as file:

    file.write(
        "NIRMAAN BASELINE MODEL RESULTS\n"
    )

    file.write(
        "================================\n\n"
    )

    file.write(
        "Model: ResNet-50\n"
    )

    file.write(
        f"Classes: {', '.join(class_names)}\n"
    )

    file.write(
        f"Total Images: {len(dataset)}\n"
    )

    file.write(
        f"Training Images: {len(train_dataset)}\n"
    )

    file.write(
        f"Validation Images: {len(val_dataset)}\n"
    )

    file.write(
        f"Test Images: {len(test_dataset)}\n\n"
    )

    file.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision: {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall: {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score: {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "=====================\n"
    )

    file.write(report)


# =========================================================
# 21. SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "nirmaan_baseline_resnet50.pth"
)


# =========================================================
# DONE
# =========================================================

print("\n")
print("=" * 50)
print("BASELINE EXPERIMENT COMPLETED ✅")
print("=" * 50)

print("\nGenerated files:")

print("1. baseline_accuracy_graph.png")
print("2. baseline_loss_graph.png")
print("3. baseline_confusion_matrix.png")
print("4. baseline_metrics_graph.png")
print("5. baseline_results.csv")
print("6. baseline_results.txt")
print("7. nirmaan_baseline_resnet50.pth")