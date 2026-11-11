import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

ORIGINAL_DATASET = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

TRAIN_DIR = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_clean_balanced\train"

VAL_DIR = os.path.join(
    ORIGINAL_DATASET,
    "validation"
)

TEST_DIR = os.path.join(
    ORIGINAL_DATASET,
    "test"
)

OUTPUT_DIR = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_final_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

BATCH_SIZE = 8
EPOCHS = 10

# Same learning rate as baseline
LEARNING_RATE = 0.0001

# Small weight decay helps reduce overfitting
WEIGHT_DECAY = 1e-4

SEED = 42


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("NIRMAAN FINAL BALANCED MODEL")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. CHECK PATHS
# ============================================================

print("\nChecking dataset paths...")

paths = {
    "Training": TRAIN_DIR,
    "Validation": VAL_DIR,
    "Test": TEST_DIR
}

for name, path in paths.items():

    print(
        f"{name}: {path}"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n{name} folder not found:\n{path}"
        )

print(
    "\nAll dataset paths found successfully."
)


# ============================================================
# 6. TRANSFORMS
# ============================================================

# The training images have already been augmented.
# Therefore, no additional random augmentation is applied here.

train_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# Validation and test remain completely untouched.

val_test_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 7. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)


# ============================================================
# 8. CHECK CLASSES
# ============================================================

class_names = train_dataset.classes

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        f"{i}: {name}"
    )


# Make sure validation/test use same class structure

if val_dataset.classes != class_names:

    raise ValueError(
        "Validation classes do not match training classes."
    )

if test_dataset.classes != class_names:

    raise ValueError(
        "Test classes do not match training classes."
    )


# ============================================================
# 9. DATASET SIZES
# ============================================================

print("\nDataset sizes:")

print(
    "Training   :",
    len(train_dataset)
)

print(
    "Validation :",
    len(val_dataset)
)

print(
    "Test       :",
    len(test_dataset)
)


# ============================================================
# 10. TRAINING DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

for class_name in class_names:

    class_index = train_dataset.class_to_idx[
        class_name
    ]

    count = sum(
        label == class_index
        for _, label in train_dataset.samples
    )

    print(
        f"{class_name:15s}: {count}"
    )


# ============================================================
# 11. DATA LOADERS
# ============================================================

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


# ============================================================
# 12. RESNET-50
# ============================================================

print("\nLoading ResNet-50...")

weights = models.ResNet50_Weights.DEFAULT

model = models.resnet50(
    weights=weights
)


# ============================================================
# 13. MODIFY FINAL CLASSIFIER
# ============================================================

num_classes = len(class_names)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

print(
    "ResNet-50 loaded successfully."
)


# ============================================================
# 14. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 15. TRAINING HISTORY
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


# ============================================================
# 16. TRAINING
# ============================================================

print("\n")
print("=" * 70)
print("STARTING FINAL MODEL TRAINING")
print("=" * 70)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item()
                * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss
        / val_total
    )

    val_accuracy = (
        val_correct
        / val_total
    ) * 100


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


# ============================================================
# 17. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)

print(
    "\nBest Validation Accuracy:",
    f"{best_val_accuracy:.2f}%"
)


# ============================================================
# 18. TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


model.eval()

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# 19. OVERALL METRICS
# ============================================================

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

macro_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


print("\nFINAL RESULTS")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"Macro F1  : {macro_f1 * 100:.2f}%"
)


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)

print(
    "\nClassification Report:\n"
)

print(report)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={
        "size": 13
    }
)

plt.title(
    "Nirmaan Final Model — Confusion Matrix",
    fontsize=16,
    pad=18
)

plt.xlabel(
    "Predicted Class",
    fontsize=12,
    labelpad=10
)

plt.ylabel(
    "Actual Class",
    fontsize=12,
    labelpad=10
)

plt.xticks(
    rotation=25,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout(
    pad=2
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. ACCURACY GRAPH
# ============================================================

epochs = list(
    range(
        1,
        EPOCHS + 1
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="o",
    linewidth=2,
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy",
    fontsize=17,
    pad=20
)

plt.xlabel(
    "Epoch",
    fontsize=13
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=13
)

plt.xticks(
    epochs
)

plt.legend(
    fontsize=11,
    loc="lower right"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout(
    pad=2
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_accuracy_graph.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    linewidth=2,
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss",
    fontsize=17,
    pad=20
)

plt.xlabel(
    "Epoch",
    fontsize=13
)

plt.ylabel(
    "Loss",
    fontsize=13
)

plt.xticks(
    epochs
)

plt.legend(
    fontsize=11,
    loc="upper right"
)

plt.grid(
    True,
    alpha=0.25
)

plt.tight_layout(
    pad=2
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_loss_graph.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. OVERALL METRICS GRAPH
# ============================================================

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
    figsize=(10, 6)
)

bars = plt.bar(
    metric_names,
    metric_values,
    width=0.6
)

plt.title(
    "Final Model Performance Metrics",
    fontsize=17,
    pad=20
)

plt.xlabel(
    "Evaluation Metric",
    fontsize=13
)

plt.ylabel(
    "Score (%)",
    fontsize=13
)

plt.ylim(
    0,
    105
)

for bar, value in zip(
    bars,
    metric_values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 1,

        f"{value:.2f}%",

        ha="center",
        va="bottom",
        fontsize=11
    )

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout(
    pad=2
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_metrics_graph.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 25. CLASS-WISE F1 GRAPH
# ============================================================

report_dict = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

class_f1 = [
    report_dict[name]["f1-score"] * 100
    for name in class_names
]

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    class_names,
    class_f1,
    width=0.6
)

plt.title(
    "Class-wise F1 Score",
    fontsize=17,
    pad=20
)

plt.xlabel(
    "Class",
    fontsize=13
)

plt.ylabel(
    "F1 Score (%)",
    fontsize=13
)

plt.ylim(
    0,
    105
)

for bar, value in zip(
    bars,
    class_f1
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        value + 1,

        f"{value:.2f}%",

        ha="center",
        va="bottom",
        fontsize=11
    )

plt.xticks(
    rotation=10
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout(
    pad=2
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "final_classwise_f1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 26. SAVE TRAINING HISTORY
# ============================================================

history = pd.DataFrame({

    "Epoch": epochs,

    "Train_Loss": train_losses,

    "Validation_Loss": val_losses,

    "Train_Accuracy": train_accuracies,

    "Validation_Accuracy": val_accuracies

})

history.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_training_history.csv"
    ),
    index=False
)


# ============================================================
# 27. SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Macro Precision",
        "Macro Recall",
        "Macro F1"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        macro_f1
    ]

})

results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_results.csv"
    ),
    index=False
)


# ============================================================
# 28. SAVE MODEL
# ============================================================

torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            class_names,

        "best_validation_accuracy":
            best_val_accuracy,

        "test_accuracy":
            accuracy,

        "test_precision":
            precision,

        "test_recall":
            recall,

        "test_f1":
            f1

    },

    os.path.join(
        OUTPUT_DIR,
        "nirmaan_final_resnet50.pth"
    )
)


# ============================================================
# 29. SAVE TEXT REPORT
# ============================================================

with open(

    os.path.join(
        OUTPUT_DIR,
        "final_results.txt"
    ),

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "NIRMAAN FINAL BALANCED MODEL\n"
    )

    f.write(
        "=" * 65 + "\n\n"
    )

    f.write(
        f"Training Images: {len(train_dataset)}\n"
    )

    f.write(
        f"Validation Images: {len(val_dataset)}\n"
    )

    f.write(
        f"Test Images: {len(test_dataset)}\n\n"
    )

    f.write(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%\n\n"
    )

    f.write(
        f"Accuracy: {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Weighted Precision: "
        f"{precision * 100:.2f}%\n"
    )

    f.write(
        f"Weighted Recall: "
        f"{recall * 100:.2f}%\n"
    )

    f.write(
        f"Weighted F1: "
        f"{f1 * 100:.2f}%\n"
    )

    f.write(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%\n\n"
    )

    f.write(
        "Classification Report:\n\n"
    )

    f.write(
        report
    )


# ============================================================
# 30. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved at:")

print(
    OUTPUT_DIR
)

print("\nGenerated files:")

print("✓ final_accuracy_graph.png")
print("✓ final_loss_graph.png")
print("✓ final_metrics_graph.png")
print("✓ final_classwise_f1.png")
print("✓ final_confusion_matrix.png")
print("✓ final_training_history.csv")
print("✓ final_results.csv")
print("✓ final_results.txt")
print("✓ nirmaan_final_resnet50.pth")