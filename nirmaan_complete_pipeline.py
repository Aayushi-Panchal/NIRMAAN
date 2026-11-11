import os
import shutil
import random
import hashlib
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

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

SOURCE_DATASET = (
    r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"
)

PROJECT_DIR = (
    r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_clean_experiment"
)

CLEAN_DATASET = os.path.join(
    PROJECT_DIR,
    "clean_dataset"
)

BALANCED_DATASET = os.path.join(
    PROJECT_DIR,
    "balanced_dataset"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

REPORT_DIR = os.path.join(
    PROJECT_DIR,
    "reports"
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET_PER_CLASS = 310

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4

SEED = 42

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. CREATE FOLDERS
# ============================================================

os.makedirs(
    PROJECT_DIR,
    exist_ok=True
)

os.makedirs(
    CLEAN_DATASET,
    exist_ok=True
)

os.makedirs(
    BALANCED_DATASET,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("NIRMAAN COMPLETE CLEAN RESEARCH PIPELINE")
print("=" * 70)

print("\nDevice:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 6. CHECK ORIGINAL DATASET
# ============================================================

print("\nChecking original dataset...")

if not os.path.exists(SOURCE_DATASET):

    raise FileNotFoundError(
        f"Dataset not found:\n{SOURCE_DATASET}"
    )


ORIGINAL_SPLITS = [
    "train",
    "validation",
    "test"
]

for split in ORIGINAL_SPLITS:

    path = os.path.join(
        SOURCE_DATASET,
        split
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing folder:\n{path}"
        )


print(
    "Original dataset found successfully."
)


# ============================================================
# 7. COLLECT ALL IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1 — COLLECTING ORIGINAL IMAGES")
print("=" * 70)


all_images = {}

for original_split in ORIGINAL_SPLITS:

    split_path = os.path.join(
        SOURCE_DATASET,
        original_split
    )

    for class_name in os.listdir(
        split_path
    ):

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(
            class_path
        ):
            continue

        if class_name not in all_images:

            all_images[class_name] = []

        for root, dirs, files in os.walk(
            class_path
        ):

            for file in files:

                if file.lower().endswith(
                    VALID_EXTENSIONS
                ):

                    all_images[class_name].append(
                        os.path.join(
                            root,
                            file
                        )
                    )


print("\nOriginal image counts:")

total_original = 0

for class_name in sorted(
    all_images
):

    count = len(
        all_images[class_name]
    )

    total_original += count

    print(
        f"{class_name:15s}: {count}"
    )

print(
    f"\nTOTAL: {total_original}"
)


# ============================================================
# 8. EXACT DUPLICATE REMOVAL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 2 — REMOVING EXACT DUPLICATES")
print("=" * 70)


def file_hash(path):

    md5 = hashlib.md5()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(8192),
            b""
        ):

            md5.update(chunk)

    return md5.hexdigest()


clean_images = {}

exact_duplicate_count = 0

global_hashes = set()


for class_name in sorted(
    all_images
):

    clean_images[class_name] = []

    for path in all_images[class_name]:

        try:

            h = file_hash(path)

        except Exception:

            continue

        if h in global_hashes:

            exact_duplicate_count += 1

            continue

        global_hashes.add(h)

        clean_images[class_name].append(
            path
        )


print(
    "\nExact duplicate images removed:",
    exact_duplicate_count
)


print("\nAfter exact duplicate removal:")

for class_name in sorted(
    clean_images
):

    print(
        f"{class_name:15s}: "
        f"{len(clean_images[class_name])}"
    )


# ============================================================
# 9. CONSERVATIVE NEAR-DUPLICATE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("STEP 3 — REMOVING VERY CLOSE DUPLICATES")
print("=" * 70)

print(
    "Using a conservative perceptual hash threshold."
)


def perceptual_hash(
    path,
    size=16
):

    try:

        image = Image.open(
            path
        ).convert("L")

        image = image.resize(
            (size, size)
        )

        pixels = np.asarray(
            image
        )

        average = pixels.mean()

        return (
            pixels > average
        ).flatten()

    except Exception:

        return None


def hash_distance(
    hash1,
    hash2
):

    return np.count_nonzero(
        hash1 != hash2
    )


# We use threshold 2.
# This is deliberately conservative so that
# visually different road scenes are not removed.

NEAR_DUPLICATE_THRESHOLD = 2


near_duplicate_count = 0

final_images = {}


for class_name in sorted(
    clean_images
):

    final_images[class_name] = []

    accepted_hashes = []

    for path in clean_images[class_name]:

        current_hash = perceptual_hash(
            path
        )

        if current_hash is None:

            continue

        duplicate_found = False

        for old_hash in accepted_hashes:

            d = hash_distance(
                current_hash,
                old_hash
            )

            if d <= NEAR_DUPLICATE_THRESHOLD:

                duplicate_found = True

                break

        if duplicate_found:

            near_duplicate_count += 1

            continue

        accepted_hashes.append(
            current_hash
        )

        final_images[class_name].append(
            path
        )


print(
    "\nVery close duplicate images removed:",
    near_duplicate_count
)


print("\nClean image counts:")

total_clean = 0

for class_name in sorted(
    final_images
):

    count = len(
        final_images[class_name]
    )

    total_clean += count

    print(
        f"{class_name:15s}: {count}"
    )

print(
    f"\nTOTAL CLEAN IMAGES: {total_clean}"
)


# ============================================================
# 10. CLEAN STRATIFIED SPLIT
# ============================================================

print("\n")
print("=" * 70)
print("STEP 4 — CREATING CLEAN TRAIN / VAL / TEST SPLIT")
print("=" * 70)


for class_name in sorted(
    final_images
):

    images = final_images[class_name]

    random.shuffle(
        images
    )

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_images = images[
        :train_end
    ]

    val_images = images[
        train_end:val_end
    ]

    test_images = images[
        val_end:
    ]


    # --------------------------------------------------------
    # CREATE CLASS FOLDERS
    # --------------------------------------------------------

    for split_name in [
        "train",
        "validation",
        "test"
    ]:

        os.makedirs(
            os.path.join(
                CLEAN_DATASET,
                split_name,
                class_name
            ),
            exist_ok=True
        )


    # --------------------------------------------------------
    # COPY IMAGES
    # --------------------------------------------------------

    split_data = {

        "train": train_images,

        "validation": val_images,

        "test": test_images

    }


    for split_name, image_list in split_data.items():

        destination = os.path.join(
            CLEAN_DATASET,
            split_name,
            class_name
        )

        for index, source_path in enumerate(
            image_list
        ):

            extension = os.path.splitext(
                source_path
            )[1]

            destination_path = os.path.join(
                destination,
                f"{class_name}_{index:05d}{extension}"
            )

            shutil.copy2(
                source_path,
                destination_path
            )


    print(
        f"{class_name:15s} | "
        f"Train: {len(train_images):3d} | "
        f"Val: {len(val_images):3d} | "
        f"Test: {len(test_images):3d}"
    )


# ============================================================
# 11. BALANCE TRAINING DATA
# ============================================================

print("\n")
print("=" * 70)
print("STEP 5 — BALANCING TRAINING DATA")
print("=" * 70)


augmentation = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1)
    )

])


for class_name in sorted(
    final_images
):

    source_class = os.path.join(
        CLEAN_DATASET,
        "train",
        class_name
    )

    destination_class = os.path.join(
        BALANCED_DATASET,
        "train",
        class_name
    )

    os.makedirs(
        destination_class,
        exist_ok=True
    )


    source_files = [

        os.path.join(
            source_class,
            file
        )

        for file in os.listdir(
            source_class
        )

        if file.lower().endswith(
            VALID_EXTENSIONS
        )

    ]


    random.shuffle(
        source_files
    )


    original_count = len(
        source_files
    )


    # --------------------------------------------------------
    # IF MORE THAN TARGET
    # --------------------------------------------------------

    if original_count > TARGET_PER_CLASS:

        selected_files = random.sample(
            source_files,
            TARGET_PER_CLASS
        )

    else:

        selected_files = source_files


    # --------------------------------------------------------
    # COPY ORIGINALS
    # --------------------------------------------------------

    for index, source_path in enumerate(
        selected_files
    ):

        extension = os.path.splitext(
            source_path
        )[1]

        destination_path = os.path.join(
            destination_class,
            f"original_{index:04d}{extension}"
        )

        shutil.copy2(
            source_path,
            destination_path
        )


    current_count = len(
        selected_files
    )

    generated = 0


    # --------------------------------------------------------
    # AUGMENT UNTIL TARGET
    # --------------------------------------------------------

    while current_count < TARGET_PER_CLASS:

        source_path = random.choice(
            selected_files
        )

        image = Image.open(
            source_path
        ).convert("RGB")

        augmented_image = augmentation(
            image
        )

        output_path = os.path.join(
            destination_class,
            f"augmented_{generated:04d}.jpg"
        )

        augmented_image.save(
            output_path,
            quality=95
        )

        generated += 1

        current_count += 1


    print(
        f"{class_name:15s} | "
        f"Original: {original_count:3d} | "
        f"Generated: {generated:3d} | "
        f"Final: {current_count:3d}"
    )


# ============================================================
# 12. TRANSFORMS FOR MODEL
# ============================================================

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
# 13. DATASET LOADER FUNCTION
# ============================================================

def load_datasets(
    training_directory
):

    train_dataset = datasets.ImageFolder(
        training_directory,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(
            CLEAN_DATASET,
            "validation"
        ),
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(
            CLEAN_DATASET,
            "test"
        ),
        transform=val_test_transform
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


# ============================================================
# 14. TRAIN MODEL FUNCTION
# ============================================================

def train_model(
    model_name,
    training_directory
):

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING {model_name.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = load_datasets(
        training_directory
    )


    class_names = train_dataset.classes


    print(
        "\nTraining images:",
        len(train_dataset)
    )

    print(
        "Validation images:",
        len(val_dataset)
    )

    print(
        "Test images:",
        len(test_dataset)
    )


    # --------------------------------------------------------
    # LOADERS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RESNET-50
    # --------------------------------------------------------

    print(
        "\nLoading ResNet-50..."
    )


    weights = models.ResNet50_Weights.DEFAULT


    model = models.resnet50(
        weights=weights
    )


    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_names)
    )


    model = model.to(
        device
    )


    criterion = nn.CrossEntropyLoss()


    optimizer = optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY

    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    train_losses = []

    val_losses = []

    train_accs = []

    val_accs = []


    best_val_accuracy = 0.0

    best_weights = copy.deepcopy(
        model.state_dict()
    )


    # --------------------------------------------------------
    # EPOCHS
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            optimizer.zero_grad()


            outputs = model(
                images
            )


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


            _, predictions = torch.max(
                outputs,
                1
            )


            total += labels.size(0)


            correct += (
                predictions == labels
            ).sum().item()


        train_loss = (
            running_loss / total
        )


        train_accuracy = (
            correct / total
        ) * 100


        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_running_loss = 0.0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device
                )

                labels = labels.to(
                    device
                )


                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                val_running_loss += (
                    loss.item()
                    * images.size(0)
                )


                _, predictions = torch.max(
                    outputs,
                    1
                )


                val_total += (
                    labels.size(0)
                )


                val_correct += (
                    predictions == labels
                ).sum().item()


        val_loss = (
            val_running_loss
            / val_total
        )


        val_accuracy = (
            val_correct
            / val_total
        ) * 100


        # ====================================================
        # SAVE
        # ====================================================

        train_losses.append(
            train_loss
        )

        val_losses.append(
            val_loss
        )

        train_accs.append(
            train_accuracy
        )

        val_accs.append(
            val_accuracy
        )


        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
            )

            best_weights = copy.deepcopy(
                model.state_dict()
            )


        print(

            f"{model_name} | "

            f"Epoch [{epoch+1}/{EPOCHS}] | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train Acc: {train_accuracy:.2f}% | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Acc: {val_accuracy:.2f}%"

        )


    # ========================================================
    # BEST MODEL
    # ========================================================

    model.load_state_dict(
        best_weights
    )


    # ========================================================
    # TEST
    # ========================================================

    model.eval()


    labels_all = []

    predictions_all = []


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device
            )


            outputs = model(
                images
            )


            _, predictions = torch.max(
                outputs,
                1
            )


            labels_all.extend(
                labels.numpy()
            )


            predictions_all.extend(
                predictions.cpu().numpy()
            )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        labels_all,
        predictions_all
    )


    precision = precision_score(

        labels_all,

        predictions_all,

        average="weighted",

        zero_division=0

    )


    recall = recall_score(

        labels_all,

        predictions_all,

        average="weighted",

        zero_division=0

    )


    f1 = f1_score(

        labels_all,

        predictions_all,

        average="weighted",

        zero_division=0

    )


    macro_f1 = f1_score(

        labels_all,

        predictions_all,

        average="macro",

        zero_division=0

    )


    report = classification_report(

        labels_all,

        predictions_all,

        target_names=class_names,

        zero_division=0

    )


    cm = confusion_matrix(

        labels_all,

        predictions_all

    )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n")
    print(
        f"{model_name} RESULTS"
    )

    print(
        f"Accuracy  : {accuracy*100:.2f}%"
    )

    print(
        f"Precision : {precision*100:.2f}%"
    )

    print(
        f"Recall    : {recall*100:.2f}%"
    )

    print(
        f"F1 Score  : {f1*100:.2f}%"
    )

    print(
        f"Macro F1  : {macro_f1*100:.2f}%"
    )

    print(
        "\nClassification Report:"
    )

    print(report)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(

        RESULTS_DIR,

        f"{model_name.lower()}_resnet50.pth"

    )


    torch.save(

        model.state_dict(),

        model_path

    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history = pd.DataFrame({

        "Epoch":
            range(
                1,
                EPOCHS + 1
            ),

        "Train_Loss":
            train_losses,

        "Validation_Loss":
            val_losses,

        "Train_Accuracy":
            train_accs,

        "Validation_Accuracy":
            val_accs

    })


    history.to_csv(

        os.path.join(

            RESULTS_DIR,

            f"{model_name.lower()}_history.csv"

        ),

        index=False

    )


    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

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

        f"{model_name} — Confusion Matrix",

        fontsize=16,

        pad=20

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
        rotation=20,
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

            RESULTS_DIR,

            f"{model_name.lower()}_confusion_matrix.png"

        ),

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    # ========================================================
    # SAVE CLASSIFICATION REPORT
    # ========================================================

    with open(

        os.path.join(

            RESULTS_DIR,

            f"{model_name.lower()}_report.txt"

        ),

        "w",

        encoding="utf-8"

    ) as f:

        f.write(
            f"{model_name}\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Best Validation Accuracy: "
            f"{best_val_accuracy:.2f}%\n\n"
        )

        f.write(
            f"Accuracy: "
            f"{accuracy*100:.2f}%\n"
        )

        f.write(
            f"Precision: "
            f"{precision*100:.2f}%\n"
        )

        f.write(
            f"Recall: "
            f"{recall*100:.2f}%\n"
        )

        f.write(
            f"F1 Score: "
            f"{f1*100:.2f}%\n"
        )

        f.write(
            f"Macro F1: "
            f"{macro_f1*100:.2f}%\n\n"
        )

        f.write(
            "Classification Report\n\n"
        )

        f.write(report)


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "model":
            model_name,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "macro_f1":
            macro_f1,

        "best_val_accuracy":
            best_val_accuracy,

        "train_acc":
            train_accs,

        "val_acc":
            val_accs,

        "train_loss":
            train_losses,

        "val_loss":
            val_losses

    }


# ============================================================
# 15. TRAIN BASELINE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 6 — BASELINE MODEL")
print("=" * 70)


baseline_results = train_model(

    "Baseline",

    os.path.join(
        CLEAN_DATASET,
        "train"
    )

)


# ============================================================
# 16. TRAIN FINAL BALANCED MODEL
# ============================================================

print("\n")
print("=" * 70)
print("STEP 7 — FINAL BALANCED MODEL")
print("=" * 70)


final_results = train_model(

    "Final",

    os.path.join(
        BALANCED_DATASET,
        "train"
    )

)


# ============================================================
# 17. COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 8 — BASELINE VS FINAL COMPARISON")
print("=" * 70)


comparison = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "Macro F1",

        "Best Validation Accuracy"

    ],

    "Baseline": [

        baseline_results["accuracy"] * 100,

        baseline_results["precision"] * 100,

        baseline_results["recall"] * 100,

        baseline_results["f1"] * 100,

        baseline_results["macro_f1"] * 100,

        baseline_results["best_val_accuracy"]

    ],

    "Final Balanced": [

        final_results["accuracy"] * 100,

        final_results["precision"] * 100,

        final_results["recall"] * 100,

        final_results["f1"] * 100,

        final_results["macro_f1"] * 100,

        final_results["best_val_accuracy"]

    ]

})


comparison["Improvement"] = (

    comparison["Final Balanced"]

    - comparison["Baseline"]

)


comparison.to_csv(

    os.path.join(

        RESULTS_DIR,

        "baseline_vs_final_comparison.csv"

    ),

    index=False

)


print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 18. COMPARISON GRAPH
# ============================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score",

    "Macro F1"

]


baseline_values = [

    baseline_results["accuracy"] * 100,

    baseline_results["precision"] * 100,

    baseline_results["recall"] * 100,

    baseline_results["f1"] * 100,

    baseline_results["macro_f1"] * 100

]


final_values = [

    final_results["accuracy"] * 100,

    final_results["precision"] * 100,

    final_results["recall"] * 100,

    final_results["f1"] * 100,

    final_results["macro_f1"] * 100

]


x = np.arange(
    len(metrics)
)

width = 0.35


plt.figure(
    figsize=(11, 7)
)


plt.bar(

    x - width / 2,

    baseline_values,

    width,

    label="Baseline"

)


plt.bar(

    x + width / 2,

    final_values,

    width,

    label="Final Balanced"

)


plt.title(

    "Baseline vs Final Model Performance",

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


plt.xticks(

    x,

    metrics,

    rotation=10

)


plt.ylim(
    0,
    105
)


plt.legend(
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

        RESULTS_DIR,

        "baseline_vs_final_comparison.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 19. TRAINING ACCURACY COMPARISON
# ============================================================

plt.figure(
    figsize=(11, 7)
)


plt.plot(

    range(1, EPOCHS + 1),

    baseline_results["train_acc"],

    marker="o",

    linewidth=2,

    label="Baseline Training"

)


plt.plot(

    range(1, EPOCHS + 1),

    baseline_results["val_acc"],

    marker="o",

    linewidth=2,

    linestyle="--",

    label="Baseline Validation"

)


plt.plot(

    range(1, EPOCHS + 1),

    final_results["train_acc"],

    marker="o",

    linewidth=2,

    label="Final Training"

)


plt.plot(

    range(1, EPOCHS + 1),

    final_results["val_acc"],

    marker="o",

    linewidth=2,

    linestyle="--",

    label="Final Validation"

)


plt.title(

    "Baseline vs Final — Accuracy Across Epochs",

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
    range(1, EPOCHS + 1)
)


plt.legend(
    fontsize=10
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

        RESULTS_DIR,

        "baseline_vs_final_accuracy.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 20. TRAINING LOSS COMPARISON
# ============================================================

plt.figure(
    figsize=(11, 7)
)


plt.plot(

    range(1, EPOCHS + 1),

    baseline_results["train_loss"],

    marker="o",

    linewidth=2,

    label="Baseline Training"

)


plt.plot(

    range(1, EPOCHS + 1),

    baseline_results["val_loss"],

    marker="o",

    linewidth=2,

    linestyle="--",

    label="Baseline Validation"

)


plt.plot(

    range(1, EPOCHS + 1),

    final_results["train_loss"],

    marker="o",

    linewidth=2,

    label="Final Training"

)


plt.plot(

    range(1, EPOCHS + 1),

    final_results["val_loss"],

    marker="o",

    linewidth=2,

    linestyle="--",

    label="Final Validation"

)


plt.title(

    "Baseline vs Final — Loss Across Epochs",

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
    range(1, EPOCHS + 1)
)


plt.legend(
    fontsize=10
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

        RESULTS_DIR,

        "baseline_vs_final_loss.png"

    ),

    dpi=300,

    bbox_inches="tight"

)


plt.close()


# ============================================================
# 21. SAVE FINAL SUMMARY
# ============================================================

summary_path = os.path.join(

    REPORT_DIR,

    "complete_experiment_summary.txt"

)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        "NIRMAAN COMPLETE EXPERIMENT SUMMARY\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    f.write(
        "DATA CLEANING\n"
    )

    f.write(
        f"Original images: {total_original}\n"
    )

    f.write(
        f"Exact duplicates removed: "
        f"{exact_duplicate_count}\n"
    )

    f.write(
        f"Very close duplicates removed: "
        f"{near_duplicate_count}\n"
    )

    f.write(
        f"Final clean images: {total_clean}\n\n"
    )


    f.write(
        "BASELINE\n"
    )

    f.write(
        f"Accuracy: "
        f"{baseline_results['accuracy']*100:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{baseline_results['precision']*100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{baseline_results['recall']*100:.2f}%\n"
    )

    f.write(
        f"F1: "
        f"{baseline_results['f1']*100:.2f}%\n\n"
    )


    f.write(
        "FINAL BALANCED MODEL\n"
    )

    f.write(
        f"Accuracy: "
        f"{final_results['accuracy']*100:.2f}%\n"
    )

    f.write(
        f"Precision: "
        f"{final_results['precision']*100:.2f}%\n"
    )

    f.write(
        f"Recall: "
        f"{final_results['recall']*100:.2f}%\n"
    )

    f.write(
        f"F1: "
        f"{final_results['f1']*100:.2f}%\n\n"
    )


    f.write(
        "COMPARISON\n"
    )

    f.write(
        comparison.to_string(
            index=False
        )
    )


# ============================================================
# 22. COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("COMPLETE PIPELINE FINISHED ✅")
print("=" * 70)

print("\nProject folder:")

print(
    PROJECT_DIR
)

print("\nResults folder:")

print(
    RESULTS_DIR
)

print("\nImportant files:")

print(
    "✓ baseline_vs_final_comparison.csv"
)

print(
    "✓ baseline_vs_final_comparison.png"
)

print(
    "✓ baseline_vs_final_accuracy.png"
)

print(
    "✓ baseline_vs_final_loss.png"
)

print(
    "✓ Baseline confusion matrix"
)

print(
    "✓ Final confusion matrix"
)

print(
    "✓ Baseline report"
)

print(
    "✓ Final report"
)

print(
    "✓ complete_experiment_summary.txt"
)

print("\nDone. 🎯")