import os
import shutil
import random

from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

SOURCE_TRAIN = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset\train"

OUTPUT_TRAIN = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_clean_balanced\train"

TARGET_COUNT = 310

SEED = 42

random.seed(SEED)


# ============================================================
# SETTINGS
# ============================================================

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# AUGMENTATION
# ============================================================

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


# ============================================================
# CREATE OUTPUT
# ============================================================

os.makedirs(
    OUTPUT_TRAIN,
    exist_ok=True
)


# ============================================================
# FIND CLASSES
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(SOURCE_TRAIN)
    if os.path.isdir(
        os.path.join(SOURCE_TRAIN, folder)
    )
])


print("=" * 60)
print("NIRMAAN CLEAN BALANCED TRAINING DATASET")
print("=" * 60)

print("\nClasses:")

for c in classes:
    print("-", c)


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in classes:

    source_class = os.path.join(
        SOURCE_TRAIN,
        class_name
    )

    output_class = os.path.join(
        OUTPUT_TRAIN,
        class_name
    )

    os.makedirs(
        output_class,
        exist_ok=True
    )

    images = sorted([
        file
        for file in os.listdir(source_class)
        if file.lower().endswith(
            VALID_EXTENSIONS
        )
    ])

    original_count = len(images)

    print("\n" + "-" * 50)
    print("Class:", class_name)
    print("Original:", original_count)


    # --------------------------------------------------------
    # If class has MORE than 310
    # --------------------------------------------------------

    if original_count > TARGET_COUNT:

        selected = random.sample(
            images,
            TARGET_COUNT
        )

    else:

        selected = images


    # --------------------------------------------------------
    # COPY ORIGINAL TRAIN IMAGES
    # --------------------------------------------------------

    for i, image_name in enumerate(selected):

        source_path = os.path.join(
            source_class,
            image_name
        )

        extension = os.path.splitext(
            image_name
        )[1]

        output_path = os.path.join(
            output_class,
            f"original_{i:04d}{extension}"
        )

        shutil.copy2(
            source_path,
            output_path
        )


    current_count = len(selected)

    generated = 0


    # --------------------------------------------------------
    # GENERATE UNTIL 310
    # --------------------------------------------------------

    while current_count < TARGET_COUNT:

        source_name = random.choice(
            selected
        )

        source_path = os.path.join(
            source_class,
            source_name
        )

        image = Image.open(
            source_path
        ).convert("RGB")

        augmented = augmentation(
            image
        )

        output_path = os.path.join(
            output_class,
            f"augmented_{generated:04d}.jpg"
        )

        augmented.save(
            output_path,
            quality=95
        )

        generated += 1
        current_count += 1


    print("Generated:", generated)
    print("Final:", current_count)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n")
print("=" * 60)
print("FINAL COUNTS")
print("=" * 60)

total = 0

for class_name in classes:

    class_path = os.path.join(
        OUTPUT_TRAIN,
        class_name
    )

    count = len([
        f
        for f in os.listdir(class_path)
        if f.lower().endswith(
            VALID_EXTENSIONS
        )
    ])

    total += count

    print(
        f"{class_name:15s}: {count}"
    )

print("-" * 40)
print("TOTAL:", total)

print("\nSaved at:")
print(OUTPUT_TRAIN)

print("\nValidation and Test were NOT touched.")