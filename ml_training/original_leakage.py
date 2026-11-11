import os
import hashlib
from PIL import Image
import numpy as np


# ============================================================
# ORIGINAL NIRMAAN DATASET
# ============================================================

BASE_DIR = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    result = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(VALID_EXTENSIONS):

                result.append(
                    os.path.join(root, file)
                )

    return result


# ============================================================
# EXACT HASH
# ============================================================

def exact_hash(path):

    h = hashlib.md5()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8192),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def perceptual_hash(path, size=16):

    try:

        image = Image.open(
            path
        ).convert("L")

        image = image.resize(
            (size, size)
        )

        pixels = np.array(
            image
        )

        average = pixels.mean()

        return pixels > average

    except:

        return None


# ============================================================
# HAMMING DISTANCE
# ============================================================

def distance(hash1, hash2):

    return np.count_nonzero(
        hash1 != hash2
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NIRMAAN ORIGINAL DATASET LEAKAGE CHECK")
print("=" * 70)


splits = {

    "TRAIN":
        get_images(TRAIN_DIR),

    "VALIDATION":
        get_images(VAL_DIR),

    "TEST":
        get_images(TEST_DIR)

}


for name, images in splits.items():

    print(
        f"{name} images: {len(images)}"
    )


# ============================================================
# EXACT DUPLICATES
# ============================================================

print("\n")
print("=" * 70)
print("1. EXACT DUPLICATE CHECK")
print("=" * 70)


hash_map = {}

for split, images in splits.items():

    for path in images:

        try:

            h = exact_hash(path)

            if h not in hash_map:

                hash_map[h] = []

            hash_map[h].append(
                (split, path)
            )

        except Exception as e:

            print(
                "Error:",
                path
            )


duplicate_groups = []

for h, items in hash_map.items():

    split_names = set(
        item[0]
        for item in items
    )

    if len(split_names) > 1:

        duplicate_groups.append(
            items
        )


if len(duplicate_groups) == 0:

    print(
        "\n✓ No exact duplicates across splits."
    )

else:

    print(
        "\n⚠️ Exact duplicate groups:",
        len(duplicate_groups)
    )

    for group in duplicate_groups[:20]:

        print("\nDuplicate:")

        for split, path in group:

            print(
                f"[{split}] {path}"
            )


# ============================================================
# PERCEPTUAL HASH
# ============================================================

print("\n")
print("=" * 70)
print("2. NEAR-DUPLICATE CHECK")
print("=" * 70)

print(
    "Comparing original train, validation and test images..."
)


phashes = {}

for split, images in splits.items():

    phashes[split] = []

    for path in images:

        h = perceptual_hash(path)

        if h is not None:

            phashes[split].append(
                (path, h)
            )


# ============================================================
# COMPARE TWO SPLITS
# ============================================================

def compare(
    split1,
    split2,
    threshold=6
):

    matches = []

    for path1, hash1 in phashes[split1]:

        best_distance = 999

        best_path = None

        for path2, hash2 in phashes[split2]:

            d = distance(
                hash1,
                hash2
            )

            if d < best_distance:

                best_distance = d

                best_path = path2

        if best_distance <= threshold:

            matches.append(
                (
                    path1,
                    best_path,
                    best_distance
                )
            )

    return matches


# ============================================================
# TRAIN vs VALIDATION
# ============================================================

train_val = compare(
    "TRAIN",
    "VALIDATION"
)


# ============================================================
# TRAIN vs TEST
# ============================================================

train_test = compare(
    "TRAIN",
    "TEST"
)


# ============================================================
# VALIDATION vs TEST
# ============================================================

val_test = compare(
    "VALIDATION",
    "TEST"
)


# ============================================================
# RESULTS
# ============================================================

print("\nTrain ↔ Validation:")
print(
    "Potential near-duplicates:",
    len(train_val)
)

print("\nTrain ↔ Test:")
print(
    "Potential near-duplicates:",
    len(train_test)
)

print("\nValidation ↔ Test:")
print(
    "Potential near-duplicates:",
    len(val_test)
)


# ============================================================
# SHOW TRAIN-TEST MATCHES
# ============================================================

if len(train_test) > 0:

    print("\n")
    print("=" * 70)
    print("TRAIN ↔ TEST POTENTIAL MATCHES")
    print("=" * 70)

    for train_path, test_path, d in train_test[:30]:

        print(
            f"\nDistance: {d}"
        )

        print(
            "TRAIN:",
            train_path
        )

        print(
            "TEST :",
            test_path
        )

else:

    print(
        "\n✓ No strong train-test near-duplicates detected."
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    r"C:\Users\LENOVO\OneDrive\Documents"
    r"\original_leakage_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NIRMAAN ORIGINAL DATASET LEAKAGE CHECK\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for split, images in splits.items():

        f.write(
            f"{split}: {len(images)} images\n"
        )

    f.write("\n")

    f.write(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups)}\n"
    )

    f.write(
        f"Train-Validation near duplicates: "
        f"{len(train_val)}\n"
    )

    f.write(
        f"Train-Test near duplicates: "
        f"{len(train_test)}\n"
    )

    f.write(
        f"Validation-Test near duplicates: "
        f"{len(val_test)}\n"
    )


print("\n")
print("=" * 70)
print("CHECK COMPLETED")
print("=" * 70)

print(
    "\nReport saved at:"
)

print(
    report_path
)