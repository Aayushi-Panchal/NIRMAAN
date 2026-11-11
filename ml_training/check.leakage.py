import os
import hashlib
from PIL import Image
import numpy as np


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_balanced"

SPLITS = {
    "train": os.path.join(BASE_DIR, "train"),
    "validation": os.path.join(BASE_DIR, "validation"),
    "test": os.path.join(BASE_DIR, "test")
}

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# GET ALL IMAGES
# ============================================================

def get_images(folder):

    image_paths = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(VALID_EXTENSIONS):

                image_paths.append(
                    os.path.join(root, file)
                )

    return image_paths


# ============================================================
# EXACT HASH
# ============================================================

def exact_hash(path):

    h = hashlib.md5()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(8192)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def average_hash(path, size=16):

    try:

        image = Image.open(path).convert("L")

        image = image.resize(
            (size, size)
        )

        pixels = np.array(image)

        avg = pixels.mean()

        bits = pixels > avg

        return bits.flatten()

    except Exception:

        return None


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(hash1, hash2):

    return np.count_nonzero(
        hash1 != hash2
    )


# ============================================================
# LOAD IMAGES
# ============================================================

print("=" * 70)
print("NIRMAAN DATA LEAKAGE CHECK")
print("=" * 70)


all_images = {}

for split, folder in SPLITS.items():

    images = get_images(folder)

    all_images[split] = images

    print(
        f"\n{split.upper()} images: {len(images)}"
    )


# ============================================================
# EXACT DUPLICATE CHECK
# ============================================================

print("\n")
print("=" * 70)
print("1. EXACT DUPLICATE CHECK")
print("=" * 70)


hashes = {}

for split, images in all_images.items():

    for path in images:

        try:

            h = exact_hash(path)

            if h not in hashes:

                hashes[h] = []

            hashes[h].append(
                (split, path)
            )

        except Exception as e:

            print(
                "Error:",
                path,
                e
            )


exact_cross_split = []

for h, items in hashes.items():

    splits_present = set(
        item[0]
        for item in items
    )

    if len(splits_present) > 1:

        exact_cross_split.append(
            items
        )


if len(exact_cross_split) == 0:

    print(
        "\n✓ No exact duplicate found across splits."
    )

else:

    print(
        f"\n⚠️ {len(exact_cross_split)} "
        "exact duplicate groups found across splits."
    )

    for group in exact_cross_split[:20]:

        print("\nDuplicate group:")

        for split, path in group:

            print(
                f"  [{split}] {path}"
            )


# ============================================================
# PERCEPTUAL HASH CHECK
# ============================================================

print("\n")
print("=" * 70)
print("2. NEAR-DUPLICATE CHECK")
print("=" * 70)

print(
    "This checks whether visually very similar images "
    "exist across different splits."
)


phashes = {}

for split, images in all_images.items():

    phashes[split] = []

    for path in images:

        h = average_hash(path)

        if h is not None:

            phashes[split].append(
                (path, h)
            )


# ------------------------------------------------------------
# Compare train vs validation
# ------------------------------------------------------------

def compare_splits(
    split1,
    split2,
    threshold=8
):

    results = []

    for path1, hash1 in phashes[split1]:

        best_distance = 999

        best_path = None

        for path2, hash2 in phashes[split2]:

            distance = hamming_distance(
                hash1,
                hash2
            )

            if distance < best_distance:

                best_distance = distance

                best_path = path2

        if best_distance <= threshold:

            results.append(
                (
                    path1,
                    best_path,
                    best_distance
                )
            )

    return results


train_val = compare_splits(
    "train",
    "validation"
)

train_test = compare_splits(
    "train",
    "test"
)

val_test = compare_splits(
    "validation",
    "test"
)


# ============================================================
# RESULTS
# ============================================================

print("\nTrain vs Validation:")

print(
    "Potential near-duplicates:",
    len(train_val)
)

print("\nTrain vs Test:")

print(
    "Potential near-duplicates:",
    len(train_test)
)

print("\nValidation vs Test:")

print(
    "Potential near-duplicates:",
    len(val_test)
)


# ============================================================
# SHOW TRAIN-TEST POTENTIAL LEAKAGE
# ============================================================

if len(train_test) > 0:

    print("\n")
    print("=" * 70)
    print("⚠️ POTENTIAL TRAIN-TEST NEAR DUPLICATES")
    print("=" * 70)

    for path1, path2, distance in train_test[:30]:

        print(
            f"\nDistance: {distance}"
        )

        print(
            "TRAIN:",
            path1
        )

        print(
            "TEST :",
            path2
        )

else:

    print("\n✓ No strong train-test near-duplicates detected.")


# ============================================================
# SAVE SUMMARY
# ============================================================

report_path = os.path.join(
    os.path.dirname(BASE_DIR),
    "leakage_check_results.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "NIRMAAN DATA LEAKAGE CHECK\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for split, images in all_images.items():

        f.write(
            f"{split}: {len(images)} images\n"
        )

    f.write("\n")

    f.write(
        f"Exact duplicate groups: "
        f"{len(exact_cross_split)}\n"
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