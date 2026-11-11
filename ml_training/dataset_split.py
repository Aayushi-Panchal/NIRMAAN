import os
import shutil
from sklearn.model_selection import train_test_split

# Original dataset
source = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

# New final dataset
destination = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_final"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]

# Image extensions
extensions = (".jpg", ".jpeg", ".png", ".gif")

# Create folders
for split in ["train", "validation", "test"]:
    for class_name in classes:
        os.makedirs(
            os.path.join(destination, split, class_name),
            exist_ok=True
        )

# Split each class separately
for class_name in classes:

    source_folder = os.path.join(source, class_name)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith(extensions)
    ]

    # 70% train, 30% temporary
    train_images, temp_images = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    # 15% validation, 15% test
    validation_images, test_images = train_test_split(
        temp_images,
        test_size=0.50,
        random_state=42
    )

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        for image in split_images:

            source_path = os.path.join(source_folder, image)
            destination_path = os.path.join(
                destination,
                split_name,
                class_name,
                image
            )

            shutil.copy2(source_path, destination_path)

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(validation_images)}, "
        f"Test={len(test_images)}"
    )

print("\nDataset split complete!")