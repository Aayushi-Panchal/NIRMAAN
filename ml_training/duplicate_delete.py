import os
import hashlib

dataset_path = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]
image_extensions = (".jpg", ".jpeg", ".png", ".gif")

hashes = {}
duplicates = []

for class_name in classes:
    folder_path = os.path.join(dataset_path, class_name)

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(image_extensions):
            continue

        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        if file_hash in hashes:
            duplicates.append(file_path)
        else:
            hashes[file_hash] = file_path

print("Duplicates found:", len(duplicates))

for file_path in duplicates:
    print("Deleting:", file_path)
    os.remove(file_path)

print("\nDuplicate removal complete!")pip install torch torchvision scikit-learn matplotlib seaborn pandas