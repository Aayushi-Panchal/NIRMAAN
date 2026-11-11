import os
from PIL import Image

dataset_path = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]
image_extensions = (".jpg", ".jpeg", ".png", ".gif")

for class_name in classes:
    folder_path = os.path.join(dataset_path, class_name)

    print(f"\n--- {class_name.upper()} ---")

    total = 0
    small = 0
    grayscale = 0

    min_width = float("inf")
    min_height = float("inf")
    max_width = 0
    max_height = 0

    for file_name in os.listdir(folder_path):

        if not file_name.lower().endswith(image_extensions):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            with Image.open(file_path) as img:
                width, height = img.size

                total += 1

                min_width = min(min_width, width)
                min_height = min(min_height, height)
                max_width = max(max_width, width)
                max_height = max(max_height, height)

                if width < 224 or height < 224:
                    small += 1

                if img.mode in ["L", "1"]:
                    grayscale += 1

        except:
            pass

    print("Images:", total)
    print("Minimum size:", min_width, "x", min_height)
    print("Maximum size:", max_width, "x", max_height)
    print("Images smaller than 224x224:", small)
    print("Grayscale images:", grayscale)