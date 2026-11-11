import os
from PIL import Image

dataset_path = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]

for class_name in classes:
    folder_path = os.path.join(dataset_path, class_name)

    print(f"\n--- {class_name.upper()} ---")

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img.verify()

            except Exception as e:
                print("Problem file:", file_name)
                print("Reason:", e)