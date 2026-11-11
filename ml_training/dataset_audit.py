import os
from PIL import Image

dataset_path = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]

total = 0

for class_name in classes:
    folder_path = os.path.join(dataset_path, class_name)

    count = 0
    formats = set()
    sizes = set()
    corrupted = 0

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img.verify()

                with Image.open(file_path) as img:
                    formats.add(img.format)
                    sizes.add(img.size)

                count += 1

            except:
                corrupted += 1

    total += count

    print(f"\n{class_name.upper()}")
    print("Images:", count)
    print("Formats:", formats)
    print("Different dimensions:", len(sizes))
    print("Corrupted:", corrupted)

print("\n--------------------")
print("TOTAL IMAGES:", total)