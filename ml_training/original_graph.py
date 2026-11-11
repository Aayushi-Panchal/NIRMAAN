import os
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan dataset"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]
image_extensions = (".jpg", ".jpeg", ".png", ".gif")

counts = []

for class_name in classes:
    folder_path = os.path.join(dataset_path, class_name)

    count = 0

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(image_extensions):
            count += 1

    counts.append(count)

print("Original Dataset Distribution:")

for class_name, count in zip(classes, counts):
    print(f"{class_name}: {count}")

plt.figure(figsize=(8, 5))

bars = plt.bar(classes, counts)

plt.title("Original Nirmaan Dataset Distribution")
plt.xlabel("Civic Problem")
plt.ylabel("Number of Images")

for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()