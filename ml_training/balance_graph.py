import os
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\LENOVO\OneDrive\Documents\nirmaan_balanced\train"

classes = ["garbage", "potholes", "streetlights", "waterlogging"]
extensions = (".jpg", ".jpeg", ".png", ".gif")

counts = []

for class_name in classes:
    folder = os.path.join(dataset_path, class_name)

    count = 0

    for file in os.listdir(folder):
        if file.lower().endswith(extensions):
            count += 1

    counts.append(count)

# Print counts
print("\nFINAL BALANCED DATASET")

for class_name, count in zip(classes, counts):
    print(f"{class_name}: {count}")

print("Total:", sum(counts))

# Create graph
plt.figure(figsize=(9, 6))

bars = plt.bar(classes, counts)

plt.title("Final Balanced Training Dataset")
plt.xlabel("Civic Problem")
plt.ylabel("Number of Images")

# Show values on bars
for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(count),
        ha="center",
        va="bottom"
    )

plt.ylim(0, max(counts) + 40)

plt.tight_layout()

# Save high-quality image
plt.savefig(
    r"C:\Users\LENOVO\OneDrive\Documents\balanced_dataset_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()