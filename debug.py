import os
import shutil
from torchvision import datasets, transforms

# ================= CONFIG =================
TRAIN_DIR = "dataset_balanced/traffic_Data/DATA"
TEST_SORTED_DIR = "dataset/traffic_Data/TEST_SORTED"

# ================= LOAD TRAIN CLASSES =================
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transforms.ToTensor()
)

train_classes = set(train_dataset.class_to_idx.keys())

print("✅ Train classes:", train_classes)

# ================= FILTER TEST CLASSES =================
removed = 0
kept = 0

for cls in os.listdir(TEST_SORTED_DIR):
    cls_path = os.path.join(TEST_SORTED_DIR, cls)

    if not os.path.isdir(cls_path):
        continue

    if cls not in train_classes:
        shutil.rmtree(cls_path)
        removed += 1
        print(f"❌ Removed untrained class: {cls}")
    else:
        kept += 1

print("\n✅ Test dataset cleaned")
print(f"Classes kept   : {kept}")
print(f"Classes removed: {removed}")
