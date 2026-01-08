import os
import shutil
from torchvision import datasets, transforms

# ================= CONFIG =================
RAW_TEST_DIR = "dataset/traffic_Data/TEST"
TRAIN_DIR = "dataset_balanced/traffic_Data/DATA"
TARGET_TEST_DIR = "dataset/traffic_Data/TEST_SORTED"

# ================= LOAD TRAIN CLASSES =================
train_ds = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transforms.ToTensor()
)

valid_classes = set(train_ds.class_to_idx.keys())
print("✅ Valid train classes:", valid_classes)

# ================= RESET TARGET DIR =================
if os.path.exists(TARGET_TEST_DIR):
    shutil.rmtree(TARGET_TEST_DIR)

os.makedirs(TARGET_TEST_DIR, exist_ok=True)

kept = 0
skipped = 0

# ================= PROCESS TEST IMAGES =================
for img_name in os.listdir(RAW_TEST_DIR):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Extract class from filename
    class_id = img_name[:3]          # "000"
    class_id = str(int(class_id))    # "0"

    # Skip untrained classes
    if class_id not in valid_classes:
        skipped += 1
        continue

    class_dir = os.path.join(TARGET_TEST_DIR, class_id)
    os.makedirs(class_dir, exist_ok=True)

    src = os.path.join(RAW_TEST_DIR, img_name)
    dst = os.path.join(class_dir, img_name)
    shutil.copy(src, dst)
    kept += 1

print("\n✅ TEST_SORTED rebuilt")
print(f"Images kept   : {kept}")
print(f"Images skipped: {skipped}")
