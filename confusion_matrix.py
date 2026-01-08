import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================= CONFIG =================
TEST_DIR = "dataset/traffic_Data/TEST_SORTED"
MODEL_PATH = "traffic_sign_model.pth"
NUM_CLASSES = 58
IMG_SIZE = 64
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= TRANSFORMS =================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ================= DATASET =================
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Test samples: {len(test_dataset)}")

# ================= MODEL =================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ================= PREDICTION =================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ================= METRICS =================
acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# ================= PLOT =================
plt.figure(figsize=(18, 16))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=range(NUM_CLASSES),
    yticklabels=range(NUM_CLASSES),
    cbar=True,
    square=True
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Traffic Sign Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("📁 Confusion matrix saved as confusion_matrix.png")
