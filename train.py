import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ================= CONFIG =================
DATA_DIR = "dataset_balanced/traffic_Data/DATA"
NUM_CLASSES = 58
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ================= DATASET =================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# IMPORTANT: validation should NOT have augmentation
val_ds.dataset.transform = val_transform

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples:   {len(val_ds)}")

# ================= MODEL =================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ================= TRAINING SETUP =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN LOOP =================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = 100 * train_correct / train_total

    # ================= VALIDATION =================
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # ================= SAVE BEST MODEL =================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "traffic_sign_model.pth")
        print("✅ Model saved")

print("\n🎉 Training Complete")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
