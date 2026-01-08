import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image

# ================= CONFIG =================
MODEL_PATH = "traffic_sign_model.pth"
TRAIN_DIR = "dataset_balanced/traffic_Data/DATA"
LABELS_CSV = "dataset/traffic_Data/labels.csv"
IMG_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.6

# ================= LOAD LABELS CSV =================
labels_df = pd.read_csv(LABELS_CSV)
classid_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))

# ================= LOAD TRAIN CLASSES =================
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transforms.ToTensor()
)

# maps model output index -> actual class id (folder name)
idx_to_classid = {v: int(k) for k, v in train_dataset.class_to_idx.items()}

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= LOAD MODEL (SAFE FIX) =================
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

# infer output classes from checkpoint
num_classes_ckpt = state_dict["fc.weight"].shape[0]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes_ckpt)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# ================= OPENCV =================
cap = cv2.VideoCapture(0)
print("🚦 Live Traffic Sign Prediction")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_img = Image.fromarray(rgb)

    # Apply transforms
    img = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    label_text = "Unknown"
    color = (0, 0, 255)

    if conf.item() >= CONF_THRESHOLD and pred_idx.item() in idx_to_classid:
        class_id = idx_to_classid[pred_idx.item()]
        class_name = classid_to_name.get(class_id, f"Class {class_id}")
        label_text = f"{class_name} ({conf.item()*100:.1f}%)"
        color = (0, 255, 0)

    # Display result
    cv2.putText(
        frame,
        label_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
