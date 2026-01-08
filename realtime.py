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

# Detection parameters
WINDOW_SIZE = 96
STRIDE = 48
CONF_THRESHOLD = 0.97        # strong confidence
MARGIN_THRESHOLD = 0.20      # top1 - top2 margin
MAX_DETECTIONS = 3           # max boxes per frame

# ================= LOAD LABELS =================
labels_df = pd.read_csv(LABELS_CSV)
classid_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))

# ================= LOAD TRAIN CLASSES =================
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transforms.ToTensor()
)
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

# ================= LOAD MODEL (SAFE) =================
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
num_classes = state_dict["fc.weight"].shape[0]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

# ================= OPENCV =================
cap = cv2.VideoCapture(0)
print("🚦 Traffic Sign Detection (Sliding Window)")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    detections = []

    # ===== SLIDING WINDOW SCAN =====
    for y in range(0, h - WINDOW_SIZE, STRIDE):
        for x in range(0, w - WINDOW_SIZE, STRIDE):
            roi = frame[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]

            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img = transform(pil_img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(img)
                probs = torch.softmax(outputs, dim=1)[0]

            top2 = torch.topk(probs, 2)
            conf1 = top2.values[0].item()
            conf2 = top2.values[1].item()
            pred_idx = top2.indices[0].item()

            # ===== BACKGROUND REJECTION =====
            if (
                conf1 > CONF_THRESHOLD and
                (conf1 - conf2) > MARGIN_THRESHOLD and
                pred_idx in idx_to_classid
            ):
                class_id = idx_to_classid[pred_idx]
                label = classid_to_name.get(class_id, "Traffic Sign")
                detections.append((x, y, label, conf1))

    # ===== KEEP ONLY BEST DETECTIONS =====
    detections = sorted(detections, key=lambda x: x[3], reverse=True)
    detections = detections[:MAX_DETECTIONS]

    # ===== DRAW DETECTIONS =====
    for (x, y, label, conf) in detections:
        cv2.rectangle(
            frame,
            (x, y),
            (x + WINDOW_SIZE, y + WINDOW_SIZE),
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"{label} ({conf*100:.1f}%)",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
