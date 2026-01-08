from ultralytics import YOLO
import pandas as pd
import os

# ================= PATHS =================
MODEL_PATH = "runs/detect/train3/weights/best.pt"
LABELS_CSV = "dataset/traffic_Data/labels.csv"
TRAIN_DIR = "dataset_balanced/traffic_Data/DATA"

# ================= LOAD LABEL NAMES =================
labels_df = pd.read_csv(LABELS_CSV)
realid_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))

# ================= BUILD YOLO-ID → REAL-ID MAP =================
real_class_ids = sorted(
    int(d) for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
)

yolo_to_real = {idx: real_id for idx, real_id in enumerate(real_class_ids)}

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# ================= RUN LIVE DETECTION =================
results = model.predict(
    source=0,
    show=True,
    conf=0.4,
    stream=True
)

# ================= MODIFY LABELS =================
for r in results:
    if r.boxes is None:
        continue

    for box in r.boxes:
        yolo_id = int(box.cls[0])
        real_id = yolo_to_real.get(yolo_id, None)
        label_name = realid_to_name.get(real_id, f"Class {real_id}")

        # Override YOLO label
        box.cls = box.cls  # keep class
        box.conf = box.conf

        # Print label in terminal (YOLO GUI already shows it)
        print(f"Detected: {label_name}")
