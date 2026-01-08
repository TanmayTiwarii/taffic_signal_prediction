from ultralytics import YOLO
import pandas as pd
import os

# ================= PATHS =================
MODEL_PATH = "runs/detect/train3/weights/best.pt"
LABELS_CSV = "dataset/traffic_Data/labels.csv"
TRAIN_DIR = "dataset_balanced/traffic_Data/DATA"

# ================= LOAD CSV LABELS =================
labels_df = pd.read_csv(LABELS_CSV)
realid_to_name = dict(zip(labels_df["ClassId"], labels_df["Name"]))

# ================= BUILD YOLO-ID -> LABEL NAME MAP =================
real_class_ids = sorted(
    int(d) for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
)

# YOLO expects index -> name
yolo_names = {idx: realid_to_name.get(real_id, f"Class {real_id}")
              for idx, real_id in enumerate(real_class_ids)}

# ================= LOAD MODEL =================
model = YOLO(MODEL_PATH)

# 🔥 THIS IS THE KEY LINE (CORRECT PLACE)
model.model.names = yolo_names

# ================= LIVE INFERENCE =================
model.predict(
    source=0,
    show=True,
    conf=0.4,
    save=False
)
