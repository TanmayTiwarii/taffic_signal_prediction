from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="yolo_dataset/data.yaml",
        epochs=10,
        imgsz=416,
        batch=16,
        device=0
    )

if __name__ == "__main__":
    main()
