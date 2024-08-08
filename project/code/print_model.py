from ultralytics import YOLO

model = YOLO(r'yoloV8n.pt')  # Load model
print(model.model)
