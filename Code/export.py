from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO('models/Custom_Model2.pt')

# Export the model to OpenVINO format
model.export(format='openvino', dynamic=True)