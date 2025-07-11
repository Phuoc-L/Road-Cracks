from ultralytics import YOLO
import cv2

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on the frame
    results = model(frame)

    # Plot the results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLOv8n Webcam Detection", annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
