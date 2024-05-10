from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('yolov8n.pt')  # Adjust model as needed

# Open your video
cap = cv2.VideoCapture('E:\Учеба\ВРЕМЕННОЕ\Object-Detection-YOLO\Video\sec.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Extract detected objects (for example, filtering by confidence and class)
    for *box, conf, cls in results.pred[0]:
        if conf > 0.5:  # Example confidence threshold
            # Your custom processing here
            pass

    # Example: Show the frame with detections
    cv2.imshow('Frame', np.squeeze(results.pred()))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()