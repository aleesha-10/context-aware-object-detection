# quick_test.py
import cv2
from ultralytics import YOLO

# Capture from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if ret:
    # Save it
    cv2.imwrite("dataset/raw_images/real_test.jpg", frame)
    
    # Run YOLO
    model = YOLO('yolov8n.pt')
    results = model(frame, verbose=False)
    
    # Check what's detected
    if results[0].boxes is not None:
        print(f" Detected {len(results[0].boxes)} objects!")
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            print(f"  - {class_name} ({conf:.2f})")
    else:
        print("No objects detected")
else:
    print(" Failed to capture")