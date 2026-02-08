import cv2
import numpy as np
import os
import sys
from pathlib import Path
import urllib.request

print("=" * 60)
print("Smart Workspace Monitor - Real Image Test")
print("=" * 60)

try:
    import torch
    import cv2
    import numpy as np
    from ultralytics import YOLO
    print(" All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def capture_from_webcam():
    """Capture an image from webcam"""
    print("\nCapturing image from webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open webcam")
        return None
    
    print("Webcam opened. Press 's' to capture, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to grab frame")
            break
        
        # Display the frame
        cv2.imshow('Webcam - Press "s" to capture, "q" to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save the captured frame
            os.makedirs("dataset/raw_images", exist_ok=True)
            save_path = "dataset/raw_images/webcam_capture.jpg"
            cv2.imwrite(save_path, frame)
            print(f" Image captured and saved: {save_path}")
            captured_image = frame.copy()
            break
        elif key == ord('q'):
            print(" Capture cancelled")
            captured_image = None
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return captured_image

def download_sample_image():
    """Download a sample desk image from the internet"""
    print("\n Downloading sample desk image...")
    url = "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&auto=format&fit=crop"
    save_path = "dataset/raw_images/sample_real_desk.jpg"
    
    try:
        # Download image
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded sample image: {save_path}")
        
        # Load and return the image
        img = cv2.imread(save_path)
        return img
    except Exception as e:
        print(f" Failed to download image: {e}")
        return None

def main():
    # Ensure directories exist
    os.makedirs("dataset/raw_images", exist_ok=True)
    os.makedirs("results/sample_outputs", exist_ok=True)
    
    print("\nChoose image source:")
    print("1. Use webcam to capture my desk")
    print("2. Download sample desk image")
    print("3. Use existing images in dataset/raw_images/")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    img = None
    
    if choice == "1":
        img = capture_from_webcam()
    elif choice == "2":
        img = download_sample_image()
    elif choice == "3":
        # Check for existing images
        image_dir = Path("dataset/raw_images")
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if image_files:
            print(f"\nFound {len(image_files)} images:")
            for i, img_file in enumerate(image_files[:5]):  # Show first 5
                print(f"  {i+1}. {img_file.name}")
            
            if len(image_files) == 1:
                img_path = image_files[0]
            else:
                selection = input(f"\nSelect image (1-{min(5, len(image_files))}): ").strip()
                try:
                    idx = int(selection) - 1
                    if 0 <= idx < len(image_files):
                        img_path = image_files[idx]
                    else:
                        img_path = image_files[0]
                except:
                    img_path = image_files[0]
            
            print(f"Using image: {img_path}")
            img = cv2.imread(str(img_path))
        else:
            print("No existing images found. Using webcam instead.")
            img = capture_from_webcam()
    else:
        print("Invalid choice. Using webcam.")
        img = capture_from_webcam()
    
    if img is None:
        print("Failed to get image. Exiting.")
        return
    
    # Initialize YOLO
    print("\n Loading YOLOv8 model...")
    try:
        model = YOLO('yolov8n.pt')  # Should already be downloaded
        print("YOLOv8n model loaded!")
    except Exception as e:
        print(f" Error loading YOLO: {e}")
        return
    
    # Run detection
    print("\nRunning object detection...")
    try:
        results = model(img, verbose=False)
        result = results[0]
        
        print(f"\nDetection Results:")
        print(f"Total objects detected: {len(result.boxes) if result.boxes is not None else 0}")
        
        # List of objects we care about for workspace monitoring
        workspace_objects = {
            'person': 0,
            'laptop': 0,
            'mouse': 0,
            'keyboard': 0,
            'cell phone': 0,
            'book': 0,
            'cup': 0,
            'bottle': 0,
            'chair': 0,
            'dining table': 0,
            'tv': 0,
            'remote': 0
        }
        
        detected_objects = []
        
        if result.boxes is not None:
            print("\nDetected objects:")
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                # Check if it's one of our target objects
                if class_name in workspace_objects:
                    workspace_objects[class_name] += 1
                
                # Only print high confidence detections
                if confidence > 0.3:
                    print(f"  {i+1:2d}. {class_name:15s} - Confidence: {confidence:.2f}")
                    detected_objects.append({
                        'name': class_name,
                        'confidence': confidence,
                        'bbox': box.xyxy[0].tolist()
                    })
        
        print(f"\n Workspace object summary:")
        for obj_name, count in workspace_objects.items():
            if count > 0:
                print(f"  {obj_name:15s}: {count}")
        
        # Calculate a simple "clutter score"
        total_objects = sum(workspace_objects.values())
        print(f"\n Workspace Metrics:")
        print(f"  Total workspace objects: {total_objects}")
        
        # Simple analysis
        if workspace_objects['laptop'] > 0 or workspace_objects['book'] > 0:
            print(" Learning/Focused objects detected")
        if workspace_objects['cell phone'] > 0:
            print(" Potential distraction detected (phone)")
        if workspace_objects['cup'] > 0 or workspace_objects['bottle'] > 0:
            print(" Refreshments present")
        
        # Save annotated image
        annotated_img = result.plot()
        output_path = "results/sample_outputs/real_detection.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"\n Saved annotated image: {output_path}")
        
        # Display results
        cv2.imshow("YOLO Detection Results", annotated_img)
        print("Displaying results. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save detailed results
        with open("results/sample_outputs/detection_analysis.txt", "w") as f:
            f.write("Workspace Detection Analysis\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Image analyzed: Real desk image\n")
            f.write(f"Total detections: {len(detected_objects)}\n\n")
            
            f.write("Detected Objects:\n")
            f.write("-" * 30 + "\n")
            for obj in detected_objects:
                f.write(f"{obj['name']:15s} - Conf: {obj['confidence']:.2f}\n")
            
            f.write("\nWorkspace Summary:\n")
            f.write("-" * 30 + "\n")
            for obj_name, count in workspace_objects.items():
                if count > 0:
                    f.write(f"{obj_name:15s}: {count}\n")
            
            # Simple productivity assessment
            f.write("\nProductivity Assessment:\n")
            f.write("-" * 30 + "\n")
            productive_items = workspace_objects['laptop'] + workspace_objects['book']
            distracting_items = workspace_objects['cell phone'] + workspace_objects['tv']
            
            if productive_items > distracting_items:
                f.write("Status: LIKELY FOCUSED\n")
                f.write("Reason: More work-related items than distractions\n")
            elif distracting_items > productive_items:
                f.write("Status: LIKELY DISTRACTED\n")
                f.write("Reason: More distractions than work items\n")
            else:
                f.write("Status: NEUTRAL\n")
                f.write("Reason: Balanced workspace\n")
        
        print(f" Saved detailed analysis: results/sample_outputs/detection_analysis.txt")
        
    except Exception as e:
        print(f" Error during detection: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Real Image Test Complete!")
    print("\nNext: We'll build the Feature Extraction module")
    print("=" * 60)

if __name__ == "__main__":
    main()