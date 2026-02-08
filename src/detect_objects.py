import cv2
import numpy as np
import os
import sys
from pathlib import Path

print("=" * 60)
print("Smart Workspace Monitor - YOLO Detection Test")
print("=" * 60)

# Test basic imports
try:
    import torch
    import cv2
    import numpy as np
    from ultralytics import YOLO
    print("✅ All imports successful")
    print(f"PyTorch: {torch.__version__}")
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Create a sample desk image if none exists
def create_sample_desk_image():
    """Create a synthetic desk image for testing"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a desk (brown surface)
    cv2.rectangle(img, (0, 300), (640, 480), (42, 42, 165), -1)  # Brown desk
    
    # Draw some desk objects
    # Laptop (silver rectangle)
    cv2.rectangle(img, (100, 200), (400, 350), (200, 200, 200), -1)
    cv2.rectangle(img, (120, 220), (380, 330), (50, 50, 50), 2)
    
    # Coffee cup (cylinder shape)
    cv2.ellipse(img, (500, 250), (30, 15), 0, 0, 360, (0, 100, 255), -1)
    cv2.rectangle(img, (470, 250), (530, 320), (0, 100, 255), -1)
    cv2.ellipse(img, (500, 320), (30, 15), 0, 0, 360, (0, 50, 200), -1)
    
    # Book (red rectangle)
    cv2.rectangle(img, (50, 150), (150, 250), (0, 0, 255), -1)
    cv2.putText(img, "BOOK", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Phone (small black rectangle)
    cv2.rectangle(img, (450, 150), (520, 200), (50, 50, 50), -1)
    cv2.rectangle(img, (460, 160), (510, 190), (100, 100, 100), -1)
    
    return img

def main():
    # Ensure directories exist
    os.makedirs("dataset/raw_images", exist_ok=True)
    os.makedirs("results/sample_outputs", exist_ok=True)
    
    # Create or load sample image
    sample_path = "dataset/raw_images/sample_desk.jpg"
    if os.path.exists(sample_path):
        print(f"\n📁 Using existing sample image: {sample_path}")
        img = cv2.imread(sample_path)
    else:
        print(f"\n🖼️ Creating sample desk image...")
        img = create_sample_desk_image()
        cv2.imwrite(sample_path, img)
        print(f"✅ Created sample image: {sample_path}")
    
    # Display the created image
    cv2.imshow("Sample Desk Image", img)
    print("📸 Displaying sample image. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Initialize YOLO detector
    print("\n🤖 Loading YOLOv8 model...")
    try:
        # This will download the model if not already present
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8n model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        return
    
    # Run object detection
    print("\n🔍 Running object detection...")
    try:
        results = model(img, verbose=False)
        
        # Get the first result
        result = results[0]
        
        # Show detection summary
        print(f"\n📊 Detection Results:")
        print(f"Total objects detected: {len(result.boxes) if result.boxes is not None else 0}")
        
        # Count specific objects we care about
        target_classes = ['person', 'laptop', 'cell phone', 'book', 'cup', 'bottle', 'chair']
        class_counts = {cls: 0 for cls in target_classes}
        
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name in target_classes:
                    class_counts[class_name] += 1
                    print(f"  - {class_name}: {confidence:.2f}")
        
        print(f"\n🎯 Target object counts:")
        for cls, count in class_counts.items():
            if count > 0:
                print(f"  {cls}: {count}")
        
        # Save annotated image
        annotated_img = result.plot()
        output_path = "results/sample_outputs/first_detection.jpg"
        cv2.imwrite(output_path, annotated_img)
        print(f"\n💾 Saved annotated image: {output_path}")
        
        # Display the result
        cv2.imshow("YOLO Detection Results", annotated_img)
        print("👀 Displaying detection results. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save detection data for later use
        detection_data = []
        if result.boxes is not None:
            for i, box in enumerate(result.boxes):
                detection_data.append({
                    'object_id': i,
                    'class': model.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist(),
                    'area': (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                })
        
        # Save as simple text file
        with open("results/sample_outputs/detection_summary.txt", "w") as f:
            f.write("Object Detection Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Image: {sample_path}\n")
            f.write(f"Total detections: {len(detection_data)}\n\n")
            for det in detection_data:
                f.write(f"Object {det['object_id']}:\n")
                f.write(f"  Class: {det['class']}\n")
                f.write(f"  Confidence: {det['confidence']:.2f}\n")
                f.write(f"  BBox: {det['bbox']}\n")
                f.write(f"  Area: {det['area']:.0f} pixels\n")
                f.write("-" * 20 + "\n")
        
        print(f"📝 Saved detection summary: results/sample_outputs/detection_summary.txt")
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Phase 1 Complete! YOLO is working!")
    print("\nNext steps:")
    print("1. Add real desk images to dataset/raw_images/")
    print("2. We'll build the feature extraction module")
    print("=" * 60)

if __name__ == "__main__":
    main()