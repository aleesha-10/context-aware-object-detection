"""
Simple demo of feature extraction
"""
import sys
import os
sys.path.append('.')

from extract_features import FeatureExtractor
from ultralytics import YOLO
import cv2

def main():
    print(" Feature Extraction Demo")
    print("-" * 40)
    
    # Load YOLO
    model = YOLO('yolov8n.pt')
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Use webcam or sample image
    use_webcam = input("Use webcam? (y/n): ").lower().strip() == 'y'
    
    if use_webcam:
        # Capture from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(" Failed to capture from webcam")
            return
        
        # Run detection
        results = model(frame, verbose=False)
        features, _ = extractor.extract_from_yolo_results(results, frame.shape[:2])
        
    else:
        # Use sample image
        image_path = "dataset/raw_images/sample_desk.jpg"
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        features, _, _ = extractor.extract_from_image_file(image_path, model)
    
    # Display key features
    print("\n Key Features Extracted:")
    print("-" * 30)
    
    important_features = [
        'total_objects',
        'productive_count', 
        'distracting_count',
        'focused_score',
        'clutter_score',
        'has_laptop',
        'has_phone',
        'has_book'
    ]
    
    for feat in important_features:
        if feat in features:
            print(f"{feat:20s}: {features[feat]}")
    
    # Simple classification
    print("\n Simple Classification:")
    print("-" * 30)
    
    focused_score = features.get('focused_score', 0)
    
    if focused_score > 0.3:
        print(" Workspace: FOCUSED")
        print(" Good for productivity!")
    elif focused_score < -0.3:
        print("Workspace: DISTRACTED")
        print("   Too many distractions!")
    else:
        print("Workspace: NEUTRAL")
        print(" Could go either way")
    
    # Save features
    os.makedirs("results/metrics", exist_ok=True)
    
    import pandas as pd
    df = pd.DataFrame([features])
    csv_path = "results/metrics/live_features.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nFeatures saved to: {csv_path}")

if __name__ == "__main__":
    main()