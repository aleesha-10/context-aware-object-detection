import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional

class FeatureExtractor:
    """Extract features from YOLO detections for workspace analysis"""
    
    # Target objects for workspace monitoring
    TARGET_CLASSES = {
        'person': 0,
        'laptop': 63,
        'mouse': 64,
        'keyboard': 66,
        'cell phone': 67,
        'book': 73,
        'cup': 41,
        'bottle': 39,
        'chair': 56,
        'dining table': 60,
        'tv': 62,
        'remote': 65
    }
    
    # Productivity categories
    PRODUCTIVE_CLASSES = ['laptop', 'book', 'keyboard', 'mouse']
    DISTRACTING_CLASSES = ['cell phone', 'tv', 'remote']
    NEUTRAL_CLASSES = ['person', 'cup', 'bottle', 'chair', 'dining table']
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def extract_from_yolo_results(self, yolo_results, image_shape: Tuple[int, int]):
        """
        Extract features from YOLO detection results
        
        Args:
            yolo_results: YOLO results object
            image_shape: (height, width) of original image
            
        Returns:
            Dictionary of extracted features
        """
        height, width = image_shape
        
        # Initialize counters
        workspace_objects = {cls: 0 for cls in self.TARGET_CLASSES.keys()}
        object_areas = []
        object_confidences = []
        object_positions = []
        
        # Parse YOLO results
        result = yolo_results[0] if isinstance(yolo_results, list) else yolo_results
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                # Get class name
                class_name = result.names.get(cls_id, f"class_{cls_id}")
                
                # Only process target classes
                if class_name in self.TARGET_CLASSES:
                    workspace_objects[class_name] += 1
                    
                    # Calculate object area (normalized to image size)
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1) / (width * height)
                    object_areas.append(area)
                    object_confidences.append(conf)
                    
                    # Calculate normalized center position
                    center_x = ((x1 + x2) / 2) / width
                    center_y = ((y1 + y2) / 2) / height
                    object_positions.append((center_x, center_y))
        
        # Calculate features
        features = {}
        
        # 1. Count features
        features['total_objects'] = sum(workspace_objects.values())
        features['productive_count'] = sum(workspace_objects.get(cls, 0) for cls in self.PRODUCTIVE_CLASSES)
        features['distracting_count'] = sum(workspace_objects.get(cls, 0) for cls in self.DISTRACTING_CLASSES)
        features['neutral_count'] = sum(workspace_objects.get(cls, 0) for cls in self.NEUTRAL_CLASSES)
        
        # 2. Specific object counts
        for cls in self.TARGET_CLASSES.keys():
            features[f'count_{cls}'] = workspace_objects.get(cls, 0)
        
        # 3. Ratio features
        if features['total_objects'] > 0:
            features['productive_ratio'] = features['productive_count'] / features['total_objects']
            features['distracting_ratio'] = features['distracting_count'] / features['total_objects']
            features['focused_score'] = features['productive_ratio'] - features['distracting_ratio']
        else:
            features['productive_ratio'] = 0.0
            features['distracting_ratio'] = 0.0
            features['focused_score'] = 0.0
        
        # 4. Area and confidence features
        if object_areas:
            features['avg_object_area'] = np.mean(object_areas)
            features['max_object_area'] = np.max(object_areas)
            features['total_area_coverage'] = np.sum(object_areas)
            features['avg_confidence'] = np.mean(object_confidences)
        else:
            features['avg_object_area'] = 0.0
            features['max_object_area'] = 0.0
            features['total_area_coverage'] = 0.0
            features['avg_confidence'] = 0.0
        
        # 5. Spatial distribution features
        if object_positions:
            positions = np.array(object_positions)
            features['center_x_mean'] = np.mean(positions[:, 0])
            features['center_y_mean'] = np.mean(positions[:, 1])
            features['position_variance'] = np.var(positions)
        else:
            features['center_x_mean'] = 0.5
            features['center_y_mean'] = 0.5
            features['position_variance'] = 0.0
        
        # 6. Clutter score (higher = more cluttered)
        features['clutter_score'] = features['total_objects'] * features['total_area_coverage']
        
        # 7. Workspace type indicators
        features['has_laptop'] = 1 if workspace_objects.get('laptop', 0) > 0 else 0
        features['has_phone'] = 1 if workspace_objects.get('cell phone', 0) > 0 else 0
        features['has_book'] = 1 if workspace_objects.get('book', 0) > 0 else 0
        features['has_person'] = 1 if workspace_objects.get('person', 0) > 0 else 0
        
        return features, workspace_objects
    
    def extract_from_image_file(self, image_path: str, yolo_model):
        """
        Extract features directly from an image file
        
        Args:
            image_path: Path to image file
            yolo_model: Loaded YOLO model
            
        Returns:
            Dictionary of features and object counts
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Run YOLO detection
        results = yolo_model(img, verbose=False)
        
        # Extract features
        height, width = img.shape[:2]
        features, object_counts = self.extract_from_yolo_results(results, (height, width))
        
        return features, object_counts, results
    
    def save_features_to_csv(self, features: Dict, output_path: str, label: str = None):
        """
        Save extracted features to CSV file
        
        Args:
            features: Dictionary of features
            output_path: Path to save CSV
            label: Optional label (e.g., 'focused', 'distracted')
        """
        # Add label if provided
        if label:
            features['label'] = label
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Save to CSV
        if os.path.exists(output_path):
            # Append to existing file
            df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            # Create new file
            df.to_csv(output_path, index=False)
        
        print(f"✅ Features saved to: {output_path}")
    
    def create_feature_dataset(self, image_dir: str, yolo_model, output_csv: str, 
                               label: str = None):
        """
        Create a dataset of features from multiple images
        
        Args:
            image_dir: Directory containing images
            yolo_model: Loaded YOLO model
            output_csv: Path to save CSV dataset
            label: Optional label for all images
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_files:
            print(f"❌ No images found in {image_dir}")
            return
        
        print(f"📊 Processing {len(image_files)} images...")
        
        all_features = []
        
        for i, img_path in enumerate(image_files):
            print(f"  Processing {i+1}/{len(image_files)}: {img_path.name}")
            
            try:
                features, object_counts, _ = self.extract_from_image_file(str(img_path), yolo_model)
                
                # Add image filename
                features['image_filename'] = img_path.name
                
                # Add label if provided
                if label:
                    features['label'] = label
                
                all_features.append(features)
                
                # Print summary for first few images
                if i < 3:
                    print(f"    Objects: {sum(object_counts.values())}")
                    print(f"    Focused Score: {features.get('focused_score', 0):.2f}")
            
            except Exception as e:
                print(f"    ❌ Error processing {img_path.name}: {e}")
        
        # Save to CSV
        if all_features:
            df = pd.DataFrame(all_features)
            df.to_csv(output_csv, index=False)
            print(f"\n✅ Dataset saved: {output_csv}")
            print(f"   Total samples: {len(all_features)}")
            print(f"   Features per sample: {len(all_features[0])}")
        else:
            print("❌ No features extracted")

def main():
    """Main function to test feature extraction"""
    print("=" * 60)
    print("Smart Workspace Monitor - Feature Extraction Test")
    print("=" * 60)
    
    # Import YOLO
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return
    
    # Load YOLO model
    print("\n🤖 Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Test with sample image
    sample_image = "dataset/raw_images/sample_desk.jpg"
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        print("Please run test_with_real_images.py first to create sample images")
        return
    
    print(f"\n🔍 Extracting features from: {sample_image}")
    
    try:
        # Extract features
        features, object_counts, results = extractor.extract_from_image_file(sample_image, model)
        
        print(f"\n📊 Extracted Features:")
        print("-" * 40)
        
        # Display important features
        key_features = {
            'total_objects': 'Total Objects',
            'productive_count': 'Productive Items',
            'distracting_count': 'Distracting Items',
            'focused_score': 'Focused Score',
            'clutter_score': 'Clutter Score',
            'avg_confidence': 'Avg Confidence',
            'has_laptop': 'Has Laptop',
            'has_phone': 'Has Phone',
            'has_book': 'Has Book'
        }
        
        for key, description in key_features.items():
            if key in features:
                print(f"{description:20s}: {features[key]}")
        
        print(f"\n🎯 Object Counts:")
        for obj, count in object_counts.items():
            if count > 0:
                print(f"  {obj:15s}: {count}")
        
        # Save features to CSV
        os.makedirs("results/metrics", exist_ok=True)
        csv_path = "results/metrics/extracted_features.csv"
        extractor.save_features_to_csv(features, csv_path)
        
        # Save detailed features to JSON
        json_path = "results/metrics/feature_details.json"
        with open(json_path, 'w') as f:
            json.dump({
                'features': features,
                'object_counts': object_counts,
                'image_file': sample_image
            }, f, indent=2)
        print(f"💾 Detailed features saved: {json_path}")
        
        # Create annotated image
        if results and hasattr(results[0], 'plot'):
            annotated_img = results[0].plot()
            output_path = "results/sample_outputs/features_detection.jpg"
            cv2.imwrite(output_path, annotated_img)
            print(f"🖼️ Annotated image saved: {output_path}")
            
            # Display
            cv2.imshow("Feature Extraction Results", annotated_img)
            print("\n👀 Displaying results. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Test creating dataset from multiple images
        print(f"\n📁 Testing dataset creation...")
        dataset_csv = "results/metrics/workspace_dataset.csv"
        
        if os.path.exists("dataset/raw_images") and len(list(Path("dataset/raw_images").glob("*.jpg"))) > 0:
            extractor.create_feature_dataset(
                "dataset/raw_images", 
                model, 
                dataset_csv,
                label="unlabeled"  # You can change this to 'focused' or 'distracted' later
            )
        else:
            print("⚠️  Not enough images for dataset creation")
            print("   Add more images to dataset/raw_images/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Feature Extraction Test Complete!")
    print("\nNext: Train context classification model")
    print("=" * 60)

if __name__ == "__main__":
    main()