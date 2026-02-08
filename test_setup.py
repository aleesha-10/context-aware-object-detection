#!/usr/bin/env python3
"""
Quick test to verify the environment setup
"""
import os
import sys
import subprocess
import pkg_resources

def check_packages():
    """Check if required packages are installed"""
    required = {
        'ultralytics',
        'torch',
        'tensorflow',
        'opencv-python',
        'numpy'
    }
    
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        print(f"❌ Missing packages: {missing}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def test_imports():
    """Test importing main libraries"""
    try:
        import torch
        import tensorflow as tf
        import cv2
        import numpy as np
        from ultralytics import YOLO
        
        print("✅ All imports successful")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  TensorFlow: {tf.__version__}")
        print(f"  OpenCV: {cv2.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🔧 Testing Smart Workspace Monitor Setup")
    print("-" * 40)
    
    # Check directory structure
    required_dirs = ['dataset/raw_images', 'src', 'models', 'results']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"⚠️  Directory missing: {dir_path}")
    
    print("-" * 40)
    
    # Check packages
    packages_ok = check_packages()
    
    print("-" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    print("-" * 40)
    
    if packages_ok and imports_ok:
        print("🎉 Setup complete! You can start building.")
        print("\nNext steps:")
        print("1. Add some sample images to dataset/raw_images/")
        print("2. Run: python src/detect_objects.py")
        print("3. Or open notebooks/experiments.ipynb")
    else:
        print("⚠️  Some issues found. Please fix them before proceeding.")

if __name__ == "__main__":
    main()