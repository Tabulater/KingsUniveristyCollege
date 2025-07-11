#!/usr/bin/env python3
"""
Queue Theory Neural Network GUI Launcher
========================================

This script launches the GUI application for queue prediction.

Usage:
    python launch_gui.py

Features:
- Basic prediction using lambda and Lq
- Enhanced prediction using all variables
- Model comparison
- Data analysis
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'tkinter',
        'numpy', 
        'pandas',
        'matplotlib',
        'sklearn'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Missing required modules:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nPlease install missing modules using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_model_files():
    """Check if trained model files exist"""
    required_files = [
        'improved_queue_nn_model.pkl',
        'enhanced_queue_nn_model.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease train the models first using:")
        print("python improved_neural_network.py")
        print("python enhanced_neural_network.py")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🚀 Queue Theory Neural Network GUI Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies available")
    
    # Check model files
    print("Checking model files...")
    if not check_model_files():
        sys.exit(1)
    print("✅ All model files available")
    
    # Launch GUI
    print("Launching GUI...")
    try:
        from queue_prediction_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 