#!/usr/bin/env python3
"""Quick start script for the on-device learning demo."""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'streamlit', 'numpy', 'plotly', 
        'sklearn', 'omegaconf', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    return True


def run_training():
    """Run the training script."""
    print("🚀 Starting training...")
    try:
        subprocess.run([
            sys.executable, "scripts/train.py",
            "--config", "configs/device/default.yaml",
            "--log-level", "INFO"
        ], check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False


def run_demo():
    """Run the Streamlit demo."""
    print("🎯 Starting interactive demo...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "demo/streamlit_demo.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        return True


def main():
    """Main function."""
    print("🧠 On-Device Learning Implementation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("demo").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies are available")
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run training script")
    print("2. Launch interactive demo")
    print("3. Run both (training + demo)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_training()
    elif choice == "2":
        run_demo()
    elif choice == "3":
        if run_training():
            print("\n" + "=" * 50)
            input("Press Enter to start the demo...")
            run_demo()
    elif choice == "4":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()
