import tkinter as tk
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'ultralytics': 'ultralytics'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   - {package}")
        print("\nInstall missing packages using:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and paths"""
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        # Set environment variables for better performance
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU if available
        
        return True
    except Exception as e:
        print(f"Environment setup error: {e}")
        return False

def main():
    print("🛣️ Traffic Regulation")
    print("=" * 70)
    
    # Check dependencies
    print("📋 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before running the application.")
        sys.exit(1)
    print("✅ All dependencies satisfied.")
    
    # Setup environment
    print("🔧 Setting up environment...")
    if not setup_environment():
        print("\n⚠️ Environment setup failed, but continuing...")
    else:
        print("✅ Environment configured.")
    
    # Import and start dual window GUI
    try:
        print("🚀 Starting application...")
        from gui import TrafficOptimizationApp
        
        # Create main root for the application
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        # Create dual window application
        app = TrafficOptimizationApp()
        
        print("✅ Dual window application started successfully!")
        print("📺 Both windows should now be visible:")
        print("=" * 70)
        
        # Start main loop
        root.mainloop()
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all required files are in the same directory:")
        print("   - main_improved.py")
        print("   - gui_improved.py") 
        print("   - utils_improved.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        print("\n📋 Error details:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
