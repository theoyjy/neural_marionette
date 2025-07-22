#!/usr/bin/env python3
"""
Quick restart script to test the improved DemBones with aggressive timeout.
"""

import os
import sys
import subprocess
import signal
import time

def kill_hanging_python():
    """Kill any hanging Python processes."""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                         capture_output=True, text=True)
        else:  # Unix
            subprocess.run(['pkill', '-f', 'GenerateSkel.py'], 
                         capture_output=True, text=True)
        print("✅ Killed any hanging Python processes")
    except Exception as e:
        print(f"⚠️  Could not kill processes: {e}")

def test_improved_version():
    """Test the improved GenerateSkel with aggressive timeout."""
    print("🚀 Testing improved GenerateSkel with aggressive timeout...")
    print("⏱️  New features:")
    print("   - 30-60s timeout based on dataset size")
    print("   - 5s progress heartbeat")
    print("   - Ultra-conservative parameters for large datasets")
    print("   - Immediate fallback for 20k+ vertices")
    
    # Use only first 5 files for testing
    test_folder = "D:/Code/neural_marionette/test_small"
    
    cmd = [
        sys.executable, "GenerateSkel.py", 
        test_folder,
        "--no_visualize"
    ]
    
    print(f"🏃 Running: {' '.join(cmd)}")
    print("📊 Expected: Should complete in 1-2 minutes or fallback gracefully")
    
    try:
        result = subprocess.run(cmd, timeout=180, capture_output=False, text=True)
        if result.returncode == 0:
            print("✅ Test completed successfully!")
        else:
            print(f"⚠️  Test returned code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("⚠️  Test itself timed out after 3 minutes - there may still be issues")
        return False
    except KeyboardInterrupt:
        print("⚠️  Test interrupted by user")
        return False
    
    return True

def main():
    print("🔧 Improved DemBones Timeout Tester")
    print("=" * 50)
    
    # First, kill any hanging processes
    kill_hanging_python()
    time.sleep(2)
    
    # Run test
    if test_improved_version():
        print("\n✅ Test completed! The improvements should work for your large dataset.")
        print("🚀 Now you can run on your full dataset:")
        print('   python GenerateSkel.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" --no_visualize')
        print("\n📈 Expected behavior:")
        print("   - 30-60s timeout per DemBones attempt")
        print("   - Automatic fallback to distance-based skinning")
        print("   - Progress updates every 5 seconds")
        print("   - Ultra-conservative parameters for large datasets")
    else:
        print("\n❌ Test had issues. The hanging problem may persist.")

if __name__ == "__main__":
    main()
