#!/usr/bin/env python3
"""
Test with just a few files to verify the DemBones timeout fixes.
"""

import os
import shutil
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_small_test():
    """Create a small test with just 3-5 files."""
    source_dir = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    test_dir = "D:/Code/neural_marionette/test_small"
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist")
        return None
    
    # Create test directory
    os.makedirs(test_dir, exist_ok=True)
    
    # Copy just the first 3 .obj files
    obj_files = [f for f in os.listdir(source_dir) if f.endswith('.obj')]
    if not obj_files:
        print(f"No .obj files found in {source_dir}")
        return None
    
    test_files = obj_files[:3]  # Just first 3 files
    print(f"Copying {len(test_files)} files to test directory...")
    
    for file in test_files:
        src = os.path.join(source_dir, file)
        dst = os.path.join(test_dir, file)
        shutil.copy2(src, dst)
        print(f"  Copied {file}")
    
    return test_dir

def run_small_test():
    """Run the test with small dataset."""
    test_dir = create_small_test()
    if not test_dir:
        return False
    
    print(f"\nRunning GenerateSkel on small test dataset: {test_dir}")
    
    try:
        from GenerateSkel import process_obj_files
        
        # Run with very conservative settings
        process_obj_files(
            folder_path=test_dir,
            output_dir=os.path.join(test_dir, "test_output_small"),
            is_bind=False,
            visualize=False
        )
        
        print("✓ Small test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Small test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing DemBones with small dataset...")
    print("This should complete quickly (within 2-3 minutes)")
    
    success = run_small_test()
    
    if success:
        print("\n✓ Small test passed! The timeout fixes are working.")
        print("✓ You can now safely run on larger datasets.")
    else:
        print("\n✗ Small test failed. There may still be issues.")
    
    print(f"\nIf the small test works, you can run the full dataset:")
    print(f"python GenerateSkel.py 'D:/Code/VVEditor/Rafa_Approves_hd_4k' --no_visualize")

if __name__ == "__main__":
    main()
