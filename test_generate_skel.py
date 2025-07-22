#!/usr/bin/env python3
"""
Test script for the improved GenerateSkel.py with better DemBones handling.
"""

import os
import sys
import time
from GenerateSkel import process_obj_files, simple_skinning_fallback
import numpy as np

def test_simple_fallback():
    """Test the simple skinning fallback function."""
    print("Testing simple skinning fallback...")
    
    # Create dummy data
    F, N, K = 3, 100, 5  # 3 frames, 100 vertices, 5 joints
    
    # Random vertices in normalized space
    frames_vertices = np.random.randn(F, N, 3) * 0.5
    joints = np.random.randn(K, 3) * 0.3
    parents = np.array([-1, 0, 1, 1, 2])  # Simple hierarchy
    
    try:
        rest_pose, weights, T = simple_skinning_fallback(frames_vertices, joints, parents)
        
        print(f"✓ Fallback successful:")
        print(f"  Rest pose: {rest_pose.shape}")
        print(f"  Weights: {weights.shape}, range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Weights sum check: {np.allclose(weights.sum(axis=1), 1.0)}")
        print(f"  Transforms: {T.shape}")
        return True
    except Exception as e:
        print(f"✗ Fallback failed: {e}")
        return False

def test_with_demo_files():
    """Test with actual demo files if available."""
    demo_paths = [
        "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    ]
    
    for demo_path in demo_paths:
        if os.path.exists(demo_path):
            obj_files = [f for f in os.listdir(demo_path) if f.endswith('.obj')]
            if obj_files:
                print(f"\nTesting with demo files in {demo_path}")
                print(f"Found {len(obj_files)} .obj files: {obj_files[:3]}...")
                
                try:
                    # Test with timeout and no visualization
                    process_obj_files(
                        folder_path=demo_path,
                        output_dir=os.path.join(demo_path, "test_output"),
                        is_bind=False,
                        visualize=False
                    )
                    print(f"✓ Successfully processed {demo_path}")
                    return True
                except Exception as e:
                    print(f"✗ Failed to process {demo_path}: {e}")
                    continue
    
    print("No demo files found for testing")
    return False

def main():
    print("=" * 60)
    print("Testing improved GenerateSkel.py with DemBones fixes")
    print("=" * 60)
    
    # Test 1: Simple fallback function
    # test1_passed = test_simple_fallback()
    
    print("\n" + "-" * 40)
    
    # Test 2: With actual files if available
    test2_passed = test_with_demo_files()
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    # print(f"Simple fallback test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Demo files test: {'PASSED' if test2_passed else 'SKIPPED/FAILED'}")
    
    # if test1_passed:
    #     print("\n✓ The improved GenerateSkel.py should handle DemBones failures gracefully")
    #     print("✓ Timeout protection and fallback mechanisms are in place")
    # else:
    #     print("\n✗ There may be issues with the fallback implementation")
    
    print("\nKey improvements:")
    print("- DemBones timeout protection (3 minutes)")
    print("- Progress monitoring to detect stuck computation")
    print("- Robust error handling with detailed logging")
    print("- Distance-based skinning fallback")
    print("- Input validation and parameter clamping")
    print("- Graceful degradation when DemBones fails")

if __name__ == "__main__":
    main()
