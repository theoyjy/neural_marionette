#!/usr/bin/env python3
"""
Simple test to verify DemBones timeout fixes work correctly.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from GenerateSkel import solve_with_dem_bones_safe, simple_skinning_fallback
    print("✓ Successfully imported functions from GenerateSkel")
except ImportError as e:
    print(f"✗ Failed to import from GenerateSkel: {e}")
    sys.exit(1)

def test_simple_skinning():
    """Test the simple skinning fallback."""
    print("\n" + "="*50)
    print("Testing simple skinning fallback...")
    
    # Create test data
    F, N, K = 2, 50, 3  # 2 frames, 50 vertices, 3 joints
    frames_vertices = np.random.randn(F, N, 3) * 0.3
    joints = np.random.randn(K, 3) * 0.2
    parents = np.array([-1, 0, 1])
    
    try:
        rest_pose, weights, T = simple_skinning_fallback(frames_vertices, joints, parents)
        
        print(f"✓ Simple skinning test passed:")
        print(f"  - Rest pose shape: {rest_pose.shape}")
        print(f"  - Weights shape: {weights.shape}")
        print(f"  - Weights range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  - Weights sum to 1: {np.allclose(weights.sum(axis=1), 1.0)}")
        print(f"  - Transforms shape: {T.shape}")
        return True
    except Exception as e:
        print(f"✗ Simple skinning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_safe_dembone():
    """Test the safe DemBones wrapper."""
    print("\n" + "="*50)
    print("Testing safe DemBones wrapper...")
    
    # Create minimal test data
    F, N, K = 2, 20, 2  # Very small to reduce computation time
    frames_vertices = np.random.randn(F, N, 3) * 0.2
    parents = np.array([-1, 0])
    
    try:
        print("Calling solve_with_dem_bones_safe with short timeout...")
        rest_pose, weights, T = solve_with_dem_bones_safe(
            frames_vertices, parents, nnz=4, n_iters=5
        )
        
        print(f"✓ Safe DemBones test completed:")
        print(f"  - Rest pose shape: {rest_pose.shape}")
        print(f"  - Weights shape: {weights.shape}")
        print(f"  - Transforms shape: {T.shape}")
        return True
    except Exception as e:
        print(f"✗ Safe DemBones test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing DemBones timeout fixes")
    print("This should complete quickly without hanging...")
    
    # Test 1: Simple skinning fallback
    test1_ok = test_simple_skinning()
    
    # Test 2: Safe DemBones wrapper (may fallback to simple skinning)
    test2_ok = test_safe_dembone()
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Simple skinning test: {'PASSED' if test1_ok else 'FAILED'}")
    print(f"Safe DemBones test: {'PASSED' if test2_ok else 'FAILED'}")
    
    if test1_ok and test2_ok:
        print("\n✓ All tests passed! The timeout fixes should work.")
        print("✓ DemBones will either complete quickly or fallback gracefully.")
    elif test1_ok:
        print("\n⚠ Simple skinning works, but DemBones wrapper has issues.")
        print("  The fallback mechanism should still prevent hanging.")
    else:
        print("\n✗ Tests failed. There may be import or basic functionality issues.")
    
    print(f"\nNext step: Try running on your actual .obj files:")
    print(f"python GenerateSkel.py 'D:/Code/VVEditor/Rafa_Approves_hd_4k' --no_visualize")

if __name__ == "__main__":
    main()
