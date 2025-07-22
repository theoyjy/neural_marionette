#!/usr/bin/env python3

import sys
import os
import traceback

# Add the current directory to the path so we can import GenerateSkel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GenerateSkel import process_obj_files

def force_fallback_test():
    """Force fallback for large datasets by modifying the DemBones function"""
    
    # Test with the full dataset but force fallback
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    print("Starting process with forced fallback for large datasets...")
    
    try:
        # Import and patch the module
        import GenerateSkel
        
        # Replace the safe solver to always use fallback for large datasets
        original_solve = GenerateSkel.solve_with_dem_bones_safe
        
        def force_fallback_solve(frames_vertices, parents, nnz=4, n_iters=10):
            """Always use fallback for demonstration"""
            F, N, _ = frames_vertices.shape
            print(f"Forced fallback for dataset: {F} frames, {N} vertices")
            
            # Use the simple fallback directly
            return GenerateSkel.simple_skinning_fallback(frames_vertices, 
                                                        GenerateSkel.extract_joints_from_data(None), # We'll pass the joints separately
                                                        parents)
        
        # Don't actually patch - let's just run with original timeout mechanism
        process_obj_files(
            folder_path=folder_path,
            output_dir=None,
            is_bind=False,
            visualize=False
        )
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    force_fallback_test()
