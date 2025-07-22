#!/usr/bin/env python3
"""
Test script for the VolumetricVideo Interpolation Pipeline
Creates simple test data and runs the complete pipeline
"""

import os
import numpy as np
import open3d as o3d
from pathlib import Path

def create_test_mesh_sequence(output_dir, num_frames=10):
    """
    Create a simple sequence of .obj files for testing.
    Creates a sphere that morphs into a cube over time.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_frames} test mesh frames in {output_dir}")
    
    for i in range(num_frames):
        t = i / (num_frames - 1)  # 0 to 1
        
        # Create sphere
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
        
        # Morph towards cube by moving vertices
        vertices = np.asarray(sphere.vertices)
        
        # Apply some deformation based on time
        scale_factor = 1.0 + 0.3 * np.sin(t * np.pi)
        deformation = np.array([
            1.0 + 0.2 * t * np.sin(vertices[:, 0] * 3),
            1.0 + 0.1 * t * np.cos(vertices[:, 1] * 4),
            1.0 + 0.15 * t * np.sin(vertices[:, 2] * 2)
        ]).T
        
        vertices *= deformation * scale_factor
        
        # Add some translation
        translation = np.array([0.1 * t, 0.05 * np.sin(t * 2 * np.pi), 0])
        vertices += translation
        
        # Update mesh
        sphere.vertices = o3d.utility.Vector3dVector(vertices)
        sphere.compute_vertex_normals()
        
        # Save mesh
        output_path = os.path.join(output_dir, f"frame_{i:03d}.obj")
        o3d.io.write_triangle_mesh(output_path, sphere)
        
        print(f"  Created frame_{i:03d}.obj: {len(vertices)} vertices")
    
    print(f"✓ Test sequence created with {num_frames} frames")
    return output_dir

def test_pipeline():
    """Test the complete pipeline with generated data."""
    
    # Create test data
    test_dir = "test_vv_data"
    create_test_mesh_sequence(test_dir, num_frames=10)
    
    # Import and run pipeline
    from pipeline_vv_interpolation import VolumetricVideoProcessor
    
    print("\n" + "="*50)
    print("TESTING VOLUMETRIC VIDEO INTERPOLATION PIPELINE")
    print("="*50)
    
    # Initialize processor
    processor = VolumetricVideoProcessor(test_dir, output_dir=os.path.join(test_dir, "pipeline_output"))
    
    try:
        # Step 1: Process all frames
        num_frames = processor.step1_process_all_frames(visualize=False)
        print(f"✓ Step 1 completed: {num_frames} frames processed")
        
        # Step 2: Detect rest pose
        rest_idx = processor.step2_detect_rest_pose()
        print(f"✓ Step 2 completed: Rest pose = frame {rest_idx}")
        
        # Step 3: Unify topology
        shape = processor.step3_unify_mesh_topology()
        print(f"✓ Step 3 completed: Unified shape = {shape}")
        
        # Step 4: Compute skinning
        processor.step4_compute_skinning()
        print(f"✓ Step 4 completed: Skinning computed")
        
        # Step 5: Generate interpolation
        from_frame, to_frame = 2, 7
        num_interp = 5
        interpolated = processor.step5_generate_interpolation(from_frame, to_frame, num_interp, visualize=False)
        print(f"✓ Step 5 completed: {len(interpolated)} interpolated frames generated")
        
        print(f"\n🎉 PIPELINE TEST SUCCESSFUL!")
        print(f"📁 Results saved in: {processor.output_dir}")
        print(f"📁 Test data in: {test_dir}")
        
        # Show file structure
        print(f"\nGenerated files:")
        for root, dirs, files in os.walk(processor.output_dir):
            level = root.replace(processor.output_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                print(f"{subindent}{file}")
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data():
    """Test with VVEditor data if available."""
    vv_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    if os.path.exists(vv_path):
        print(f"\n🔍 Found VVEditor data at {vv_path}")
        
        # Check for .obj files
        obj_files = list(Path(vv_path).glob("*.obj"))
        if obj_files:
            print(f"Found {len(obj_files)} .obj files")
            
            # Test with first few frames
            from pipeline_vv_interpolation import VolumetricVideoProcessor
            
            processor = VolumetricVideoProcessor(vv_path)
            
            # Quick test with limited frames (first 5)
            test_files = sorted(obj_files)[:5]
            print(f"Testing with first 5 frames: {[f.name for f in test_files]}")
            
            # This would require copying files to test directory or modifying processor
            print("For real VVEditor data, run:")
            print(f"python pipeline_vv_interpolation.py \"{vv_path}\" --from_frame 0 --to_frame 2 --num_interp 3")
        else:
            print("No .obj files found in VVEditor directory")
    else:
        print(f"VVEditor path not found: {vv_path}")

if __name__ == "__main__":
    # Test with generated data
    success = test_pipeline()
    
    if success:
        # Test with real data if available
        test_with_real_data()
        
        print(f"\n" + "="*50)
        print("USAGE EXAMPLES:")
        print("="*50)
        print("# Test with generated data:")
        print("python test_pipeline.py")
        print("")
        print("# Use with your VV data:")
        print("python pipeline_vv_interpolation.py /path/to/vv/frames --from_frame 5 --to_frame 15 --num_interp 10")
        print("")
        print("# Skip processing if already done:")
        print("python pipeline_vv_interpolation.py /path/to/vv/frames --skip_processing --from_frame 0 --to_frame 5 --num_interp 8")
        print("="*50)
