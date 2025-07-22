import os
import sys
sys.path.append('.')

from mesh_interpolation import load_mesh_data, interpolate_meshes

# Test the interpolation system
data_folder = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons"

print("Testing mesh interpolation system...")

try:
    # Load the data to see available frames
    dembone_results, mesh_data = load_mesh_data(data_folder)
    
    # Show available frames
    frame_names = list(mesh_data.keys())
    print(f"Available frames: {len(frame_names)}")
    print("First 10 frames:", frame_names[:10])
    print("Last 10 frames:", frame_names[-10:])
    
    # Test interpolation between first and 10th frame
    frame_a = frame_names[0]
    frame_b = frame_names[9] if len(frame_names) > 9 else frame_names[-1]
    
    print(f"\nTesting interpolation between:")
    print(f"  Frame A: {frame_a}")
    print(f"  Frame B: {frame_b}")
    
    # Test direct interpolation
    print("\n--- Testing Direct Vertex Interpolation ---")
    meshes_direct, output_dir_direct = interpolate_meshes(
        data_folder, 
        frame_a, 
        frame_b,
        num_steps=5,
        method='direct'
    )
    
    # Test skeletal interpolation
    print("\n--- Testing Skeletal Interpolation ---")
    meshes_skeleton, output_dir_skeleton = interpolate_meshes(
        data_folder, 
        frame_a, 
        frame_b,
        num_steps=5,
        method='skeleton'
    )
    
    print(f"\n✅ Interpolation test completed successfully!")
    print(f"Direct interpolation results: {output_dir_direct}")
    print(f"Skeletal interpolation results: {output_dir_skeleton}")
    
except Exception as e:
    print(f"❌ Error testing interpolation: {e}")
    import traceback
    traceback.print_exc()
