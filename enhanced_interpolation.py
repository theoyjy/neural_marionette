import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse

def load_skeleton_driven_data(data_folder):
    """Load skeleton-driven mesh data and results."""
    
    # Try to load skeleton-driven results first
    skeleton_driven_path = os.path.join(data_folder, 'skeleton_driven_results.pkl')
    nearest_neighbor_path = os.path.join(data_folder, 'nearest_neighbor_results.pkl')
    original_dembone_path = os.path.join(data_folder, 'dembone_results.pkl')
    
    available_methods = []
    results = {}
    
    # Load skeleton-driven results if available
    if os.path.exists(skeleton_driven_path):
        with open(skeleton_driven_path, 'rb') as f:
            results['skeleton_driven'] = pickle.load(f)
        available_methods.append('skeleton_driven')
        print(f"✅ Loaded skeleton-driven results: {results['skeleton_driven']['unified_vertices'].shape}")
    
    # Load nearest neighbor results for comparison
    if os.path.exists(nearest_neighbor_path):
        with open(nearest_neighbor_path, 'rb') as f:
            results['nearest_neighbor'] = pickle.load(f)
        available_methods.append('nearest_neighbor')
        print(f"📊 Loaded nearest neighbor results: {results['nearest_neighbor']['unified_vertices'].shape}")
    
    # Load original dembone results as fallback
    if os.path.exists(original_dembone_path):
        with open(original_dembone_path, 'rb') as f:
            original_results = pickle.load(f)
        
        # Convert to unified format
        frame_names = original_results['frame_names']
        rest_pose = original_results['rest_pose']
        
        # Create unified vertices from rest pose (placeholder)
        unified_vertices = np.tile(rest_pose[np.newaxis, :, :], (len(frame_names), 1, 1))
        
        results['original_dembone'] = {
            'rest_pose': rest_pose,
            'skinning_weights': original_results['skinning_weights'],
            'transforms': original_results['transforms'],
            'joints': original_results['joints'],
            'parents': original_results['parents'],
            'frame_names': frame_names,
            'unified_vertices': unified_vertices,
            'method': 'original_dembone'
        }
        available_methods.append('original_dembone')
        print(f"🔄 Loaded original DemBones results as fallback")
    
    if not available_methods:
        raise FileNotFoundError(f"No valid results found in {data_folder}")
    
    print(f"📋 Available methods: {available_methods}")
    return results, available_methods

def load_individual_mesh_data(data_folder):
    """Load individual mesh data files for metadata."""
    data_files = [f for f in os.listdir(data_folder) if f.endswith('_data.pkl')]
    data_files.sort()
    
    mesh_data = {}
    for data_file in data_files:
        frame_name = data_file.replace('_data.pkl', '')
        with open(os.path.join(data_folder, data_file), 'rb') as f:
            mesh_data[frame_name] = pickle.load(f)
    
    return mesh_data

def interpolate_skeleton_driven(frame_a_idx, frame_b_idx, t, unified_results):
    """
    Interpolate between two frames using skeleton-driven unified meshes.
    This should provide much better quality than the original method.
    """
    unified_vertices = unified_results['unified_vertices']
    joints = unified_results['joints']
    parents = unified_results['parents']
    
    # Get the unified mesh vertices for both frames
    vertices_a = unified_vertices[frame_a_idx]  # (N, 3)
    vertices_b = unified_vertices[frame_b_idx]  # (N, 3)
    
    # Linear interpolation between unified vertices
    # Since both meshes now have the same topology and are properly aligned,
    # this should produce much better results
    interpolated_vertices = (1 - t) * vertices_a + t * vertices_b
    
    # Optionally interpolate joint positions as well
    # Note: transforms shape is (F, K, 4, 4)
    if 'transforms' in unified_results:
        transforms = unified_results['transforms']
        if len(transforms.shape) == 4:  # (F, K, 4, 4)
            # Extract joint positions from transforms (simplified)
            joints_a = joints.copy()  # Use template joints as base
            joints_b = joints.copy()
            
            # Apply bone transforms to get frame-specific joint positions
            for k in range(len(joints)):
                if frame_a_idx < transforms.shape[0] and k < transforms.shape[1]:
                    translation_a = transforms[frame_a_idx, k, :3, 3]
                    joints_a[k] += translation_a
                
                if frame_b_idx < transforms.shape[0] and k < transforms.shape[1]:
                    translation_b = transforms[frame_b_idx, k, :3, 3]
                    joints_b[k] += translation_b
            
            interpolated_joints = (1 - t) * joints_a + t * joints_b
        else:
            interpolated_joints = joints
    else:
        interpolated_joints = joints
    
    return interpolated_vertices, interpolated_joints

def interpolate_vertices_direct(verts_a, verts_b, t):
    """Direct linear interpolation between vertex positions (original method)."""
    # Handle different vertex counts by using nearest neighbor mapping
    if verts_a.shape[0] != verts_b.shape[0]:
        # Find correspondence using nearest neighbors
        tree = cKDTree(verts_b)
        _, indices = tree.query(verts_a, k=1)
        verts_b_mapped = verts_b[indices]
    else:
        verts_b_mapped = verts_b
    
    return (1 - t) * verts_a + t * verts_b_mapped

def create_interpolated_mesh(vertices, triangles, scale_params):
    """Create an Open3D mesh from interpolated vertices."""
    # Convert from normalized space back to world space
    bmin, blen = scale_params['bmin'], scale_params['blen']
    vertices_world = (vertices + 1) * 0.5 * blen + bmin
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices_world),
        triangles=o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    return mesh

def interpolate_meshes_enhanced(data_folder, frame_a, frame_b, num_steps=10, 
                               unification_method='skeleton_driven', output_dir=None):
    """
    Enhanced mesh interpolation using skeleton-driven unified meshes.
    
    Args:
        data_folder: Path to generated_skeletons folder
        frame_a, frame_b: Frame names or indices
        num_steps: Number of interpolation steps
        unification_method: 'skeleton_driven', 'nearest_neighbor', or 'original_dembone'
        output_dir: Where to save interpolated meshes
    """
    
    # Load unified mesh data
    results, available_methods = load_skeleton_driven_data(data_folder)
    
    # Check if requested method is available
    if unification_method not in available_methods:
        print(f"⚠️  Requested method '{unification_method}' not available.")
        print(f"📋 Available methods: {available_methods}")
        unification_method = available_methods[0]
        print(f"🔄 Using '{unification_method}' instead.")
    
    unified_results = results[unification_method]
    frame_names = unified_results['frame_names']
    
    # Convert frame names to indices
    if isinstance(frame_a, str) and frame_a.isdigit():
        # It's a frame number (1-based)
        frame_a_idx = int(frame_a) - 1
        if 0 <= frame_a_idx < len(frame_names):
            frame_a = frame_names[frame_a_idx]
        else:
            raise ValueError(f"Frame number {frame_a} out of range (1-{len(frame_names)})")
    elif isinstance(frame_a, str):
        # It's a frame name
        if frame_a not in frame_names:
            raise ValueError(f"Frame '{frame_a}' not found in {len(frame_names)} frames")
        frame_a_idx = frame_names.index(frame_a)
    else:
        # It's already a number
        frame_a_idx = int(frame_a) - 1  # Convert 1-based to 0-based
        frame_a = frame_names[frame_a_idx]
    
    if isinstance(frame_b, str) and frame_b.isdigit():
        # It's a frame number (1-based)
        frame_b_idx = int(frame_b) - 1
        if 0 <= frame_b_idx < len(frame_names):
            frame_b = frame_names[frame_b_idx]
        else:
            raise ValueError(f"Frame number {frame_b} out of range (1-{len(frame_names)})")
    elif isinstance(frame_b, str):
        # It's a frame name
        if frame_b not in frame_names:
            raise ValueError(f"Frame '{frame_b}' not found in {len(frame_names)} frames")
        frame_b_idx = frame_names.index(frame_b)
    else:
        # It's already a number
        frame_b_idx = int(frame_b) - 1  # Convert 1-based to 0-based
        frame_b = frame_names[frame_b_idx]
    
    # Load individual mesh data for scaling parameters
    mesh_data = load_individual_mesh_data(data_folder)
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(data_folder, f'interpolation_{frame_a}_to_{frame_b}_{unification_method}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎯 Creating {num_steps} interpolated meshes using {unification_method} method")
    print(f"   From: {frame_a} (index {frame_a_idx})")
    print(f"   To: {frame_b} (index {frame_b_idx})")
    print(f"   Output: {output_dir}")
    
    # Get mesh topology (triangles)
    template_frame_name = frame_names[0]
    triangles = mesh_data[template_frame_name]['mesh_triangles']
    scale_params = mesh_data[template_frame_name]  # Use first frame's scale
    
    # Generate interpolated meshes
    interpolated_meshes = []
    
    for i in range(num_steps + 1):
        t = i / num_steps  # 0.0 to 1.0
        
        if unification_method in ['skeleton_driven', 'nearest_neighbor']:
            # Use the new skeleton-driven interpolation
            interp_vertices, interp_joints = interpolate_skeleton_driven(
                frame_a_idx, frame_b_idx, t, unified_results
            )
        else:
            # Fallback to original method
            vertices_a = unified_results['unified_vertices'][frame_a_idx]
            vertices_b = unified_results['unified_vertices'][frame_b_idx]
            interp_vertices = interpolate_vertices_direct(vertices_a, vertices_b, t)
            interp_joints = unified_results['joints']
        
        # Create mesh
        mesh = create_interpolated_mesh(interp_vertices, triangles, scale_params)
        interpolated_meshes.append(mesh)
        
        # Save mesh
        filename = f'interpolated_{i:03d}_t{t:.3f}_{unification_method}.obj'
        filepath = os.path.join(output_dir, filename)
        o3d.io.write_triangle_mesh(filepath, mesh)
        
        print(f"💾 Saved: {filename} (t={t:.3f})")
    
    # Create a summary file
    summary = {
        'unification_method': unification_method,
        'frame_a': frame_a,
        'frame_b': frame_b,
        'frame_a_idx': frame_a_idx,
        'frame_b_idx': frame_b_idx,
        'num_steps': num_steps,
        'total_meshes': len(interpolated_meshes),
        'available_methods': available_methods,
        'filenames': [f'interpolated_{i:03d}_t{i/num_steps:.3f}_{unification_method}.obj' 
                     for i in range(num_steps + 1)]
    }
    
    summary_path = os.path.join(output_dir, 'interpolation_info.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n✅ Enhanced interpolation complete!")
    print(f"📁 Generated {len(interpolated_meshes)} meshes using {unification_method} method")
    print(f"🎬 Results saved in: {output_dir}")
    
    return interpolated_meshes, output_dir

def compare_interpolation_methods(data_folder, frame_a, frame_b, num_steps=5):
    """Compare interpolation quality between different unification methods."""
    print("🔍 Comparing interpolation methods...")
    
    results, available_methods = load_skeleton_driven_data(data_folder)
    
    comparison_results = {}
    
    for method in available_methods:
        print(f"\n📊 Testing {method} method...")
        try:
            meshes, output_dir = interpolate_meshes_enhanced(
                data_folder, frame_a, frame_b, num_steps, method
            )
            comparison_results[method] = {
                'success': True,
                'output_dir': output_dir,
                'num_meshes': len(meshes)
            }
            print(f"✅ {method}: {len(meshes)} meshes generated")
        except Exception as e:
            comparison_results[method] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ {method}: Failed - {e}")
    
    print(f"\n📋 Comparison complete!")
    for method, result in comparison_results.items():
        if result['success']:
            print(f"✅ {method}: Success ({result['num_meshes']} meshes)")
        else:
            print(f"❌ {method}: Failed - {result['error']}")
    
    return comparison_results

def main():
    parser = argparse.ArgumentParser(description="Enhanced mesh interpolation with skeleton-driven unification")
    parser.add_argument('frame_a', nargs='?', help='First frame name or number')
    parser.add_argument('frame_b', nargs='?', help='Second frame name or number')
    parser.add_argument('--data_folder', 
                       default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
                       help='Path to generated_skeletons folder')
    parser.add_argument('--steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--method', choices=['skeleton_driven', 'nearest_neighbor', 'original_dembone'], 
                       default='skeleton_driven', help='Unification method')
    parser.add_argument('--output_dir', help='Output directory for interpolated meshes')
    parser.add_argument('--compare', action='store_true', help='Compare all available methods')
    parser.add_argument('--list', action='store_true', help='List available frames and methods')
    
    args = parser.parse_args()
    
    # List available data
    if args.list:
        try:
            results, available_methods = load_skeleton_driven_data(args.data_folder)
            print(f"📋 Available unification methods: {available_methods}")
            
            # Show frame information
            if available_methods:
                frame_names = results[available_methods[0]]['frame_names']
                print(f"\n📋 Available frames ({len(frame_names)} total):")
                for i, name in enumerate(frame_names[:10]):  # Show first 10
                    print(f"  {i+1:3d}: {name}")
                if len(frame_names) > 10:
                    print(f"  ... and {len(frame_names)-10} more frames")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
        return
    
    # Compare methods
    if args.compare:
        if not args.frame_a or not args.frame_b:
            print("❌ Please specify frame_a and frame_b for comparison")
            return
        
        compare_interpolation_methods(args.data_folder, args.frame_a, args.frame_b, args.steps)
        return
    
    # Regular interpolation
    if not args.frame_a or not args.frame_b:
        print("❌ Please specify both frame_a and frame_b")
        print("💡 Use --list to see available frames and methods")
        return
    
    try:
        meshes, output_dir = interpolate_meshes_enhanced(
            args.data_folder, args.frame_a, args.frame_b, 
            args.steps, args.method, args.output_dir
        )
        print(f"🎉 Success! Generated {len(meshes)} interpolated meshes")
        print(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during interpolation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
