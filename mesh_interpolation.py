import os
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import argparse

def load_mesh_data(data_folder):
    """Load all mesh data and DemBones results."""
    
    # Load the combined results
    dembone_path = os.path.join(data_folder, 'dembone_results.pkl')
    if not os.path.exists(dembone_path):
        raise FileNotFoundError(f"DemBones results not found: {dembone_path}")
    
    with open(dembone_path, 'rb') as f:
        dembone_results = pickle.load(f)
    
    # Load individual mesh data files
    data_files = [f for f in os.listdir(data_folder) if f.endswith('_data.pkl')]
    data_files.sort()  # Ensure consistent ordering
    
    mesh_data = {}
    for data_file in data_files:
        frame_name = data_file.replace('_data.pkl', '')
        with open(os.path.join(data_folder, data_file), 'rb') as f:
            mesh_data[frame_name] = pickle.load(f)
    
    print(f"Loaded {len(mesh_data)} mesh data files")
    print(f"Available frames: {list(mesh_data.keys())[:5]}...")  # Show first 5
    
    return dembone_results, mesh_data

def interpolate_joints(joints_a, joints_b, t):
    """Linear interpolation between two joint configurations."""
    return (1 - t) * joints_a + t * joints_b

def interpolate_vertices_direct(verts_a, verts_b, t):
    """Direct linear interpolation between vertex positions."""
    # Handle different vertex counts by using nearest neighbor mapping
    if verts_a.shape[0] != verts_b.shape[0]:
        # Find correspondence using nearest neighbors
        tree = cKDTree(verts_b)
        _, indices = tree.query(verts_a, k=1)
        verts_b_mapped = verts_b[indices]
    else:
        verts_b_mapped = verts_b
    
    return (1 - t) * verts_a + t * verts_b_mapped

def interpolate_via_skeleton(frame_a_name, frame_b_name, t, dembone_results, mesh_data):
    """
    Interpolate between two meshes using skeletal animation.
    This creates a more realistic interpolation that respects the bone structure.
    """
    # Get the data for both frames
    data_a = mesh_data[frame_a_name]
    data_b = mesh_data[frame_b_name]
    
    # Get joint positions (already in normalized space)
    joints_a = data_a['joints']  # (K, 3)
    joints_b = data_b['joints']  # (K, 3)
    
    # Interpolate joint positions
    joints_interp = interpolate_joints(joints_a, joints_b, t)
    
    # Get rest pose and skinning weights from DemBones results
    rest_pose = dembone_results['rest_pose']  # (N, 3) normalized
    skinning_weights = dembone_results['skinning_weights']  # (N, K)
    parents = dembone_results['parents']  # (K,)
    
    # Compute bone transformations for interpolated joints
    # This is a simplified approach - for more accuracy, you'd want to interpolate rotations
    rest_joints_a = joints_a
    rest_joints_b = joints_b
    rest_joints_interp = joints_interp
    
    # Apply Linear Blend Skinning with interpolated joints
    deformed_vertices = np.zeros_like(rest_pose)
    
    for k in range(len(parents)):
        # Simple translation-based deformation
        # For each bone, compute translation from rest to current position
        if k < len(joints_a) and k < len(joints_b):
            bone_translation = rest_joints_interp[k] - rest_joints_a[k]
            
            # Apply weighted translation to vertices influenced by this bone
            for v in range(len(rest_pose)):
                weight = skinning_weights[v, k]
                deformed_vertices[v] += weight * bone_translation
    
    # Add the base rest pose
    final_vertices = rest_pose + deformed_vertices
    
    return final_vertices, joints_interp

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

def interpolate_meshes(data_folder, frame_a, frame_b, num_steps=10, method='direct', output_dir=None):
    """
    Create interpolated meshes between two frames.
    
    Args:
        data_folder: Path to generated_skeletons folder
        frame_a, frame_b: Frame names (without _data.pkl suffix)
        num_steps: Number of interpolation steps
        method: 'direct' for vertex interpolation, 'skeleton' for skeletal interpolation
        output_dir: Where to save interpolated meshes
    """
    
    # Load data
    dembone_results, mesh_data = load_mesh_data(data_folder)
    
    if frame_a not in mesh_data:
        print(f"Available frames: {list(mesh_data.keys())}")
        raise ValueError(f"Frame {frame_a} not found")
    if frame_b not in mesh_data:
        raise ValueError(f"Frame {frame_b} not found")
    
    data_a = mesh_data[frame_a]
    data_b = mesh_data[frame_b]
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(data_folder, f'interpolation_{frame_a}_to_{frame_b}_{method}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_steps} interpolated meshes from {frame_a} to {frame_b}")
    print(f"Method: {method}")
    print(f"Output directory: {output_dir}")
    
    # Get triangles (use the first mesh's triangles since all are mapped to same topology)
    first_frame_name = list(mesh_data.keys())[0]
    triangles = mesh_data[first_frame_name]['mesh_triangles']
    
    # Generate interpolated meshes
    interpolated_meshes = []
    
    for i in range(num_steps + 1):
        t = i / num_steps  # 0.0 to 1.0
        
        if method == 'direct':
            # Direct vertex interpolation
            verts_a = data_a['pts_norm']
            verts_b = data_b['pts_norm']
            interp_vertices = interpolate_vertices_direct(verts_a, verts_b, t)
            scale_params = data_a  # Use frame A's scale parameters
            
        elif method == 'skeleton':
            # Skeletal interpolation
            interp_vertices, interp_joints = interpolate_via_skeleton(
                frame_a, frame_b, t, dembone_results, mesh_data
            )
            scale_params = data_a  # Use frame A's scale parameters
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'direct' or 'skeleton'")
        
        # Create mesh
        mesh = create_interpolated_mesh(interp_vertices, triangles, scale_params)
        interpolated_meshes.append(mesh)
        
        # Save mesh
        filename = f'interpolated_{i:03d}_t{t:.3f}.obj'
        filepath = os.path.join(output_dir, filename)
        o3d.io.write_triangle_mesh(filepath, mesh)
        
        print(f"Saved: {filename} (t={t:.3f})")
    
    # Create a summary file
    summary = {
        'method': method,
        'frame_a': frame_a,
        'frame_b': frame_b,
        'num_steps': num_steps,
        'total_meshes': len(interpolated_meshes),
        'filenames': [f'interpolated_{i:03d}_t{i/num_steps:.3f}.obj' for i in range(num_steps + 1)]
    }
    
    summary_path = os.path.join(output_dir, 'interpolation_info.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nInterpolation complete!")
    print(f"Generated {len(interpolated_meshes)} meshes in {output_dir}")
    
    return interpolated_meshes, output_dir

def visualize_interpolation(interpolated_meshes, delay=0.5):
    """Visualize the interpolated sequence."""
    print("Visualizing interpolation sequence...")
    print("Press 'q' to quit, space to pause/resume")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Mesh Interpolation', width=800, height=600)
    
    # Add first mesh
    current_mesh = interpolated_meshes[0]
    vis.add_geometry(current_mesh)
    
    import time
    paused = False
    
    for i, mesh in enumerate(interpolated_meshes):
        if not paused:
            # Update mesh
            vis.remove_geometry(current_mesh)
            vis.add_geometry(mesh)
            current_mesh = mesh
            
            # Update title
            vis.get_render_option().point_size = 1.0
            print(f"\rFrame {i+1}/{len(interpolated_meshes)}", end='', flush=True)
        
        vis.poll_events()
        vis.update_renderer()
        
        if not paused:
            time.sleep(delay)
    
    print("\nVisualization complete!")
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description="Interpolate between two meshes using Neural Marionette data")
    parser.add_argument('data_folder', help='Path to generated_skeletons folder')
    parser.add_argument('frame_a', help='First frame name (without _data.pkl)')
    parser.add_argument('frame_b', help='Second frame name (without _data.pkl)')
    parser.add_argument('--steps', type=int, default=10, help='Number of interpolation steps')
    parser.add_argument('--method', choices=['direct', 'skeleton'], default='direct', 
                       help='Interpolation method')
    parser.add_argument('--output_dir', help='Output directory for interpolated meshes')
    parser.add_argument('--visualize', action='store_true', help='Visualize the interpolation')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between frames in visualization')
    
    args = parser.parse_args()
    
    # Create interpolated meshes
    meshes, output_dir = interpolate_meshes(
        args.data_folder, 
        args.frame_a, 
        args.frame_b,
        args.steps,
        args.method,
        args.output_dir
    )
    
    # Optionally visualize
    if args.visualize:
        visualize_interpolation(meshes, args.delay)

if __name__ == "__main__":
    main()
