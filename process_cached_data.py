#!/usr/bin/env python3

import os
import pickle
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.distance import cdist

def simple_distance_skinning(frames_vertices, joints, parents, nnz=4):
    """
    Simple distance-based skinning as fallback
    """
    F, N, _ = frames_vertices.shape
    K = len(joints)
    
    print(f"Computing simple distance-based skinning: {N} vertices, {K} joints")
    
    # Use the first frame as rest pose
    rest_vertices = frames_vertices[0]  # Shape: (N, 3)
    
    # Compute distances from each vertex to each joint
    distances = cdist(rest_vertices, joints)  # Shape: (N, K)
    
    # Convert distances to weights (closer = higher weight)
    # Use inverse distance with small epsilon to avoid division by zero
    epsilon = 1e-6
    weights = 1.0 / (distances + epsilon)
    
    # Keep only the nnz closest joints for each vertex
    skinning_weights = np.zeros((N, K))
    for i in range(N):
        # Get indices of the nnz closest joints
        closest_joints = np.argsort(distances[i])[:nnz]
        vertex_weights = weights[i, closest_joints]
        
        # Normalize weights to sum to 1
        vertex_weights = vertex_weights / np.sum(vertex_weights)
        skinning_weights[i, closest_joints] = vertex_weights
    
    print(f"Simple skinning complete: weights range [{skinning_weights.min():.4f}, {skinning_weights.max():.4f}]")
    print(f"Average weights per vertex: {np.sum(skinning_weights > 0, axis=1).mean():.2f}")
    
    # Create identity transforms for all frames
    transforms = np.tile(np.eye(4), (F, K, 1, 1))
    
    return rest_vertices, skinning_weights, transforms

def process_cached_data():
    """Process the cached data to create skeleton and skinning results"""
    
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = os.path.join(folder_path, "generated_skeletons")
    
    print("Processing cached data...")
    
    # Load all cached data
    mesh_data = {}
    obj_files = [f for f in os.listdir(folder_path) if f.endswith('.obj')]
    obj_files.sort()
    
    print(f"Found {len(obj_files)} .obj files")
    
    for obj_file in obj_files:
        base_name = obj_file.replace('.obj', '')
        data_path = os.path.join(output_dir, f'{base_name}_data.pkl')
        
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                mesh_data[base_name] = data
                print(f"Loaded {base_name}: {len(data['mesh_vertices'])} vertices")
    
    if not mesh_data:
        print("No cached data found!")
        return
    
    # Prepare data for batch processing
    print("\nPreparing data for DemBones processing...")
    
    # Find the mesh with the most vertices to use as template
    template_name = max(mesh_data.keys(), key=lambda k: len(mesh_data[k]['mesh_vertices']))
    template_data = mesh_data[template_name]
    template_vertices_count = len(template_data['mesh_vertices'])
    
    print(f"Using {template_name} as template with {template_vertices_count} vertices")
    
    # Collect all mesh data
    all_vertices = []
    all_joints = []
    all_parents = []
    
    for i, (name, data) in enumerate(mesh_data.items()):
        print(f"Processing {name}: {len(data['mesh_vertices'])} vertices -> {template_vertices_count} vertices")
        
        # Remap vertices to template size (simplified)
        vertices = np.array(data['mesh_vertices'])
        if len(vertices) != template_vertices_count:
            # Simple resampling - in practice you'd want proper correspondence
            indices = np.linspace(0, len(vertices)-1, template_vertices_count, dtype=int)
            vertices = vertices[indices]
        
        all_vertices.append(vertices)
        all_joints.append(np.array(data['joints']))
        all_parents.append(np.array(data['parents']))
    
    # Convert to numpy arrays
    frames_vertices = np.array(all_vertices)  # Shape: (F, N, 3)
    joints = all_joints[0]  # Use joints from first frame
    parents = all_parents[0]  # Use parents from first frame
    
    print(f"Prepared data: {frames_vertices.shape[0]} frames, {frames_vertices.shape[1]} vertices, {len(joints)} joints")
    
    # Run simple distance-based skinning
    rest_vertices, skinning_weights, transforms = simple_distance_skinning(
        frames_vertices, joints, parents, nnz=4
    )
    
    print("Skinning computation completed!")
    
    # Save results
    results = {
        'rest_pose': rest_vertices,
        'skinning_weights': skinning_weights,
        'transforms': transforms,
        'joints': joints,
        'parents': parents,
        'frame_names': list(mesh_data.keys())
    }
    
    result_path = os.path.join(output_dir, 'dembone_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Saved results to {result_path}")
    
    # Save individual skeleton files
    for i, (name, data) in enumerate(mesh_data.items()):
        skeleton_data = {
            'joints': joints,
            'parents': parents,
            'skinning_weights': skinning_weights,
            'rest_pose': rest_vertices
        }
        
        skeleton_path = os.path.join(output_dir, f'{name}_skeleton.pkl')
        with open(skeleton_path, 'wb') as f:
            pickle.dump(skeleton_data, f)
        
        # Save rest pose mesh
        rest_v_world = (rest_vertices + 1) * 0.5 * data['blen'] + data['bmin']
        mesh_rest = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(rest_v_world),
            triangles=o3d.utility.Vector3iVector(data['mesh_triangles'])
        )
        mesh_rest.compute_vertex_normals()
        
        rest_mesh_path = os.path.join(output_dir, f'{name}_rest_pose.obj')
        o3d.io.write_triangle_mesh(rest_mesh_path, mesh_rest)
        
        print(f"Saved {name} skeleton and rest pose")
    
    print("\n--- Processing complete ---")
    print(f"Results saved in: {output_dir}")
    print(f"- Individual mesh data: *_data.pkl")
    print(f"- Individual skeleton results: *_skeleton.pkl") 
    print(f"- Individual rest pose meshes: *_rest_pose.obj")
    print(f"- Combined DemBones results: dembone_results.pkl")

if __name__ == "__main__":
    process_cached_data()
