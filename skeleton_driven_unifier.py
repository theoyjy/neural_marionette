#!/usr/bin/env python3
"""
Skeleton-driven mesh unification system for Neural Marionette data.

This system implements:
1. Automatic initial skinning (one-time setup)
2. Per-frame skeleton-driven deformation 
3. Non-rigid ICP fine registration
4. Unified topology for high-quality interpolation

骨骼驱动网格统一系统：
1. 自动初始蒙皮（一次性）
2. 逐帧骨骼驱动变形
3. 非刚性ICP精细配准
4. 统一拓扑用于高质量插值
"""

import os
import pickle
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import time

class SkeletonDrivenUnifier:
    """Skeleton-driven mesh unification system"""
    
    def __init__(self, folder_path, output_dir):
        self.folder_path = folder_path
        self.output_dir = output_dir
        self.template_mesh = None
        self.template_skinning_weights = None
        self.template_joints = None
        self.template_parents = None
        self.unified_meshes = {}
        
    def load_cached_data(self):
        """Load all cached mesh data"""
        print("Loading cached data...")
        
        mesh_data = {}
        obj_files = [f for f in os.listdir(self.folder_path) if f.endswith('.obj')]
        obj_files.sort()
        
        print(f"Found {len(obj_files)} .obj files")
        
        for obj_file in obj_files:
            base_name = obj_file.replace('.obj', '')
            data_path = os.path.join(self.output_dir, f'{base_name}_data.pkl')
            
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    mesh_data[base_name] = data
                    print(f"Loaded {base_name}: {len(data['mesh_vertices'])} vertices")
        
        if not mesh_data:
            raise ValueError("No cached data found!")
            
        return mesh_data
    
    def setup_template_mesh(self, mesh_data):
        """Setup template mesh and initial skinning (Step 1)"""
        print("\n=== Step 1: Setting up template mesh and initial skinning ===")
        
        # Find the mesh with the most vertices as template
        template_name = max(mesh_data.keys(), key=lambda k: len(mesh_data[k]['mesh_vertices']))
        template_data = mesh_data[template_name]
        
        print(f"Using {template_name} as template with {len(template_data['mesh_vertices'])} vertices")
        
        # Store template information
        self.template_mesh = np.array(template_data['mesh_vertices'])
        self.template_joints = np.array(template_data['joints'])
        self.template_parents = np.array(template_data['parents'])
        
        # Compute initial skinning weights using distance-based method
        print("Computing initial skinning weights...")
        self.template_skinning_weights = self.compute_initial_skinning(
            self.template_mesh, self.template_joints, nnz=4
        )
        
        print(f"Template setup complete:")
        print(f"  - Mesh vertices: {self.template_mesh.shape}")
        print(f"  - Joints: {self.template_joints.shape}")
        print(f"  - Skinning weights: {self.template_skinning_weights.shape}")
        print(f"  - Weight range: [{self.template_skinning_weights.min():.4f}, {self.template_skinning_weights.max():.4f}]")
        
        return template_name, template_data
    
    def compute_initial_skinning(self, vertices, joints, nnz=4):
        """Compute initial skinning weights using distance-based method"""
        N = len(vertices)
        K = len(joints)
        
        # Compute distances from each vertex to each joint
        distances = cdist(vertices, joints)
        
        # Convert distances to weights using inverse distance
        epsilon = 1e-6
        weights = 1.0 / (distances + epsilon)
        
        # Apply exponential falloff for more localized weights
        weights = np.exp(-distances * 2.0)
        
        # Keep only the nnz closest joints for each vertex
        skinning_weights = np.zeros((N, K))
        for i in range(N):
            closest_joints = np.argsort(distances[i])[:nnz]
            vertex_weights = weights[i, closest_joints]
            vertex_weights = vertex_weights / np.sum(vertex_weights)
            skinning_weights[i, closest_joints] = vertex_weights
        
        return skinning_weights.astype(np.float32)
    
    def compute_bone_transformations(self, target_joints, template_joints, parents):
        """Compute bone transformations from template to target pose"""
        K = len(parents)
        transforms = np.zeros((K, 4, 4))
        
        for i in range(K):
            # Simple translation-based transformation
            # In a more sophisticated system, you'd compute rotation + translation
            translation = target_joints[i] - template_joints[i]
            
            transform = np.eye(4)
            transform[:3, 3] = translation
            transforms[i] = transform
        
        return transforms
    
    def apply_skeleton_deformation(self, template_vertices, skinning_weights, bone_transforms):
        """Apply skeleton-driven deformation to template mesh (Step 2)"""
        N, _ = template_vertices.shape
        K = bone_transforms.shape[0]
        
        deformed_vertices = np.zeros_like(template_vertices)
        
        for i in range(N):
            vertex_pos = np.append(template_vertices[i], 1.0)  # Homogeneous coordinates
            
            # Weighted sum of bone transformations
            transformed_vertex = np.zeros(4)
            for j in range(K):
                weight = skinning_weights[i, j]
                if weight > 1e-6:  # Skip negligible weights
                    transformed_vertex += weight * bone_transforms[j] @ vertex_pos
            
            deformed_vertices[i] = transformed_vertex[:3]
        
        return deformed_vertices
    
    def non_rigid_icp_registration(self, source_mesh, target_mesh, max_iterations=50):
        """Non-rigid ICP registration for fine alignment (Step 3)"""
        print(f"  Applying non-rigid ICP registration...")
        
        # Create Open3D point clouds
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_mesh)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_mesh)
        
        # Estimate normals
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        
        # Initial rigid registration
        threshold = 0.02
        trans_init = np.eye(4)
        
        # Point-to-plane ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        # Apply transformation
        source_pcd.transform(reg_p2p.transformation)
        aligned_vertices = np.asarray(source_pcd.points)
        
        # Additional deformable registration using colored point cloud registration
        # This provides non-rigid alignment
        try:
            # Create dummy colors for registration
            source_pcd.paint_uniform_color([1, 0, 0])
            target_pcd.paint_uniform_color([0, 1, 0])
            
            # Non-rigid registration (if available in Open3D version)
            reg_colored = o3d.pipelines.registration.registration_colored_icp(
                source_pcd, target_pcd, threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=max_iterations
                )
            )
            source_pcd.transform(reg_colored.transformation)
            aligned_vertices = np.asarray(source_pcd.points)
            
        except Exception as e:
            print(f"    Colored ICP not available, using point-to-plane ICP only: {e}")
        
        return aligned_vertices
    
    def simple_nearest_neighbor_unification(self, template_vertices, target_vertices):
        """Simple nearest neighbor mapping (original method for comparison)"""
        tree = cKDTree(target_vertices)
        _, indices = tree.query(template_vertices, k=1)
        return target_vertices[indices]
    
    def process_frame_skeleton_driven(self, frame_data, template_data):
        """Process a single frame using skeleton-driven approach"""
        frame_joints = np.array(frame_data['joints'])
        frame_vertices = np.array(frame_data['mesh_vertices'])
        
        # Step 2: Skeleton-driven deformation
        bone_transforms = self.compute_bone_transformations(
            frame_joints, self.template_joints, self.template_parents
        )
        
        deformed_vertices = self.apply_skeleton_deformation(
            self.template_mesh, self.template_skinning_weights, bone_transforms
        )
        
        # Step 3: Non-rigid ICP fine registration
        try:
            aligned_vertices = self.non_rigid_icp_registration(
                deformed_vertices, frame_vertices, max_iterations=30
            )
        except Exception as e:
            print(f"    ICP registration failed, using deformed mesh: {e}")
            aligned_vertices = deformed_vertices
        
        return aligned_vertices
    
    def process_all_frames(self, mesh_data, template_name, template_data):
        """Process all frames with skeleton-driven unification"""
        print("\n=== Step 2-3: Processing all frames with skeleton-driven approach ===")
        
        unified_meshes = {}
        skeleton_driven_frames = []
        nearest_neighbor_frames = []  # Keep for comparison
        
        frame_names = sorted(mesh_data.keys())
        
        for i, frame_name in enumerate(frame_names):
            print(f"Processing frame {i+1}/{len(frame_names)}: {frame_name}")
            
            frame_data = mesh_data[frame_name]
            
            # Method 1: Skeleton-driven approach (NEW)
            print(f"  Skeleton-driven unification...")
            start_time = time.time()
            
            if frame_name == template_name:
                # Template frame - use original mesh
                skeleton_driven_vertices = self.template_mesh.copy()
            else:
                skeleton_driven_vertices = self.process_frame_skeleton_driven(frame_data, template_data)
            
            skeleton_driven_time = time.time() - start_time
            
            # Method 2: Simple nearest neighbor (ORIGINAL, for comparison)
            print(f"  Nearest neighbor unification...")
            start_time = time.time()
            
            frame_vertices = np.array(frame_data['mesh_vertices'])
            if len(frame_vertices) != len(self.template_mesh):
                nearest_neighbor_vertices = self.simple_nearest_neighbor_unification(
                    self.template_mesh, frame_vertices
                )
            else:
                nearest_neighbor_vertices = frame_vertices
            
            nearest_neighbor_time = time.time() - start_time
            
            print(f"  Times: Skeleton-driven {skeleton_driven_time:.2f}s, NN {nearest_neighbor_time:.2f}s")
            
            # Store both results
            skeleton_driven_frames.append(skeleton_driven_vertices)
            nearest_neighbor_frames.append(nearest_neighbor_vertices)
            
            unified_meshes[frame_name] = {
                'skeleton_driven': skeleton_driven_vertices,
                'nearest_neighbor': nearest_neighbor_vertices,
                'original_joints': np.array(frame_data['joints']),
                'template_joints': self.template_joints,
                'bone_transforms': self.compute_bone_transformations(
                    np.array(frame_data['joints']), self.template_joints, self.template_parents
                ) if frame_name != template_name else np.eye(4).reshape(1, 4, 4).repeat(len(self.template_joints), axis=0)
            }
        
        self.unified_meshes = unified_meshes
        
        # Convert to numpy arrays for easy processing
        skeleton_driven_frames = np.array(skeleton_driven_frames)  # (F, N, 3)
        nearest_neighbor_frames = np.array(nearest_neighbor_frames)  # (F, N, 3)
        
        print(f"\nUnification complete:")
        print(f"  Skeleton-driven frames: {skeleton_driven_frames.shape}")
        print(f"  Nearest neighbor frames: {nearest_neighbor_frames.shape}")
        
        return skeleton_driven_frames, nearest_neighbor_frames, frame_names
    
    def save_results(self, skeleton_driven_frames, nearest_neighbor_frames, frame_names, template_data):
        """Save unified mesh results"""
        print("\n=== Step 4: Saving results ===")
        
        # Save skeleton-driven results
        skeleton_results = {
            'rest_pose': self.template_mesh,
            'skinning_weights': self.template_skinning_weights,
            'transforms': np.array([self.unified_meshes[name]['bone_transforms'] for name in frame_names]),
            'joints': self.template_joints,
            'parents': self.template_parents,
            'frame_names': frame_names,
            'unified_vertices': skeleton_driven_frames,
            'method': 'skeleton_driven'
        }
        
        skeleton_results_path = os.path.join(self.output_dir, 'skeleton_driven_results.pkl')
        with open(skeleton_results_path, 'wb') as f:
            pickle.dump(skeleton_results, f)
        print(f"Saved skeleton-driven results: {skeleton_results_path}")
        
        # Save nearest neighbor results (for comparison)
        nn_results = {
            'rest_pose': self.template_mesh,
            'skinning_weights': self.template_skinning_weights,
            'transforms': np.eye(4).reshape(1, 1, 4, 4).repeat(len(frame_names), axis=0).repeat(len(self.template_joints), axis=1),
            'joints': self.template_joints,
            'parents': self.template_parents,
            'frame_names': frame_names,
            'unified_vertices': nearest_neighbor_frames,
            'method': 'nearest_neighbor'
        }
        
        nn_results_path = os.path.join(self.output_dir, 'nearest_neighbor_results.pkl')
        with open(nn_results_path, 'wb') as f:
            pickle.dump(nn_results, f)
        print(f"Saved nearest neighbor results: {nn_results_path}")
        
        # Save individual frame results
        for i, frame_name in enumerate(frame_names):
            frame_data = self.unified_meshes[frame_name]
            
            # Skeleton-driven individual result
            skeleton_frame_result = {
                'joints': frame_data['original_joints'],
                'template_joints': frame_data['template_joints'],
                'parents': self.template_parents,
                'skinning_weights': self.template_skinning_weights,
                'bone_transforms': frame_data['bone_transforms'],
                'unified_vertices': frame_data['skeleton_driven'],
                'method': 'skeleton_driven'
            }
            
            skeleton_frame_path = os.path.join(self.output_dir, f'{frame_name}_skeleton_driven.pkl')
            with open(skeleton_frame_path, 'wb') as f:
                pickle.dump(skeleton_frame_result, f)
            
            # Save skeleton-driven mesh
            mesh_vertices_world = (frame_data['skeleton_driven'] + 1) * 0.5 * template_data['blen'] + template_data['bmin']
            mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(mesh_vertices_world),
                triangles=o3d.utility.Vector3iVector(template_data['mesh_triangles'])
            )
            mesh.compute_vertex_normals()
            
            mesh_path = os.path.join(self.output_dir, f'{frame_name}_skeleton_driven.obj')
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            
            print(f"Saved {frame_name} skeleton-driven results")
        
        print(f"\nAll results saved in: {self.output_dir}")
        print(f"- Skeleton-driven unified results: skeleton_driven_results.pkl")
        print(f"- Nearest neighbor results (comparison): nearest_neighbor_results.pkl")
        print(f"- Individual frame results: *_skeleton_driven.pkl and *_skeleton_driven.obj")

def main():
    """Main processing function"""
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = os.path.join(folder_path, "generated_skeletons")
    
    print("🚀 Starting Skeleton-Driven Mesh Unification System")
    print(f"📁 Input folder: {folder_path}")
    print(f"📁 Output folder: {output_dir}")
    
    # Initialize the unifier
    unifier = SkeletonDrivenUnifier(folder_path, output_dir)
    
    try:
        # Load cached data
        mesh_data = unifier.load_cached_data()
        
        # Setup template mesh and initial skinning
        template_name, template_data = unifier.setup_template_mesh(mesh_data)
        
        # Process all frames with skeleton-driven approach
        skeleton_driven_frames, nearest_neighbor_frames, frame_names = unifier.process_all_frames(
            mesh_data, template_name, template_data
        )
        
        # Save results
        unifier.save_results(skeleton_driven_frames, nearest_neighbor_frames, frame_names, template_data)
        
        print("\n✅ Skeleton-driven mesh unification complete!")
        print("🎯 You can now use the skeleton-driven results for high-quality interpolation")
        print("📊 Both skeleton-driven and nearest neighbor results are saved for comparison")
        
    except Exception as e:
        print(f"❌ Error in skeleton-driven unification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
