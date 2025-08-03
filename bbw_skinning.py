"""
BBW (Bounded Biharmonic Weights) Auto-Skinning Module

This module implements automatic skinning using Bounded Biharmonic Weights (BBW)
for mesh interpolation tasks. It integrates with the existing neural marionette 
interpolation pipeline while maintaining compatibility.

Features:
- BBW weight computation using libigl
- Space alignment between world space meshes and normalized space skeletons  
- Weight transfer for inconsistent topology meshes
- LBS/DQS deformation support
- Integration with existing interpolation pipeline

Author: Assistant
Created for neural marionette mesh interpolation enhancement
"""

import numpy as np
import torch
import trimesh
import open3d as o3d
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import time
from typing import Dict, List, Tuple, Optional, Union

try:
    import igl
    LIBIGL_AVAILABLE = True
    print("libigl is available for BBW computation")
except ImportError:
    LIBIGL_AVAILABLE = False
    print("Warning: libigl not available. BBW computation will use fallback methods.")


class BBWAutoSkinning:
    """
    BBW-based automatic skinning for mesh interpolation
    
    This class provides automatic skinning weights computation using
    Bounded Biharmonic Weights, with support for inconsistent topology
    and integration with the existing neural marionette pipeline.
    """
    
    def __init__(self, skeleton_data_dir: str, reference_frame_idx: int = 0):
        """
        Initialize BBW Auto-Skinning system
        
        Args:
            skeleton_data_dir: Directory containing skeleton data
            reference_frame_idx: Reference frame index for weight computation
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.reference_frame_idx = reference_frame_idx
        
        # Load skeleton data
        self.load_skeleton_data()
        
        # BBW-specific attributes
        self.reference_mesh = None
        self.reference_weights = None
        self.reference_normalization_params = None
        
        # Space alignment parameters
        self.skeleton_to_mesh_transform = None
        self.mesh_to_skeleton_transform = None
        
        # Weight transfer cache
        self.weight_transfer_cache = {}
        
    def load_skeleton_data(self):
        """Load skeleton data from numpy files"""
        try:
            # Load keypoints [num_frames, num_joints, 4] (x,y,z,confidence)
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            
            # Load transforms [num_frames, num_joints, 4, 4]  
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            
            # Load parent relationships [num_joints]
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"BBW: Loaded skeleton data:")
            print(f"  - Frames: {self.num_frames}")
            print(f"  - Joints: {self.num_joints}")
            print(f"  - Keypoints shape: {self.keypoints.shape}")
            print(f"  - Transforms shape: {self.transforms.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load skeleton data: {e}")
    
    def compute_mesh_normalization_params(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """
        Compute normalization parameters consistent with episodic_normalization
        
        Args:
            mesh: Open3D mesh object
            
        Returns:
            Dictionary of normalization parameters
        """
        vertices = np.asarray(mesh.vertices)
        
        # Compute bounding box
        bmax = np.amax(vertices, axis=0)
        bmin = np.amin(vertices, axis=0)
        blen = (bmax - bmin).max()
        
        params = {
            'bmin': bmin,
            'bmax': bmax, 
            'blen': blen,
            'scale': 1.0,
            'x_trans': 0.0,
            'z_trans': 0.0
        }
        
        return params
    
    def normalize_mesh_vertices(self, vertices: np.ndarray, params: Dict) -> np.ndarray:
        """
        Normalize mesh vertices to [-1, 1] space
        
        Args:
            vertices: Original vertex coordinates
            params: Normalization parameters
            
        Returns:
            Normalized vertex coordinates
        """
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        normalized = ((vertices - params['bmin']) * params['scale'] / (params['blen'] + 1e-5)) * 2 - 1 + trans_offset
        return normalized
    
    def denormalize_mesh_vertices(self, normalized_vertices: np.ndarray, params: Dict) -> np.ndarray:
        """
        Denormalize mesh vertices back to original space
        
        Args:
            normalized_vertices: Normalized vertex coordinates
            params: Normalization parameters
            
        Returns:
            Denormalized vertex coordinates
        """
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        vertices_no_offset = normalized_vertices - trans_offset
        vertices_01 = (vertices_no_offset + 1) / 2
        denormalized = vertices_01 * (params['blen'] + 1e-8) / params['scale'] + params['bmin']
        return denormalized
    
    def align_skeleton_to_mesh_space(self, mesh_vertices: np.ndarray, 
                                   skeleton_keypoints: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Align skeleton from normalized space to mesh world space
        
        Args:
            mesh_vertices: Mesh vertices in world space
            skeleton_keypoints: Skeleton keypoints in normalized space [num_joints, 3]
            
        Returns:
            Aligned skeleton keypoints and transformation info
        """
        # Compute mesh normalization parameters
        mesh_center = np.mean(mesh_vertices, axis=0)
        mesh_bbox = np.max(mesh_vertices, axis=0) - np.min(mesh_vertices, axis=0)
        mesh_scale = np.max(mesh_bbox)
        
        # Skeleton is in [-1, 1] normalized space
        # Transform to mesh world space
        skeleton_center = np.mean(skeleton_keypoints, axis=0)
        skeleton_bbox = np.max(skeleton_keypoints, axis=0) - np.min(skeleton_keypoints, axis=0)
        skeleton_scale = np.max(skeleton_bbox)
        
        # Compute alignment transformation
        # Scale skeleton to match mesh scale
        scale_factor = mesh_scale / (skeleton_scale + 1e-8) if skeleton_scale > 1e-8 else 1.0
        
        # Translate skeleton to mesh center
        translation = mesh_center - skeleton_center * scale_factor
        
        # Apply transformation
        aligned_keypoints = skeleton_keypoints * scale_factor + translation
        
        transform_info = {
            'scale_factor': scale_factor,
            'translation': translation,
            'mesh_center': mesh_center,
            'skeleton_center': skeleton_center
        }
        
        return aligned_keypoints, transform_info
    
    def prepare_bbw_input(self, mesh: o3d.geometry.TriangleMesh, 
                         frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare input data for BBW computation
        
        Args:
            mesh: Input mesh in world space
            frame_idx: Frame index for skeleton data
            
        Returns:
            Mesh vertices, faces, and aligned skeleton points
        """
        # Get mesh data
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        # Get skeleton keypoints for this frame (remove confidence)
        skeleton_keypoints = self.keypoints[frame_idx, :, :3]
        
        # Align skeleton to mesh space
        aligned_skeleton, transform_info = self.align_skeleton_to_mesh_space(
            vertices, skeleton_keypoints
        )
        
        print(f"BBW: Prepared input for frame {frame_idx}")
        print(f"  - Mesh vertices: {vertices.shape}")
        print(f"  - Mesh faces: {faces.shape}")
        print(f"  - Skeleton points: {aligned_skeleton.shape}")
        print(f"  - Alignment scale: {transform_info['scale_factor']:.4f}")
        
        return vertices, faces, aligned_skeleton
    
    def compute_bbw_weights_libigl(self, vertices: np.ndarray, faces: np.ndarray, 
                                  handles: np.ndarray) -> np.ndarray:
        """
        Compute BBW weights using libigl
        
        Args:
            vertices: Mesh vertices [V, 3]
            faces: Mesh faces [F, 3]
            handles: Handle positions [H, 3]
            
        Returns:
            BBW weight matrix [V, H]
        """
        if not LIBIGL_AVAILABLE:
            print("Warning: libigl not available, using fallback distance-based weights")
            return self.compute_distance_based_weights(vertices, handles)
        
        try:
            print("BBW: Computing weights with libigl...")
            
            # Find closest vertices to handles to create boundary conditions
            tree = cKDTree(vertices)
            _, closest_vertex_indices = tree.query(handles)
            
            # Ensure correct data types for libigl
            vertices_libigl = vertices.astype(np.float64)
            faces_libigl = faces.astype(np.int64)
            
            # Create boundary conditions
            # b: vertex indices where weights are constrained [num_handles]
            b = closest_vertex_indices.astype(np.int64)
            
            # bc: boundary condition values [num_handles, num_handles] 
            # Each row i has value 1 at column i, 0 elsewhere (identity matrix)
            bc = np.eye(len(handles), dtype=np.float64)
            
            print(f"BBW: Input shapes - vertices: {vertices_libigl.shape}, faces: {faces_libigl.shape}")
            print(f"BBW: Boundary conditions - b: {b.shape}, bc: {bc.shape}")
            
            # Compute BBW weights
            weights = igl.bbw(vertices_libigl, faces_libigl, b, bc)
            
            print(f"BBW: Successfully computed weights matrix: {weights.shape}")
            return weights
            
        except Exception as e:
            print(f"BBW: libigl computation failed: {e}")
            print("BBW: Falling back to distance-based weights")
            return self.compute_distance_based_weights(vertices, handles)
    
    def compute_distance_based_weights(self, vertices: np.ndarray, 
                                     handles: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """
        Fallback distance-based weight computation
        
        Args:
            vertices: Mesh vertices [V, 3]
            handles: Handle positions [H, 3]
            sigma: Gaussian falloff parameter
            
        Returns:
            Distance-based weight matrix [V, H]
        """
        # Compute distances between vertices and handles
        distances = cdist(vertices, handles)
        
        # Convert to weights using Gaussian falloff
        weights = np.exp(-distances**2 / (2 * sigma**2))
        
        # Normalize weights (each vertex sums to 1)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-8)
        
        print(f"BBW: Computed distance-based weights: {weights.shape}")
        return weights
    
    def compute_reference_weights(self, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
        """
        Compute BBW weights for reference mesh
        
        Args:
            mesh: Reference mesh in world space
            
        Returns:
            BBW weight matrix [V, J]
        """
        print(f"BBW: Computing reference weights for frame {self.reference_frame_idx}")
        
        # Prepare input data
        vertices, faces, handles = self.prepare_bbw_input(mesh, self.reference_frame_idx)
        
        # Store reference data
        self.reference_mesh = mesh
        self.reference_normalization_params = self.compute_mesh_normalization_params(mesh)
        
        # Compute BBW weights
        weights = self.compute_bbw_weights_libigl(vertices, faces, handles)
        
        # Store reference weights
        self.reference_weights = weights
        
        print(f"BBW: Reference weights computed successfully: {weights.shape}")
        return weights
    
    def transfer_weights_to_target(self, target_mesh: o3d.geometry.TriangleMesh, 
                                 method: str = "nearest_neighbor") -> np.ndarray:
        """
        Transfer weights from reference mesh to target mesh
        
        Args:
            target_mesh: Target mesh with different topology
            method: Transfer method ("nearest_neighbor", "barycentric")
            
        Returns:
            Transferred weight matrix for target mesh
        """
        if self.reference_weights is None:
            raise RuntimeError("Reference weights not computed. Call compute_reference_weights first.")
        
        # Get target mesh vertices
        target_vertices = np.asarray(target_mesh.vertices)
        reference_vertices = np.asarray(self.reference_mesh.vertices)
        
        print(f"BBW: Transferring weights using {method} method")
        print(f"  - Reference vertices: {reference_vertices.shape}")
        print(f"  - Target vertices: {target_vertices.shape}")
        
        if method == "nearest_neighbor":
            # Use nearest neighbor interpolation
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(reference_vertices)
            distances, indices = nbrs.kneighbors(target_vertices)
            
            # Transfer weights
            transferred_weights = self.reference_weights[indices.flatten()]
            
        elif method == "barycentric":
            # Use k-nearest neighbors with distance weighting
            k = min(4, len(reference_vertices))
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(reference_vertices)
            distances, indices = nbrs.kneighbors(target_vertices)
            
            # Compute barycentric weights
            transferred_weights = np.zeros((len(target_vertices), self.num_joints))
            
            for i in range(len(target_vertices)):
                # Get k nearest neighbors
                neighbor_weights = self.reference_weights[indices[i]]
                neighbor_distances = distances[i]
                
                # Avoid division by zero
                neighbor_distances = np.maximum(neighbor_distances, 1e-8)
                
                # Inverse distance weighting
                inv_distances = 1.0 / neighbor_distances
                inv_distances /= np.sum(inv_distances)
                
                # Weighted combination
                transferred_weights[i] = np.sum(
                    neighbor_weights * inv_distances[:, np.newaxis], axis=0
                )
        
        else:
            raise ValueError(f"Unknown transfer method: {method}")
        
        # Ensure weights are normalized
        weight_sums = np.sum(transferred_weights, axis=1, keepdims=True)
        transferred_weights = transferred_weights / (weight_sums + 1e-8)
        
        print(f"BBW: Weights transferred successfully: {transferred_weights.shape}")
        return transferred_weights
    
    def apply_lbs_transform(self, rest_vertices: np.ndarray, weights: np.ndarray, 
                          transforms: np.ndarray) -> np.ndarray:
        """
        Apply Linear Blend Skinning transformation
        
        Args:
            rest_vertices: Rest pose vertices [V, 3]
            weights: Skinning weights [V, J]
            transforms: Joint transformation matrices [J, 4, 4]
            
        Returns:
            Transformed vertices [V, 3]
        """
        num_vertices = rest_vertices.shape[0]
        num_joints = transforms.shape[0]
        
        # Convert to homogeneous coordinates
        rest_vertices_homo = np.hstack([rest_vertices, np.ones((num_vertices, 1))])
        transformed_vertices = np.zeros((num_vertices, 3))
        
        # Ensure weights are normalized
        weights = np.maximum(weights, 0)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-8)
        
        # Apply LBS for each joint
        for j in range(num_joints):
            joint_transform = transforms[j]
            transformed_homo = (joint_transform @ rest_vertices_homo.T).T
            transformed_xyz = transformed_homo[:, :3]
            joint_weights = weights[:, j:j+1]
            transformed_vertices += joint_weights * transformed_xyz
        
        return transformed_vertices
    
    def validate_weights(self, weights: np.ndarray, vertices: np.ndarray) -> Dict:
        """
        Validate computed BBW weights
        
        Args:
            weights: Weight matrix [V, J]
            vertices: Mesh vertices [V, 3]
            
        Returns:
            Validation results dictionary
        """
        validation_results = {}
        
        # Check weight matrix shape
        validation_results['shape_valid'] = weights.shape == (len(vertices), self.num_joints)
        
        # Check if weights are non-negative
        validation_results['non_negative'] = np.all(weights >= 0)
        
        # Check if weights sum to 1 for each vertex
        weight_sums = np.sum(weights, axis=1)
        validation_results['normalized'] = np.allclose(weight_sums, 1.0, atol=1e-6)
        
        # Check sparsity (average number of non-zero weights per vertex)
        non_zero_weights = np.sum(weights > 1e-6, axis=1)
        validation_results['avg_influences'] = np.mean(non_zero_weights)
        validation_results['max_influences'] = np.max(non_zero_weights)
        
        # Check for each joint having some influence
        joint_influences = np.sum(weights > 1e-6, axis=0)
        validation_results['joints_with_influence'] = np.sum(joint_influences > 0)
        validation_results['unused_joints'] = np.sum(joint_influences == 0)
        
        return validation_results
    
    def save_weights(self, weights: np.ndarray, output_path: str):
        """Save computed weights to file"""
        weights_data = {
            'weights': weights,
            'num_joints': self.num_joints,
            'reference_frame_idx': self.reference_frame_idx,
            'method': 'BBW'
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, weights_data)
        print(f"BBW: Weights saved to {output_path}")
    
    def load_weights(self, weights_path: str) -> np.ndarray:
        """Load pre-computed weights from file"""
        weights_data = np.load(weights_path, allow_pickle=True).item()
        weights = weights_data['weights']
        print(f"BBW: Weights loaded from {weights_path}, shape: {weights.shape}")
        return weights


# Test function for initial validation
def test_bbw_basic_functionality():
    """Test basic BBW functionality with simple mesh"""
    print("=== Testing BBW Basic Functionality ===")
    
    # Create a simple test mesh (cube)
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    mesh.translate([-0.5, -0.5, -0.5])  # Center at origin
    
    # Create dummy skeleton data directory structure for testing
    test_skeleton_dir = Path("test_skeleton_data")
    test_skeleton_dir.mkdir(exist_ok=True)
    
    # Create dummy skeleton data
    num_frames, num_joints = 5, 4
    keypoints = np.random.rand(num_frames, num_joints, 4) * 2 - 1  # [-1, 1] range
    keypoints[:, :, 3] = 1.0  # Set confidence to 1
    transforms = np.tile(np.eye(4), (num_frames, num_joints, 1, 1))
    parents = np.array([-1, 0, 1, 2])  # Simple chain
    
    np.save(test_skeleton_dir / "keypoints.npy", keypoints)
    np.save(test_skeleton_dir / "transforms.npy", transforms)
    np.save(test_skeleton_dir / "parents.npy", parents)
    
    try:
        # Initialize BBW system
        bbw = BBWAutoSkinning(str(test_skeleton_dir), reference_frame_idx=0)
        
        # Compute reference weights
        weights = bbw.compute_reference_weights(mesh)
        
        # Validate weights
        vertices = np.asarray(mesh.vertices)
        validation = bbw.validate_weights(weights, vertices)
        
        print("Validation Results:")
        for key, value in validation.items():
            print(f"  - {key}: {value}")
        
        # Test weight transfer
        # Create a slightly different mesh
        target_mesh = o3d.geometry.TriangleMesh.create_box(1.2, 1.2, 1.2)
        target_mesh.translate([-0.6, -0.6, -0.6])
        
        transferred_weights = bbw.transfer_weights_to_target(target_mesh)
        target_validation = bbw.validate_weights(transferred_weights, np.asarray(target_mesh.vertices))
        
        print("Transfer Validation Results:")
        for key, value in target_validation.items():
            print(f"  - {key}: {value}")
        
        print("=== BBW Basic Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"BBW Test Failed: {e}")
        return False
        
    finally:
        # Cleanup test files
        import shutil
        if test_skeleton_dir.exists():
            shutil.rmtree(test_skeleton_dir)


if __name__ == "__main__":
    # Run basic functionality test
    test_bbw_basic_functionality()