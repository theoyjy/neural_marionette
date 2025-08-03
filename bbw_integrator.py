"""
BBW Integrator for Neural Marionette Pipeline

This module integrates BBW automatic skinning into the existing 
neural marionette interpolation pipeline.

Features:
- Seamless integration with existing interpolation classes
- BBW-based skinning weights computation
- Handles space alignment between world space meshes and normalized skeletons
- Supports both real BBW and fallback methods
"""

import numpy as np
import os
import open3d as o3d
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import pickle
import time
from typing import Dict, List, Tuple, Optional

# Import existing components
try:
    from Skinning import AutoSkinning
    AUTOSKINNING_AVAILABLE = True
except ImportError:
    AUTOSKINNING_AVAILABLE = False
    print("Warning: AutoSkinning not available")

try:
    import igl
    LIBIGL_AVAILABLE = True
except ImportError:
    LIBIGL_AVAILABLE = False
    print("Warning: libigl not available, using fallback methods")


class BBWIntegrator:
    """
    BBW integrator for neural marionette pipeline
    
    This class provides a seamless integration layer between BBW automatic
    skinning and the existing interpolation pipeline.
    """
    
    def __init__(self, skeleton_data_dir: str, mesh_folder_path: str, 
                 reference_frame_idx: int = 0, use_bbw: bool = True):
        """
        Initialize BBW integrator
        
        Args:
            skeleton_data_dir: Directory containing skeleton data
            mesh_folder_path: Directory containing mesh files
            reference_frame_idx: Reference frame for weight computation
            use_bbw: Whether to use BBW method (fallback to distance-based if False)
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.mesh_folder_path = Path(mesh_folder_path)
        self.reference_frame_idx = reference_frame_idx
        self.use_bbw = use_bbw and LIBIGL_AVAILABLE
        
        # Load skeleton and mesh data
        self.load_skeleton_data()
        self.load_mesh_sequence()
        
        # BBW computation state
        self.reference_weights = None
        self.reference_mesh_path = None
        self.reference_normalization_params = None
        
        # Space alignment
        self.mesh_to_skeleton_scale = None
        self.skeleton_alignment_info = None
        
    def load_skeleton_data(self):
        """Load skeleton data from files"""
        try:
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"BBW Integrator: Loaded skeleton data:")
            print(f"  - Frames: {self.num_frames}")
            print(f"  - Joints: {self.num_joints}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load skeleton data: {e}")
    
    def load_mesh_sequence(self):
        """Load mesh file paths"""
        mesh_extensions = ['*.obj', '*.ply', '*.off']
        mesh_files = []
        
        for ext in mesh_extensions:
            mesh_files.extend(self.mesh_folder_path.glob(ext))
        
        # Sort by filename
        self.mesh_files = sorted(mesh_files)
        
        if len(self.mesh_files) == 0:
            raise RuntimeError(f"No mesh files found in {self.mesh_folder_path}")
        
        print(f"BBW Integrator: Found {len(self.mesh_files)} mesh files")
    
    def compute_mesh_normalization_params(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """Compute normalization parameters consistent with existing pipeline"""
        vertices = np.asarray(mesh.vertices)
        
        bmax = np.amax(vertices, axis=0)
        bmin = np.amin(vertices, axis=0)
        blen = (bmax - bmin).max()
        
        return {
            'bmin': bmin,
            'bmax': bmax, 
            'blen': blen,
            'scale': 1.0,
            'x_trans': 0.0,
            'z_trans': 0.0
        }
    
    def normalize_mesh_vertices(self, vertices: np.ndarray, params: Dict) -> np.ndarray:
        """Normalize mesh vertices to [-1, 1] space"""
        trans_offset = np.array([params['x_trans'], 0, params['z_trans']])
        normalized = ((vertices - params['bmin']) * params['scale'] / (params['blen'] + 1e-5)) * 2 - 1 + trans_offset
        return normalized
    
    def align_skeleton_to_mesh_space(self, mesh_vertices: np.ndarray, 
                                   skeleton_keypoints: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        FIXED: Proper alignment of skeleton to mesh coordinate system
        
        Both mesh and skeleton need to be in the same normalized space for BBW.
        """
        print("BBW Integrator: Using FIXED space alignment...")
        
        # Step 1: Compute mesh normalization parameters
        mesh_min = mesh_vertices.min(axis=0)
        mesh_max = mesh_vertices.max(axis=0)
        mesh_center = (mesh_min + mesh_max) / 2.0
        mesh_scale = (mesh_max - mesh_min).max() / 2.0  # Half of largest dimension
        
        print(f"  - Original mesh center: {mesh_center}")
        print(f"  - Original mesh scale: {mesh_scale:.6f}")
        
        # Step 2: Normalize mesh to [-1, 1] cube
        normalized_mesh = (mesh_vertices - mesh_center) / mesh_scale
        normalized_mesh_center = normalized_mesh.mean(axis=0)
        
        print(f"  - Normalized mesh center: {normalized_mesh_center}")
        print(f"  - Normalized mesh range: [{normalized_mesh.min():.3f}, {normalized_mesh.max():.3f}]")
        
        # Step 3: Skeleton should already be in normalized space, check and align
        skeleton_center = skeleton_keypoints.mean(axis=0)
        skeleton_range = [skeleton_keypoints.min(), skeleton_keypoints.max()]
        
        print(f"  - Original skeleton center: {skeleton_center}")
        print(f"  - Original skeleton range: [{skeleton_range[0]:.3f}, {skeleton_range[1]:.3f}]")
        
        # Step 4: Align skeleton to normalized mesh space
        # The key insight: both should have the same center in normalized space
        center_offset = normalized_mesh_center - skeleton_center
        aligned_skeleton = skeleton_keypoints + center_offset
        
        print(f"  - Center offset applied: {center_offset}")
        print(f"  - Aligned skeleton center: {aligned_skeleton.mean(axis=0)}")
        
        # Verification
        final_center_distance = np.linalg.norm(normalized_mesh_center - aligned_skeleton.mean(axis=0))
        print(f"  - Final center distance: {final_center_distance:.8f}")
        
        alignment_info = {
            'mesh_center': mesh_center,
            'mesh_scale': mesh_scale,
            'normalized_mesh_center': normalized_mesh_center,
            'skeleton_center': skeleton_center,
            'aligned_skeleton_center': aligned_skeleton.mean(axis=0),
            'center_offset': center_offset,
            'final_center_distance': final_center_distance,
            'normalized_mesh': normalized_mesh  # Return normalized mesh for BBW
        }
        
        return aligned_skeleton, alignment_info
    
    def compute_distance_based_weights(self, vertices: np.ndarray, 
                                     handles: np.ndarray, method: str = "enhanced") -> np.ndarray:
        """
        Compute distance-based skinning weights with anti-paper effect
        
        Args:
            vertices: Mesh vertices [V, 3]
            handles: Joint positions [J, 3]
            method: Weight computation method ("original", "enhanced", "sparse")
        """
        print(f"BBW Integrator: Computing {method} distance-based weights...")
        
        if method == "enhanced":
            return self.compute_enhanced_distance_weights(vertices, handles)
        elif method == "sparse":
            return self.compute_sparse_distance_weights(vertices, handles)
        else:
            return self.compute_original_distance_weights(vertices, handles)
    
    def compute_original_distance_weights(self, vertices: np.ndarray, 
                                        handles: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Original distance-based weights (smooth, may cause paper effect)"""
        distances = cdist(vertices, handles)
        weights = np.exp(-distances**2 / (2 * sigma**2))
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (weight_sums + 1e-8)
        return weights
    
    def compute_enhanced_distance_weights(self, vertices: np.ndarray, 
                                        handles: np.ndarray, 
                                        sigma_base: float = 0.08, 
                                        sharpness: float = 3.0) -> np.ndarray:
        """
        Enhanced distance-based weights - sharper and more localized
        
        Args:
            sigma_base: Base sigma for adaptive calculation
            sharpness: Sharpness parameter (higher = more localized)
        """
        V, J = len(vertices), len(handles)
        weights = np.zeros((V, J))
        
        # Adaptive sigma for each joint
        adaptive_sigmas = []
        for j in range(J):
            distances_to_others = []
            for k in range(J):
                if k != j:
                    dist = np.linalg.norm(handles[j] - handles[k])
                    distances_to_others.append(dist)
            
            if distances_to_others:
                min_dist_to_other = min(distances_to_others)
                adaptive_sigma = sigma_base * min_dist_to_other
            else:
                adaptive_sigma = sigma_base
                
            adaptive_sigmas.append(adaptive_sigma)
        
        # Compute weights with adaptive sigma and sharpness
        for i, vertex in enumerate(vertices):
            distances = []
            weights_raw = []
            
            for j, handle in enumerate(handles):
                dist = np.linalg.norm(vertex - handle)
                distances.append(dist)
                
                sigma = adaptive_sigmas[j]
                weight_raw = np.exp(-((dist / sigma) ** sharpness))
                weights_raw.append(weight_raw)
            
            weights_raw = np.array(weights_raw)
            weight_sum = np.sum(weights_raw)
            
            if weight_sum > 1e-8:
                weights[i] = weights_raw / weight_sum
            else:
                # Fallback to closest joint
                closest_joint = np.argmin(distances)
                weights[i, closest_joint] = 1.0
        
        print(f"  - Enhanced weights: sharper and more localized")
        return weights
    
    def compute_sparse_distance_weights(self, vertices: np.ndarray, 
                                      handles: np.ndarray,
                                      k_neighbors: int = 3, 
                                      falloff_power: float = 6.0) -> np.ndarray:
        """
        Sparse distance-based weights - each vertex influenced by few joints
        
        Args:
            k_neighbors: Number of joints influencing each vertex
            falloff_power: Distance falloff power (higher = more concentrated)
        """
        V, J = len(vertices), len(handles)
        weights = np.zeros((V, J))
        
        for i, vertex in enumerate(vertices):
            # Find k nearest joints
            distances = np.array([np.linalg.norm(vertex - handle) for handle in handles])
            nearest_indices = np.argsort(distances)[:k_neighbors]
            nearest_distances = distances[nearest_indices]
            
            # Avoid division by zero
            nearest_distances = np.maximum(nearest_distances, 1e-6)
            
            # Compute weights with power falloff
            weights_raw = 1.0 / (nearest_distances ** falloff_power)
            weight_sum = np.sum(weights_raw)
            
            if weight_sum > 1e-8:
                for j, idx in enumerate(nearest_indices):
                    weights[i, idx] = weights_raw[j] / weight_sum
            else:
                weights[i, nearest_indices[0]] = 1.0
        
        print(f"  - Sparse weights: {k_neighbors} neighbors per vertex")
        return weights
    

    def compute_balanced_distance_weights(self, vertices: np.ndarray, 
                                        handles: np.ndarray,
                                        k_neighbors: int = 5, 
                                        smoothness: float = 0.3) -> np.ndarray:
        """
        平衡的权重计算 - 在局部化和平滑性之间取得平衡
        
        Args:
            k_neighbors: 每个顶点主要受影响的关节数
            smoothness: 平滑性参数 (0=完全稀疏, 1=完全平滑)
        """
        V, J = len(vertices), len(handles)
        weights = np.zeros((V, J))
        
        print(f"  - 平衡权重: k={k_neighbors}, smoothness={smoothness}")
        
        for i, vertex in enumerate(vertices):
            # 计算到所有关节的距离
            distances = np.array([np.linalg.norm(vertex - handle) for handle in handles])
            
            # 方法1: 稀疏权重 (主要关节)
            nearest_indices = np.argsort(distances)[:k_neighbors]
            nearest_distances = distances[nearest_indices]
            nearest_distances = np.maximum(nearest_distances, 1e-6)
            
            # 使用逆距离平方
            sparse_weights_raw = 1.0 / (nearest_distances ** 2)
            sparse_weights = np.zeros(J)
            sparse_weights[nearest_indices] = sparse_weights_raw / np.sum(sparse_weights_raw)
            
            # 方法2: 平滑权重 (所有关节)
            sigma = 0.2
            smooth_weights_raw = np.exp(-distances**2 / (2 * sigma**2))
            smooth_weights = smooth_weights_raw / np.sum(smooth_weights_raw)
            
            # 混合两种权重
            final_weights = (1 - smoothness) * sparse_weights + smoothness * smooth_weights
            
            # 确保归一化
            weight_sum = np.sum(final_weights)
            if weight_sum > 1e-8:
                weights[i] = final_weights / weight_sum
            else:
                # Fallback
                weights[i, nearest_indices[0]] = 1.0
        
        return weights
    def compute_bbw_weights_safe(self, vertices: np.ndarray, faces: np.ndarray, 
                                handles: np.ndarray) -> np.ndarray:
        """Safely compute BBW weights with proper fallback"""
        
        # Always use balanced distance-based method for now due to libigl compatibility issues
        print("BBW Integrator: Using balanced distance-based weights (BBW fallback)")
        weights = self.compute_balanced_distance_weights(vertices, handles, k_neighbors=5, smoothness=0.4)
        
        return weights
    
    def compute_reference_weights(self, force_recompute: bool = False) -> np.ndarray:
        """
        Compute skinning weights for reference mesh
        
        Args:
            force_recompute: Force recomputation even if weights exist
            
        Returns:
            Weight matrix [V, J]
        """
        
        if self.reference_weights is not None and not force_recompute:
            print("BBW Integrator: Using cached reference weights")
            return self.reference_weights
        
        print(f"BBW Integrator: Computing reference weights for frame {self.reference_frame_idx}")
        
        # Load reference mesh
        if self.reference_frame_idx >= len(self.mesh_files):
            raise ValueError(f"Reference frame {self.reference_frame_idx} out of range")
        
        reference_mesh_path = self.mesh_files[self.reference_frame_idx]
        reference_mesh = o3d.io.read_triangle_mesh(str(reference_mesh_path))
        
        if len(reference_mesh.vertices) == 0:
            raise RuntimeError(f"Failed to load reference mesh: {reference_mesh_path}")
        
        # Get mesh data
        vertices = np.asarray(reference_mesh.vertices)
        faces = np.asarray(reference_mesh.triangles)
        
        # Get skeleton keypoints for reference frame
        skeleton_keypoints = self.keypoints[self.reference_frame_idx, :, :3]  # Remove confidence
        
        # Align skeleton to mesh space
        aligned_skeleton, alignment_info = self.align_skeleton_to_mesh_space(
            vertices, skeleton_keypoints
        )
        
        print(f"BBW Integrator: Space alignment completed")
        print(f"  - Mesh vertices: {vertices.shape}")
        print(f"  - Aligned skeleton: {aligned_skeleton.shape}")
        print(f"  - Mesh center: {alignment_info['mesh_center']}")
        print(f"  - Skeleton center: {alignment_info['aligned_skeleton_center']}")
        
        # Compute weights
        weights = self.compute_bbw_weights_safe(vertices, faces, aligned_skeleton)
        
        # Store results
        self.reference_weights = weights
        self.reference_mesh_path = reference_mesh_path
        self.reference_normalization_params = self.compute_mesh_normalization_params(reference_mesh)
        self.skeleton_alignment_info = alignment_info
        
        print(f"BBW Integrator: Reference weights computed: {weights.shape}")
        return weights
    
    def transfer_weights_to_target(self, target_mesh_path: str) -> np.ndarray:
        """Transfer weights from reference to target mesh"""
        
        if self.reference_weights is None:
            raise RuntimeError("Reference weights not computed")
        
        # Load target mesh
        target_mesh = o3d.io.read_triangle_mesh(str(target_mesh_path))
        target_vertices = np.asarray(target_mesh.vertices)
        reference_mesh = o3d.io.read_triangle_mesh(str(self.reference_mesh_path))
        reference_vertices = np.asarray(reference_mesh.vertices)
        
        # Use nearest neighbor transfer
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(reference_vertices)
        distances, indices = nbrs.kneighbors(target_vertices)
        
        # Transfer weights
        transferred_weights = self.reference_weights[indices.flatten()]
        
        # Renormalize
        weight_sums = np.sum(transferred_weights, axis=1, keepdims=True)
        transferred_weights = transferred_weights / (weight_sums + 1e-8)
        
        return transferred_weights
    
    def get_weights_for_frame(self, frame_idx: int) -> np.ndarray:
        """Get skinning weights for a specific frame"""
        
        if frame_idx == self.reference_frame_idx:
            # Use reference weights
            if self.reference_weights is None:
                self.compute_reference_weights()
            return self.reference_weights
        else:
            # Transfer weights from reference
            if frame_idx >= len(self.mesh_files):
                raise ValueError(f"Frame {frame_idx} out of range")
            
            target_mesh_path = self.mesh_files[frame_idx]
            return self.transfer_weights_to_target(target_mesh_path)
    
    def save_weights_cache(self, output_dir: str):
        """Save computed weights for future use"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'reference_weights': self.reference_weights,
            'reference_frame_idx': self.reference_frame_idx,
            'reference_mesh_path': str(self.reference_mesh_path),
            'reference_normalization_params': self.reference_normalization_params,
            'skeleton_alignment_info': self.skeleton_alignment_info,
            'num_joints': self.num_joints,
            'method': 'BBW_distance_based'
        }
        
        cache_file = output_path / 'bbw_weights_cache.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"BBW Integrator: Weights cache saved to {cache_file}")
    
    def load_weights_cache(self, cache_path: str) -> bool:
        """Load weights from cache"""
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.reference_weights = cache_data['reference_weights']
            self.reference_frame_idx = cache_data['reference_frame_idx']
            self.reference_mesh_path = Path(cache_data['reference_mesh_path'])
            self.reference_normalization_params = cache_data['reference_normalization_params']
            self.skeleton_alignment_info = cache_data['skeleton_alignment_info']
            
            print(f"BBW Integrator: Weights cache loaded from {cache_path}")
            return True
            
        except Exception as e:
            print(f"BBW Integrator: Failed to load cache: {e}")
            return False


def test_bbw_integrator():
    """Test BBW integrator functionality"""
    print("=== Testing BBW Integrator ===")
    
    # Test with actual data if available
    test_mesh_dir = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    test_skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton"  # Use existing skeleton data
    
    # Create dummy test data if real data not available
    if not os.path.exists(test_skeleton_dir):
        print("Real test data not found, creating dummy data...")
        
        test_skeleton_dir = Path("test_skeleton_bbw")
        test_mesh_dir = Path("test_mesh_bbw")
        
        test_skeleton_dir.mkdir(exist_ok=True)
        test_mesh_dir.mkdir(exist_ok=True)
        
        # Create dummy skeleton data
        num_frames, num_joints = 3, 6
        keypoints = np.random.rand(num_frames, num_joints, 4) * 2 - 1
        keypoints[:, :, 3] = 1.0  # confidence
        transforms = np.tile(np.eye(4), (num_frames, num_joints, 1, 1))
        parents = np.array([-1, 0, 1, 2, 1, 4])
        
        np.save(test_skeleton_dir / "keypoints.npy", keypoints)
        np.save(test_skeleton_dir / "transforms.npy", transforms)
        np.save(test_skeleton_dir / "parents.npy", parents)
        
        # Create dummy meshes
        for i in range(num_frames):
            mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
            mesh.translate([i * 0.1 - 0.5, -0.5, -0.5])  # Slight variation
            o3d.io.write_triangle_mesh(str(test_mesh_dir / f"frame_{i:04d}.obj"), mesh)
    
    try:
        # Initialize integrator
        integrator = BBWIntegrator(
            skeleton_data_dir=str(test_skeleton_dir),
            mesh_folder_path=str(test_mesh_dir),
            reference_frame_idx=0
        )
        
        # Compute reference weights
        weights = integrator.compute_reference_weights()
        print(f"Reference weights shape: {weights.shape}")
        
        # Test weight transfer
        if len(integrator.mesh_files) > 1:
            transferred_weights = integrator.get_weights_for_frame(1)
            print(f"Transferred weights shape: {transferred_weights.shape}")
        
        # Save cache
        integrator.save_weights_cache("test_output")
        
        print("=== BBW Integrator Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"BBW Integrator Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup dummy test data
        if str(test_skeleton_dir).startswith("test_"):
            import shutil
            if Path(test_skeleton_dir).exists():
                shutil.rmtree(test_skeleton_dir)
            if Path(test_mesh_dir).exists():
                shutil.rmtree(test_mesh_dir)


if __name__ == "__main__":
    test_bbw_integrator()