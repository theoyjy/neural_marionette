"""
Multi-Frame BBW Integrator

This module implements BBW skinning weight computation from multiple representative frames
to handle large pose variations in volumetric video sequences.

Features:
- Multi-frame weight learning from key poses
- Intelligent frame selection for weight computation
- Weight fusion strategies for better generalization
- Dynamic weight interpolation based on pose similarity
"""

import numpy as np
import torch
import os
import open3d as o3d
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import pickle
import time
from typing import Dict, List, Tuple, Optional, Union

# Import existing components
try:
    from bbw_integrator import BBWIntegrator
    BBW_INTEGRATOR_AVAILABLE = True
except ImportError:
    BBW_INTEGRATOR_AVAILABLE = False
    print("Warning: BBWIntegrator not available")


class MultiFrameBBWIntegrator:
    """
    Multi-frame BBW integrator for handling large pose variations
    
    This class extends BBW weight computation to multiple representative frames,
    providing better generalization for volumetric video with large pose changes.
    """
    
    def __init__(self, skeleton_data_dir: str, mesh_folder_path: str, 
                 num_key_frames: int = 5, frame_selection_method: str = "pose_diversity"):
        """
        Initialize multi-frame BBW integrator
        
        Args:
            skeleton_data_dir: Directory containing skeleton data
            mesh_folder_path: Directory containing mesh files
            num_key_frames: Number of key frames for weight learning
            frame_selection_method: Method for selecting key frames ("pose_diversity", "uniform", "manual")
        """
        self.skeleton_data_dir = Path(skeleton_data_dir)
        self.mesh_folder_path = Path(mesh_folder_path)
        self.num_key_frames = num_key_frames
        self.frame_selection_method = frame_selection_method
        
        # Load skeleton and mesh data
        self.load_skeleton_data()
        self.load_mesh_sequence()
        
        # Multi-frame BBW state
        self.key_frames = []
        self.key_frame_weights = {}
        self.fused_weights = None
        self.pose_features = None
        
        print(f"Multi-Frame BBW Integrator initialized:")
        print(f"  - Key frames: {self.num_key_frames}")
        print(f"  - Selection method: {self.frame_selection_method}")
        print(f"  - Total frames: {self.num_frames}")
    
    def load_skeleton_data(self):
        """Load skeleton data from files"""
        try:
            self.keypoints = np.load(self.skeleton_data_dir / 'keypoints.npy')
            self.transforms = np.load(self.skeleton_data_dir / 'transforms.npy')
            self.parents = np.load(self.skeleton_data_dir / 'parents.npy')
            
            self.num_frames, self.num_joints = self.keypoints.shape[0], self.keypoints.shape[1]
            
            print(f"Multi-Frame BBW: Loaded skeleton data:")
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
        
        print(f"Multi-Frame BBW: Found {len(self.mesh_files)} mesh files")
    
    def compute_pose_features(self):
        """
        Compute pose features for frame selection
        
        Returns:
            Pose feature matrix [num_frames, num_features]
        """
        print("Multi-Frame BBW: Computing pose features...")
        
        # Use joint positions as pose features
        pose_features = []
        
        for frame_idx in range(self.num_frames):
            # Get joint positions for this frame
            joint_positions = self.keypoints[frame_idx, :, :3].flatten()  # Remove confidence, flatten
            pose_features.append(joint_positions)
        
        pose_features = np.array(pose_features)
        self.pose_features = pose_features
        
        print(f"Multi-Frame BBW: Computed pose features: {pose_features.shape}")
        return pose_features
    
    def select_key_frames_pose_diversity(self, start_frame: int, end_frame: int):
        """
        Select key frames based on pose diversity using clustering
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            
        Returns:
            List of selected key frame indices
        """
        print(f"Multi-Frame BBW: Selecting key frames based on pose diversity...")
        
        # Limit to the interpolation range
        frame_range = list(range(start_frame, end_frame + 1))
        
        if len(frame_range) <= self.num_key_frames:
            # If we have fewer frames than desired key frames, use all
            selected_frames = frame_range
        else:
            # Get pose features for the frame range
            if self.pose_features is None:
                self.compute_pose_features()
            
            range_features = self.pose_features[frame_range]
            
            # Use K-means clustering to find diverse poses
            n_clusters = min(self.num_key_frames, len(frame_range))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(range_features)
            
            # Select one frame from each cluster (closest to centroid)
            selected_frames = []
            for cluster_id in range(n_clusters):
                cluster_frames = [frame_range[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                cluster_features = range_features[cluster_labels == cluster_id]
                
                # Find frame closest to cluster centroid
                centroid = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                closest_idx = np.argmin(distances)
                selected_frames.append(cluster_frames[closest_idx])
            
            # Always include start and end frames
            if start_frame not in selected_frames:
                selected_frames[0] = start_frame
            if end_frame not in selected_frames:
                selected_frames[-1] = end_frame
        
        selected_frames = sorted(list(set(selected_frames)))
        
        print(f"Multi-Frame BBW: Selected {len(selected_frames)} key frames: {selected_frames}")
        return selected_frames
    
    def select_key_frames_uniform(self, start_frame: int, end_frame: int):
        """
        Select key frames uniformly distributed in the range
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            
        Returns:
            List of selected key frame indices
        """
        print(f"Multi-Frame BBW: Selecting key frames uniformly...")
        
        if end_frame - start_frame + 1 <= self.num_key_frames:
            # Use all frames if we have fewer than desired
            selected_frames = list(range(start_frame, end_frame + 1))
        else:
            # Uniformly distribute key frames
            selected_frames = np.linspace(start_frame, end_frame, self.num_key_frames, dtype=int).tolist()
        
        selected_frames = sorted(list(set(selected_frames)))
        
        print(f"Multi-Frame BBW: Selected {len(selected_frames)} key frames: {selected_frames}")
        return selected_frames
    
    def select_key_frames(self, start_frame: int, end_frame: int, manual_frames: List[int] = None):
        """
        Select key frames based on the specified method
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            manual_frames: Manually specified frame indices (for manual method)
            
        Returns:
            List of selected key frame indices
        """
        if self.frame_selection_method == "pose_diversity":
            return self.select_key_frames_pose_diversity(start_frame, end_frame)
        elif self.frame_selection_method == "uniform":
            return self.select_key_frames_uniform(start_frame, end_frame)
        elif self.frame_selection_method == "manual" and manual_frames:
            # Filter manual frames to be within range
            filtered_frames = [f for f in manual_frames if start_frame <= f <= end_frame]
            return sorted(filtered_frames)
        else:
            # Default to uniform
            return self.select_key_frames_uniform(start_frame, end_frame)
    
    def compute_weights_for_key_frame(self, frame_idx: int):
        """
        Compute BBW weights for a single key frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Weight matrix for the frame
        """
        print(f"Multi-Frame BBW: Computing weights for key frame {frame_idx}")
        
        # Create a single-frame BBW integrator
        single_integrator = BBWIntegrator(
            skeleton_data_dir=str(self.skeleton_data_dir),
            mesh_folder_path=str(self.mesh_folder_path),
            reference_frame_idx=frame_idx
        )
        
        # Compute weights for this frame
        weights = single_integrator.compute_reference_weights()
        
        return weights
    
    def compute_multi_frame_weights(self, start_frame: int, end_frame: int, 
                                  manual_frames: List[int] = None):
        """
        Compute BBW weights from multiple key frames
        
        Args:
            start_frame: Start frame index
            end_frame: End frame index
            manual_frames: Optional manual frame selection
            
        Returns:
            Dictionary of key frame weights
        """
        print(f"Multi-Frame BBW: Computing weights from multiple key frames...")
        start_time = time.time()
        
        # Select key frames
        self.key_frames = self.select_key_frames(start_frame, end_frame, manual_frames)
        
        # Compute weights for each key frame
        self.key_frame_weights = {}
        
        for i, frame_idx in enumerate(self.key_frames):
            print(f"Multi-Frame BBW: Processing key frame {i+1}/{len(self.key_frames)} (frame {frame_idx})")
            
            try:
                weights = self.compute_weights_for_key_frame(frame_idx)
                self.key_frame_weights[frame_idx] = weights
                print(f"  - Computed weights: {weights.shape}")
                
            except Exception as e:
                print(f"  - Failed to compute weights for frame {frame_idx}: {e}")
                continue
        
        computation_time = time.time() - start_time
        print(f"Multi-Frame BBW: Computed weights for {len(self.key_frame_weights)} key frames in {computation_time:.2f}s")
        
        return self.key_frame_weights
    
    def fuse_weights_simple_average(self):
        """
        Fuse weights using simple averaging with topology adaptation
        
        Returns:
            Fused weight matrix
        """
        if not self.key_frame_weights:
            raise RuntimeError("No key frame weights computed")
        
        print("Multi-Frame BBW: Fusing weights using simple averaging...")
        
        # Get all weight matrices and analyze shapes
        weight_matrices = list(self.key_frame_weights.values())
        shapes = [w.shape for w in weight_matrices]
        
        # Find most common shape (topology)
        from collections import Counter
        shape_counts = Counter(shapes)
        target_shape = shape_counts.most_common(1)[0][0]
        
        print(f"  - Target topology shape: {target_shape}")
        print(f"  - Shape distribution: {dict(shape_counts)}")
        
        # Process weights for fusion
        processed_weights = []
        
        for i, (frame_idx, weights) in enumerate(zip(self.key_frames, weight_matrices)):
            if weights.shape == target_shape:
                # Direct use for matching topology
                processed_weights.append(weights)
                print(f"  - Frame {frame_idx}: Direct use (shape {weights.shape})")
            else:
                # Need topology adaptation
                print(f"  - Frame {frame_idx}: Adapting topology {weights.shape} -> {target_shape}")
                try:
                    adapted_weights = self.adapt_weights_topology(weights, target_shape, frame_idx)
                    if adapted_weights is not None:
                        processed_weights.append(adapted_weights)
                        print(f"    ✅ Successfully adapted")
                    else:
                        print(f"    ❌ Adaptation failed, skipping frame")
                except Exception as e:
                    print(f"    ❌ Adaptation error: {e}")
        
        if not processed_weights:
            raise RuntimeError("No usable weight matrices found after adaptation")
        
        # Simple average of processed weights
        fused_weights = np.mean(processed_weights, axis=0)
        
        # Renormalize to ensure each vertex sums to 1
        weight_sums = np.sum(fused_weights, axis=1, keepdims=True)
        fused_weights = fused_weights / (weight_sums + 1e-8)
        
        self.fused_weights = fused_weights
        
        print(f"Multi-Frame BBW: Fused weights shape: {fused_weights.shape}")
        print(f"  - Used {len(processed_weights)} key frames after adaptation")
        
        return fused_weights
    
    def adapt_weights_topology(self, source_weights: np.ndarray, target_shape: Tuple, 
                             source_frame_idx: int) -> Optional[np.ndarray]:
        """
        Adapt weights from source topology to target topology
        
        Args:
            source_weights: Source weight matrix [V_src, J]
            target_shape: Target shape (V_tgt, J)
            source_frame_idx: Source frame index for mesh loading
            
        Returns:
            Adapted weight matrix [V_tgt, J] or None if failed
        """
        target_vertices, target_joints = target_shape
        source_vertices, source_joints = source_weights.shape
        
        if source_joints != target_joints:
            print(f"    - Joint count mismatch: {source_joints} != {target_joints}")
            return None
        
        try:
            # Load source and target meshes for spatial correspondence
            source_mesh_path = self.mesh_files[source_frame_idx]
            source_mesh = o3d.io.read_triangle_mesh(str(source_mesh_path))
            source_mesh_vertices = np.asarray(source_mesh.vertices)
            
            # Find a frame with target topology
            target_frame_idx = None
            for frame_idx in self.key_frames:
                if frame_idx in self.key_frame_weights:
                    if self.key_frame_weights[frame_idx].shape == target_shape:
                        target_frame_idx = frame_idx
                        break
            
            if target_frame_idx is None:
                print(f"    - No reference frame with target topology found")
                return None
            
            target_mesh_path = self.mesh_files[target_frame_idx]
            target_mesh = o3d.io.read_triangle_mesh(str(target_mesh_path))
            target_mesh_vertices = np.asarray(target_mesh.vertices)
            
            # Use nearest neighbor mapping for weight transfer
            from sklearn.neighbors import NearestNeighbors
            
            # Build nearest neighbor model on source vertices
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(source_mesh_vertices)
            distances, indices = nbrs.kneighbors(target_mesh_vertices)
            
            # Transfer weights using distance-weighted interpolation
            adapted_weights = np.zeros((target_vertices, target_joints))
            
            for i in range(target_vertices):
                # Get nearest neighbors and their distances
                neighbor_indices = indices[i]
                neighbor_distances = distances[i]
                
                # Avoid division by zero
                neighbor_distances = np.maximum(neighbor_distances, 1e-8)
                
                # Inverse distance weighting
                inv_distances = 1.0 / neighbor_distances
                weight_sum = np.sum(inv_distances)
                
                # Weighted combination of neighbor weights
                for j, neighbor_idx in enumerate(neighbor_indices):
                    if neighbor_idx < source_vertices:
                        weight_factor = inv_distances[j] / weight_sum
                        adapted_weights[i] += weight_factor * source_weights[neighbor_idx]
            
            # Renormalize adapted weights
            weight_sums = np.sum(adapted_weights, axis=1, keepdims=True)
            adapted_weights = adapted_weights / (weight_sums + 1e-8)
            
            return adapted_weights
            
        except Exception as e:
            print(f"    - Adaptation failed: {e}")
            return None
    
    def fuse_weights_pose_aware(self, target_frame: int):
        """
        Fuse weights using pose-aware weighting with topology adaptation
        
        Args:
            target_frame: Target frame for pose-aware weighting
            
        Returns:
            Fused weight matrix
        """
        if not self.key_frame_weights:
            raise RuntimeError("No key frame weights computed")
        
        print(f"Multi-Frame BBW: Fusing weights with pose-aware weighting for frame {target_frame}...")
        
        # Determine target topology (use most common or target frame topology)
        target_mesh_path = self.mesh_files[target_frame]
        target_mesh = o3d.io.read_triangle_mesh(str(target_mesh_path))
        target_vertices = len(target_mesh.vertices)
        target_shape = (target_vertices, self.num_joints)
        
        print(f"  - Target topology for frame {target_frame}: {target_shape}")
        
        # Compute pose similarities
        if self.pose_features is None:
            self.compute_pose_features()
        
        target_pose = self.pose_features[target_frame]
        
        # Process weights with topology adaptation and pose weighting
        similarity_weights = []
        adapted_weights = []
        
        for frame_idx in self.key_frames:
            if frame_idx in self.key_frame_weights:
                key_pose = self.pose_features[frame_idx]
                weights = self.key_frame_weights[frame_idx]
                
                # Calculate pose similarity
                similarity = np.exp(-np.linalg.norm(target_pose - key_pose)**2 / (2 * 0.1**2))
                
                # Adapt topology if needed
                if weights.shape == target_shape:
                    # Direct use
                    adapted_weights.append(weights)
                    similarity_weights.append(similarity)
                else:
                    # Adapt topology
                    try:
                        adapted = self.adapt_weights_topology(weights, target_shape, frame_idx)
                        if adapted is not None:
                            adapted_weights.append(adapted)
                            similarity_weights.append(similarity)
                    except Exception as e:
                        print(f"  - Failed to adapt frame {frame_idx}: {e}")
                        continue
        
        if not similarity_weights:
            raise RuntimeError("No valid weight matrices for pose-aware fusion")
        
        # Normalize similarity weights
        similarity_weights = np.array(similarity_weights)
        similarity_weights = similarity_weights / (np.sum(similarity_weights) + 1e-8)
        
        print(f"  - Similarity weights: {similarity_weights}")
        
        # Weighted average with adapted weights
        fused_weights = np.zeros(target_shape)
        for i, weights in enumerate(adapted_weights):
            fused_weights += similarity_weights[i] * weights
        
        # Renormalize
        weight_sums = np.sum(fused_weights, axis=1, keepdims=True)
        fused_weights = fused_weights / (weight_sums + 1e-8)
        
        return fused_weights
    
    def get_weights_for_frame(self, frame_idx: int, fusion_method: str = "simple"):
        """
        Get weights for a specific frame using multi-frame fusion
        
        Args:
            frame_idx: Target frame index
            fusion_method: Weight fusion method ("simple" or "pose_aware")
            
        Returns:
            Weight matrix for the frame
        """
        if not self.key_frame_weights:
            raise RuntimeError("Key frame weights not computed")
        
        # If it's one of our key frames, return directly
        if frame_idx in self.key_frame_weights:
            return self.key_frame_weights[frame_idx]
        
        # Otherwise, fuse weights
        if fusion_method == "pose_aware":
            return self.fuse_weights_pose_aware(frame_idx)
        else:
            if self.fused_weights is None:
                self.fuse_weights_simple_average()
            return self.fused_weights
    
    def save_multi_frame_cache(self, output_dir: str):
        """Save multi-frame weights cache"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'key_frames': self.key_frames,
            'key_frame_weights': self.key_frame_weights,
            'fused_weights': self.fused_weights,
            'pose_features': self.pose_features,
            'num_key_frames': self.num_key_frames,
            'frame_selection_method': self.frame_selection_method,
            'num_joints': self.num_joints,
            'method': 'Multi_Frame_BBW'
        }
        
        cache_file = output_path / 'multi_frame_bbw_cache.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Multi-Frame BBW: Cache saved to {cache_file}")
    
    def load_multi_frame_cache(self, cache_path: str) -> bool:
        """Load multi-frame weights from cache"""
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.key_frames = cache_data['key_frames']
            self.key_frame_weights = cache_data['key_frame_weights']
            self.fused_weights = cache_data['fused_weights']
            self.pose_features = cache_data['pose_features']
            
            print(f"Multi-Frame BBW: Cache loaded from {cache_path}")
            return True
            
        except Exception as e:
            print(f"Multi-Frame BBW: Failed to load cache: {e}")
            return False


def test_multi_frame_bbw():
    """Test multi-frame BBW integrator"""
    print("=== Testing Multi-Frame BBW Integrator ===")
    
    # Test configuration
    test_skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
    test_mesh_dir = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    if not os.path.exists(test_skeleton_dir):
        print("Real test data not found, skipping test")
        return False
    
    try:
        # Initialize multi-frame BBW integrator
        integrator = MultiFrameBBWIntegrator(
            skeleton_data_dir=test_skeleton_dir,
            mesh_folder_path=test_mesh_dir,
            num_key_frames=5,
            frame_selection_method="pose_diversity"
        )
        
        # Test on a small range
        start_frame, end_frame = 0, 10
        
        # Compute multi-frame weights
        key_weights = integrator.compute_multi_frame_weights(start_frame, end_frame)
        print(f"Key frame weights computed: {len(key_weights)} frames")
        
        # Test weight fusion
        fused_weights = integrator.fuse_weights_simple_average()
        print(f"Fused weights shape: {fused_weights.shape}")
        
        # Test pose-aware fusion
        target_frame = 5
        pose_aware_weights = integrator.get_weights_for_frame(target_frame, "pose_aware")
        print(f"Pose-aware weights shape: {pose_aware_weights.shape}")
        
        # Save cache
        integrator.save_multi_frame_cache("test_multi_frame_output")
        
        print("=== Multi-Frame BBW Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"Multi-Frame BBW Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_multi_frame_bbw()