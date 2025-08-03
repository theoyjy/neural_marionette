"""
BBW Enhanced Interpolator

This module creates a BBW-enhanced version of the VolumetricInterpolator
that integrates seamlessly with the existing neural marionette pipeline.

Features:
- Drop-in replacement for existing interpolators
- BBW-based automatic skinning
- Maintains compatibility with existing API
- Enhanced skinning quality for mesh interpolation
"""

import numpy as np
import torch
import os
import open3d as o3d
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import time
from typing import Dict, List, Tuple, Optional

# Import existing components
try:
    from Interpolate import VolumetricInterpolator
    INTERPOLATOR_AVAILABLE = True
except ImportError:
    INTERPOLATOR_AVAILABLE = False
    print("Warning: VolumetricInterpolator not available")

try:
    from bbw_integrator import BBWIntegrator
    BBW_INTEGRATOR_AVAILABLE = True
except ImportError:
    BBW_INTEGRATOR_AVAILABLE = False
    print("Warning: BBWIntegrator not available")

try:
    from bbw_multi_frame_integrator import MultiFrameBBWIntegrator
    MULTI_FRAME_BBW_AVAILABLE = True
except ImportError:
    MULTI_FRAME_BBW_AVAILABLE = False
    print("Warning: MultiFrameBBWIntegrator not available")


class BBWEnhancedInterpolator(VolumetricInterpolator):
    """
    BBW-enhanced volumetric interpolator
    
    This class extends VolumetricInterpolator with BBW-based automatic skinning
    while maintaining full compatibility with the existing pipeline.
    """
    
    def __init__(self, skeleton_data_dir, mesh_folder_path, weights_path=None, 
                 use_bbw=True, bbw_reference_frame=0, use_multi_frame=True, 
                 num_key_frames=5, frame_selection_method="pose_diversity"):
        """
        Initialize BBW-enhanced interpolator
        
        Args:
            skeleton_data_dir: Directory containing skeleton data
            mesh_folder_path: Directory containing mesh files  
            weights_path: Optional precomputed weights path
            use_bbw: Whether to use BBW method
            bbw_reference_frame: Reference frame for BBW computation (single-frame mode)
            use_multi_frame: Whether to use multi-frame BBW learning
            num_key_frames: Number of key frames for multi-frame learning
            frame_selection_method: Key frame selection method ("pose_diversity", "uniform")
        """
        
        # Initialize parent class
        super().__init__(skeleton_data_dir, mesh_folder_path, weights_path)
        
        # BBW-specific initialization
        self.use_bbw = use_bbw and (BBW_INTEGRATOR_AVAILABLE or MULTI_FRAME_BBW_AVAILABLE)
        self.bbw_reference_frame = bbw_reference_frame
        self.use_multi_frame = use_multi_frame and MULTI_FRAME_BBW_AVAILABLE
        self.num_key_frames = num_key_frames
        self.frame_selection_method = frame_selection_method
        
        self.bbw_integrator = None
        self.multi_frame_integrator = None
        self.bbw_weights_computed = False
        self.current_interpolation_range = None
        
        print(f"BBW Enhanced Interpolator initialized:")
        print(f"  - Use BBW: {self.use_bbw}")
        print(f"  - Multi-frame mode: {self.use_multi_frame}")
        if self.use_multi_frame:
            print(f"  - Key frames: {self.num_key_frames}")
            print(f"  - Selection method: {self.frame_selection_method}")
        else:
            print(f"  - Reference frame: {self.bbw_reference_frame}")
        
        # Initialize BBW integrator if enabled
        if self.use_bbw:
            self.initialize_bbw()
    
    def initialize_bbw(self):
        """Initialize BBW integrator"""
        try:
            if self.use_multi_frame:
                # Initialize multi-frame BBW integrator
                self.multi_frame_integrator = MultiFrameBBWIntegrator(
                    skeleton_data_dir=str(self.skeleton_data_dir),
                    mesh_folder_path=str(self.mesh_folder_path),
                    num_key_frames=self.num_key_frames,
                    frame_selection_method=self.frame_selection_method
                )
                print("BBW Enhanced: Multi-frame BBW integrator initialized successfully")
            else:
                # Initialize single-frame BBW integrator
                self.bbw_integrator = BBWIntegrator(
                    skeleton_data_dir=str(self.skeleton_data_dir),
                    mesh_folder_path=str(self.mesh_folder_path),
                    reference_frame_idx=self.bbw_reference_frame,
                    use_bbw=True
                )
                print("BBW Enhanced: Single-frame BBW integrator initialized successfully")
            
        except Exception as e:
            print(f"BBW Enhanced: Failed to initialize BBW integrator: {e}")
            self.use_bbw = False
    
    def compute_bbw_skinning_weights(self, start_frame=None, end_frame=None, force_recompute=False):
        """
        Compute BBW skinning weights
        
        Args:
            start_frame: Start frame for multi-frame learning (None for single-frame)
            end_frame: End frame for multi-frame learning (None for single-frame)
            force_recompute: Force recomputation even if weights exist
            
        Returns:
            True if successful, False otherwise
        """
        if not self.use_bbw:
            print("BBW Enhanced: BBW not enabled, skipping weight computation")
            return False
            
        # Check if we need to recompute for different interpolation range
        current_range = (start_frame, end_frame) if start_frame is not None and end_frame is not None else None
        if (self.bbw_weights_computed and not force_recompute and 
            self.current_interpolation_range == current_range):
            print("BBW Enhanced: BBW weights already computed for this range")
            return True
        
        try:
            print("BBW Enhanced: Computing BBW skinning weights...")
            start_time = time.time()
            
            if self.use_multi_frame and start_frame is not None and end_frame is not None:
                # Multi-frame BBW computation
                print(f"BBW Enhanced: Using multi-frame learning for range [{start_frame}, {end_frame}]")
                
                key_weights = self.multi_frame_integrator.compute_multi_frame_weights(
                    start_frame, end_frame
                )
                
                # Use fused weights as default
                reference_weights = self.multi_frame_integrator.fuse_weights_simple_average()
                
                # Store additional multi-frame information
                self.key_frame_weights = key_weights
                self.key_frames = self.multi_frame_integrator.key_frames
                
                print(f"BBW Enhanced: Multi-frame learning completed")
                print(f"  - Key frames used: {len(self.key_frames)}")
                print(f"  - Key frame indices: {self.key_frames}")
                
            else:
                # Single-frame BBW computation
                print(f"BBW Enhanced: Using single-frame learning")
                reference_weights = self.bbw_integrator.compute_reference_weights(force_recompute)
            
            # Store in the format expected by parent class
            self.skinning_weights = reference_weights
            self.reference_frame_idx = self.bbw_reference_frame
            self.current_interpolation_range = current_range
            
            # Mark as computed
            self.bbw_weights_computed = True
            
            computation_time = time.time() - start_time
            print(f"BBW Enhanced: BBW weights computed in {computation_time:.2f}s")
            print(f"  - Weight matrix shape: {reference_weights.shape}")
            
            return True
            
        except Exception as e:
            print(f"BBW Enhanced: Failed to compute BBW weights: {e}")
            return False
    
    def get_weights_for_frame(self, frame_idx, fusion_method="pose_aware"):
        """
        Get skinning weights for a specific frame
        
        Args:
            frame_idx: Frame index
            fusion_method: Weight fusion method ("simple" or "pose_aware") for multi-frame mode
            
        Returns:
            Weight matrix for the frame
        """
        if not self.use_bbw or not self.bbw_weights_computed:
            # Fallback to parent class method
            return super().get_weights_for_frame(frame_idx) if hasattr(super(), 'get_weights_for_frame') else None
        
        try:
            if self.use_multi_frame:
                # Use multi-frame BBW integrator
                weights = self.multi_frame_integrator.get_weights_for_frame(frame_idx, fusion_method)
                return weights
            else:
                # Use single-frame BBW integrator to get/transfer weights
                weights = self.bbw_integrator.get_weights_for_frame(frame_idx)
                return weights
            
        except Exception as e:
            print(f"BBW Enhanced: Failed to get weights for frame {frame_idx}: {e}")
            # Fallback to parent class or default weights
            return self.skinning_weights if self.skinning_weights is not None else None
    
    def initialize_skinning(self, optimization_frames=None, regularization_lambda=0.01, max_iter=1000, 
                          start_frame=None, end_frame=None):
        """
        Initialize skinning system with BBW enhancement
        
        This method overrides the parent class to use BBW when available.
        
        Args:
            optimization_frames: Frames for optimization (legacy parameter)
            regularization_lambda: Regularization parameter (legacy parameter)
            max_iter: Maximum iterations (legacy parameter)
            start_frame: Start frame for multi-frame BBW learning
            end_frame: End frame for multi-frame BBW learning
        """
        print("BBW Enhanced: Initializing skinning system...")
        
        if self.use_bbw:
            # Use BBW method
            success = self.compute_bbw_skinning_weights(start_frame, end_frame)
            if success:
                mode = "multi-frame" if self.use_multi_frame else "single-frame"
                print(f"BBW Enhanced: Skinning initialized with BBW method ({mode})")
                return True
            else:
                print("BBW Enhanced: BBW failed, falling back to original method")
        
        # Fallback to parent class method
        if hasattr(super(), 'initialize_skinning'):
            return super().initialize_skinning(optimization_frames, regularization_lambda, max_iter)
        else:
            print("BBW Enhanced: No fallback skinning method available")
            return False
    
    def generate_interpolated_frames(self, frame_start, frame_end, num_interpolate, 
                                   max_optimize_frames=5, optimize_weights=True, 
                                   output_dir=None, debug_frames=None, smooth_mesh=False, subdivide_iter=3,
                                   use_vertex_colors=False, save_npy_files=False, save_standard_obj=True):
        """
        Generate interpolated frames with BBW enhancement
        
        This method ensures BBW weights are computed before interpolation.
        """
        print(f"BBW Enhanced: Generating {num_interpolate} interpolated frames between {frame_start} and {frame_end}")
        
        # Ensure BBW weights are computed if enabled
        if self.use_bbw and not self.bbw_weights_computed:
            print("BBW Enhanced: Computing BBW weights before interpolation...")
            self.compute_bbw_skinning_weights(frame_start, frame_end)
        
        # Call parent class method
        return super().generate_interpolated_frames(
            frame_start, frame_end, num_interpolate,
            max_optimize_frames, optimize_weights,
            output_dir, debug_frames, smooth_mesh, subdivide_iter,
            use_vertex_colors, save_npy_files, save_standard_obj
        )
    
    def save_bbw_weights(self, output_dir):
        """Save BBW weights for future use"""
        if self.use_bbw:
            if self.use_multi_frame and self.multi_frame_integrator:
                self.multi_frame_integrator.save_multi_frame_cache(output_dir)
                print(f"BBW Enhanced: Multi-frame weights saved to {output_dir}")
            elif self.bbw_integrator:
                self.bbw_integrator.save_weights_cache(output_dir)
                print(f"BBW Enhanced: Single-frame weights saved to {output_dir}")
        else:
            print("BBW Enhanced: No BBW weights to save")
    
    def load_bbw_weights(self, cache_path):
        """Load BBW weights from cache"""
        if self.use_bbw and self.bbw_integrator:
            success = self.bbw_integrator.load_weights_cache(cache_path)
            if success:
                self.skinning_weights = self.bbw_integrator.reference_weights
                self.bbw_weights_computed = True
                print("BBW Enhanced: BBW weights loaded from cache")
                return True
        
        print("BBW Enhanced: Failed to load BBW weights from cache")
        return False
    
    def get_bbw_info(self):
        """Get information about BBW configuration"""
        info = {
            'bbw_enabled': self.use_bbw,
            'bbw_computed': self.bbw_weights_computed,
            'multi_frame_mode': self.use_multi_frame,
            'reference_frame': self.bbw_reference_frame,
            'weights_shape': self.skinning_weights.shape if self.skinning_weights is not None else None,
            'num_joints': self.num_joints if hasattr(self, 'num_joints') else None,
            'interpolation_range': self.current_interpolation_range
        }
        
        if self.use_multi_frame:
            info.update({
                'num_key_frames': self.num_key_frames,
                'frame_selection_method': self.frame_selection_method,
                'key_frames': getattr(self, 'key_frames', None)
            })
        
        if self.bbw_integrator:
            info.update({
                'skeleton_alignment': self.bbw_integrator.skeleton_alignment_info,
                'mesh_files_count': len(self.bbw_integrator.mesh_files)
            })
        elif self.multi_frame_integrator:
            info.update({
                'mesh_files_count': len(self.multi_frame_integrator.mesh_files)
            })
        
        return info


def test_bbw_enhanced_interpolator():
    """Test BBW enhanced interpolator"""
    print("=== Testing BBW Enhanced Interpolator ===")
    
    # Test with dummy data first
    test_mesh_dir = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    test_skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
    
    # Check if real data exists
    if not os.path.exists(test_skeleton_dir):
        print("Creating dummy test data...")
        
        # Create dummy data for testing
        test_skeleton_dir = Path("test_bbw_enhanced")
        test_mesh_dir = Path("test_bbw_enhanced")
        
        test_skeleton_dir.mkdir(exist_ok=True)
        test_mesh_dir.mkdir(exist_ok=True)
        
        # Create dummy skeleton data
        num_frames, num_joints = 5, 8
        keypoints = np.random.rand(num_frames, num_joints, 4) * 2 - 1
        keypoints[:, :, 3] = 1.0  # confidence
        transforms = np.tile(np.eye(4), (num_frames, num_joints, 1, 1))
        parents = np.array([-1, 0, 1, 2, 1, 4, 5, 6])
        
        np.save(test_skeleton_dir / "keypoints.npy", keypoints)
        np.save(test_skeleton_dir / "transforms.npy", transforms)
        np.save(test_skeleton_dir / "parents.npy", parents)
        
        # Create dummy meshes
        for i in range(num_frames):
            mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
            mesh.translate([i * 0.2 - 1.0, -0.5, -0.5])  # Progressive deformation
            o3d.io.write_triangle_mesh(str(test_mesh_dir / f"frame_{i:04d}.obj"), mesh)
    
    try:
        # Initialize BBW enhanced interpolator
        interpolator = BBWEnhancedInterpolator(
            skeleton_data_dir=str(test_skeleton_dir),
            mesh_folder_path=str(test_mesh_dir),
            use_bbw=True,
            bbw_reference_frame=0
        )
        
        # Get BBW info
        bbw_info = interpolator.get_bbw_info()
        print("BBW Configuration:")
        for key, value in bbw_info.items():
            print(f"  - {key}: {value}")
        
        # Initialize skinning
        success = interpolator.initialize_skinning()
        print(f"Skinning initialization: {'Success' if success else 'Failed'}")
        
        # Test weight retrieval
        if interpolator.bbw_weights_computed:
            weights_frame_0 = interpolator.get_weights_for_frame(0)
            print(f"Frame 0 weights shape: {weights_frame_0.shape}")
            
            if len(interpolator.mesh_files) > 1:
                weights_frame_1 = interpolator.get_weights_for_frame(1)
                print(f"Frame 1 weights shape: {weights_frame_1.shape}")
        
        # Save weights
        interpolator.save_bbw_weights("test_output_enhanced")
        
        print("=== BBW Enhanced Interpolator Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"BBW Enhanced Interpolator Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup dummy test data
        if str(test_skeleton_dir).startswith("test_"):
            import shutil
            if Path(test_skeleton_dir).exists():
                shutil.rmtree(test_skeleton_dir)


if __name__ == "__main__":
    test_bbw_enhanced_interpolator()