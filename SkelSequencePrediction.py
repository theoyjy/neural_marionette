import torch
import numpy as np
import os
import pickle
import open3d as o3d
from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import crop_sequence, voxelize, episodic_normalization
import glob
from pathlib import Path
from SkelVisualizer import visualize_skeleton
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class SequenceSkeletonPredictor:
    def __init__(self, checkpoint_path, opt_path):
        """
        Initialize Neural Marionette model
        
        Args:
            checkpoint_path: Pretrained model path
            opt_path: Configuration file path
        """
        start_time = time.time()
        
        # Load configuration
        with open(opt_path, 'rb') as f:
            self.opt = pickle.load(f)
        
        # Load model
        checkpoint = torch.load(checkpoint_path)
        self.network = NeuralMarionette(self.opt).cuda()
        self.network.load_state_dict(checkpoint)
        self.network.eval()
        self.network.anneal(1)  # Enable affinity extraction
        
        model_load_time = time.time() - start_time
        print(f"Model loaded successfully, keypoint count: {self.opt.nkeypoints}")
        print(f"Model loading time: {model_load_time:.2f} seconds")
    
    def process_single_mesh(self, mesh_file, frame_idx, total_frames):
        """
        Process single mesh file (for multi-threading)
        
        Args:
            mesh_file: Mesh file path
            frame_idx: Frame index
            total_frames: Total frame count
            
        Returns:
            dict: Dictionary containing processing results
        """
        try:
            print(f"Processing file {frame_idx+1}/{total_frames}: {os.path.basename(mesh_file)}")
            
            # Load mesh
            if mesh_file.endswith('.obj') or mesh_file.endswith('.ply'):
                mesh = o3d.io.read_triangle_mesh(mesh_file)
            else:
                # Try to load as point cloud
                pcd = o3d.io.read_point_cloud(mesh_file)
                points = np.asarray(pcd.points)
            
            if 'mesh' in locals() and len(mesh.vertices) > 0:
                points = np.asarray(mesh.vertices)
                mesh_data = mesh
            elif 'pcd' in locals() and len(pcd.points) > 0:
                points = np.asarray(pcd.points)
                mesh_data = pcd
            else:
                raise ValueError(f"Cannot load file: {mesh_file}")
            
            # Normalize point cloud (mimicking original code processing)
            points_norm = episodic_normalization(points[None], scale=1.0, x_trans=0.0, z_trans=0.0)[0]
            
            # Voxelization
            try:
                voxel = voxelize(points_norm, (self.opt.grid_size,) * 3, is_binarized=True)
            except Exception as e:
                print(f"Voxelization failed: {e}")
                raise
            
            return {
                'frame_idx': frame_idx,
                'mesh': mesh_data,
                'points_norm': points_norm,
                'voxel': voxel,
                'success': True
            }
            
        except Exception as e:
            print(f"Failed to process file {mesh_file}: {e}")
            return {
                'frame_idx': frame_idx,
                'mesh': None,
                'points_norm': None,
                'voxel': None,
                'success': False,
                'error': str(e)
            }
    
    def load_mesh_sequence(self, mesh_folder, file_pattern="*.obj", max_frames=None):
        """
        Load mesh sequence and convert to voxels (multi-threaded version)
        
        Args:
            mesh_folder: Folder path containing mesh files
            file_pattern: File matching pattern, such as "*.obj", "frame_*.ply"
            max_frames: Maximum frame count limit
        
        Returns:
            voxel_sequence: (T, grid_size, grid_size, grid_size)
            mesh_sequence: Original mesh data list
        """
        start_time = time.time()
        
        mesh_files = sorted(glob.glob(os.path.join(mesh_folder, file_pattern)))
        
        # if max_frames:
        #     mesh_files = mesh_files[:max_frames]
        
        if len(mesh_files) == 0:
            raise ValueError(f"No files matching {file_pattern} found in {mesh_folder}")
        
        print(f"Found {len(mesh_files)} mesh files")
        print(f"Starting multi-threaded processing...")
        
        # Multi-threaded processing
        max_workers = min(8, len(mesh_files))  # Limit maximum thread count
        print(f"  - Using {max_workers} threads")
        
        voxel_sequence = []
        mesh_sequence = []
        points_sequence = []
        
        # List to store results (sorted by frame index)
        results = [None] * len(mesh_files)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self.process_single_mesh, mesh_file, i, len(mesh_files)): i 
                for i, mesh_file in enumerate(mesh_files)
            }
            
            # Collect results
            for future in as_completed(future_to_idx):
                result = future.result()
                if result['success']:
                    results[result['frame_idx']] = result
                    print(f"Completed file {result['frame_idx']+1}/{len(mesh_files)}")
                else:
                    print(f"File {result['frame_idx']+1} processing failed: {result.get('error', 'Unknown error')}")
        
        # Sort results in order
        for i, result in enumerate(results):
            if result is None or not result['success']:
                raise ValueError(f"File {i+1} processing failed")
            
            mesh_sequence.append(result['mesh'])
            points_sequence.append(result['points_norm'])
            voxel_sequence.append(result['voxel'])
        
        # Convert to torch tensor
        voxel_sequence = torch.from_numpy(np.stack(voxel_sequence, axis=0)).float().cuda()
        
        processing_time = time.time() - start_time
        print(f"Multi-threaded processing completed!")
        print(f"  - Voxel sequence shape: {voxel_sequence.shape}")
        print(f"  - Processing time: {processing_time:.2f} seconds")
        print(f"  - Average per frame: {processing_time/len(mesh_files):.3f} seconds")
        
        return voxel_sequence, mesh_sequence, points_sequence

    def predict_skeleton_sequence(self, voxel_sequence):
        """
        Predict skeleton for entire sequence
        
        Args:
            voxel_sequence: (T, grid_size, grid_size, grid_size)
        
        Returns:
            keypoints: (1, T, K, 4) - joint coordinates and confidence
            transforms: (T, K, 4, 4) - transformation matrices, local coordinate system for each joint
            affinity: Skeleton connection relationships
            parents: Parent-child relationships
        """
        start_time = time.time()
        print(f"Start neural network prediction...")
        print(f"  - Input shape: {voxel_sequence.shape}")
        
        with torch.no_grad():
            # Process entire sequence at once
            detector_log = self.network.kypt_detector(voxel_sequence[None])  # Add batch dimension
            keypoints = detector_log['keypoints']
            affinity = detector_log['affinity']
            
            # Maintain consistent visibility (similar to original code)
            keypoints[:, 1:, :, -1] = keypoints[:, :1, :, -1].expand(-1, voxel_sequence.size(0) - 1, -1)
            
            # Encode dynamics
            dyna_log = self.network.dyna_module.encode(keypoints, affinity)
            R = dyna_log['R'][0]  # (T, K, 3, 3)
            
            # Get structural information
            A = self.network.dyna_module.A
            priority = self.network.dyna_module.priority
            parents = self.network.dyna_module.parents
            
            # Build 4x4 transformation matrices
            pos = keypoints[0, :, :, :3][..., None]  # (T, K, 3, 1)
            T4x4 = torch.cat([R, pos], dim=-1)  # (T, K, 3, 4)
            homo = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(R.device)[None, None, None].expand(
                R.size(0), R.size(1), -1, -1)
            T4x4 = torch.cat([T4x4, homo], dim=-2)  # (T, K, 4, 4)
            
            prediction_time = time.time() - start_time
            print(f"Neural network prediction completed!")
            print(f"  - Prediction time: {prediction_time:.2f} seconds")
            print(f"  - Keypoints shape: {keypoints.shape}")
            print(f"  - Transforms shape: {T4x4.shape}")
            
            return {
                'keypoints': keypoints,
                'transforms': T4x4,
                'affinity': affinity,
                'parents': parents.cpu().numpy(),
                'priority_values': priority.values.cpu().numpy(),  # Extract values
                'priority_indices': priority.indices.cpu().numpy(),  # Extract indices
                'A': A,
                'rotations': R
            }
    
    def save_skeleton_results(self, results, output_dir, points_sequence=None):
        """
        Save skeleton prediction results
        
        Args:
            results: Output of predict_skeleton_sequence
            output_dir: Output directory
            points_sequence: Original point cloud sequence (for visualization)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numerical results
        np.save(os.path.join(output_dir, 'keypoints.npy'), results['keypoints'][0].cpu().numpy())
        np.save(os.path.join(output_dir, 'transforms.npy'), results['transforms'].cpu().numpy())
        np.save(os.path.join(output_dir, 'parents.npy'), results['parents'])
        np.save(os.path.join(output_dir, 'affinity.npy'), results['affinity'].cpu().numpy())
        np.save(os.path.join(output_dir, 'priority_values.npy'), results['priority_values'])
        np.save(os.path.join(output_dir, 'priority_indices.npy'), results['priority_indices'])
        np.save(os.path.join(output_dir, 'A.npy'), results['A'].cpu().numpy())
        np.save(os.path.join(output_dir, 'rotations.npy'), results['rotations'].cpu().numpy())

        # save normalized points
        # if points_sequence is not None:
            # np.save(os.path.join(output_dir, 'points_sequence.npy'), np.stack(points_sequence, axis=0))
            # self.visualize_skeleton_sequence(results, output_dir, points_sequence)
    
    def visualize_skeleton_sequence(self, results, output_dir, points_sequence, 
                                  vis_threshold=0.2, save_frames=True):
        """
        Visualize skeleton sequence
        """
        keypoints = results['keypoints'][0].cpu().numpy()  # (T, K, 4)
        parents = results['parents']
        
        # Generate joint colors
        np.random.seed(42)
        joint_colors = np.random.rand(keypoints.shape[1], 3)
        
        if save_frames:
            frames_dir = os.path.join(output_dir, 'skeleton_frames')
            os.makedirs(frames_dir, exist_ok=True)
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600, visible=not save_frames)
        
        for t in range(keypoints.shape[0]):
            vis.clear_geometries()
            print(f'Processing frame {t+1}/{keypoints.shape[0]}')

            # Add original point cloud
            if points_sequence:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_sequence[t])
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                vis.add_geometry(pcd)
                # print(f'min={np.min(points_sequence[t], axis=0)}, max={np.max(points_sequence[t], axis=0)}')
            
            # Add joints and bones
            kypts = keypoints[t, :, :3]
            alphas = keypoints[t, :, -1]
            print(f'joints num = {kypts.shape[0]} min={np.min(kypts, axis=0)}, max={np.max(kypts, axis=0)}')
            print(f'parents: {parents}')
            draw_count = 0
            for k in range(keypoints.shape[1]):
                if alphas[k] < vis_threshold:
                    continue
                draw_count += 1
                # Add joint sphere
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere.translate(kypts[k])
                sphere.paint_uniform_color(joint_colors[k])
                vis.add_geometry(sphere)
                
                # Add bone connections
                parent = parents[k]
                if parent != k and alphas[parent] >= vis_threshold:
                    # Create bone line
                    line_points = [kypts[parent], kypts[k]]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.paint_uniform_color([0, 0.8, 0])
                    vis.add_geometry(line_set)

            print(f'Draw joints: {draw_count}')
            if save_frames:
                # Save frame image
                img = vis.capture_screen_float_buffer(True)
                img = (np.asarray(img) * 255).astype(np.uint8)
                o3d.io.write_image(os.path.join(frames_dir, f'frame_{t:04d}.png'), 
                                 o3d.geometry.Image(img))
            else:
                # Interactive display
                vis.poll_events()
                vis.update_renderer()
        
        vis.destroy_window()
        print(f"Visualization completed, processed {keypoints.shape[0]} frames")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Skeleton sequence prediction")        
    parser.add_argument("--mesh_folder", type=str, default="D:/Code/VVEditor/Rafa_Approves_hd_4k", 
                       help="Input mesh folder path")
    parser.add_argument("--output_dir", type=str, default="output/skeleton_prediction", 
                       help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None, 
                       help="Maximum number of frames to process")
    parser.add_argument("--visualization", action="store_true", 
                       help="Enable visualization")
    
    args = parser.parse_args()
    
    # Configure paths
    exp_dir = 'pretrained/aist'
    checkpoint_path = os.path.join(exp_dir, 'aist_pretrained.pth')
    opt_path = os.path.join(exp_dir, 'opt.pickle')
    
    # Input sequence folder
    mesh_folder = args.mesh_folder
    skel_data_dir = args.output_dir
    visualize_dir = os.path.join(args.output_dir, 'visualization')

    # Create predictor
    predictor = SequenceSkeletonPredictor(checkpoint_path, opt_path)
    
    # Load mesh sequence
    voxel_sequence, mesh_sequence, points_sequence = predictor.load_mesh_sequence(
        mesh_folder, file_pattern="*.obj", max_frames=args.max_frames
    )
    
    # Predict skeleton
    print("Start predicting skeleton...")
    results = predictor.predict_skeleton_sequence(voxel_sequence)
    print("Skeleton prediction completed!")
    
    # Save results
    predictor.save_skeleton_results(results, skel_data_dir, points_sequence)

    if args.visualization:
        from SkelVisualizer import visualize_skeleton
        visualize_skeleton(skel_data_dir, visualize_dir)

    print("Processing completed!")

if __name__ == "__main__":
    main()