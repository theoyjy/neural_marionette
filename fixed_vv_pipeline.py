#!/usr/bin/env python3
"""
Fixed VolumetricVideo Interpolation Pipeline
===========================================

修复了DemBones的问题，专注于两个核心步骤：
1. NM生成骨骼 → 智能rest pose检测 → 骨骼引导网格统一
2. 统一网格 + 骨骼 → DemBones蒙皮预测 (无超时问题)

Usage:
python fixed_vv_pipeline.py "folder_path" --from_frame 5 --to_frame 15 --num_interp 5
"""

import argparse
import os
import glob
import time
import pickle
import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import voxelize
import py_dem_bones as pdb

# Import functions from GenerateSkel.py
from GenerateSkel import (
    load_voxel_from_mesh, 
    process_single_mesh,
    sanitize_parents,
    draw_skeleton,
    draw_skinning_colors,
    _as_colmajor,
    _as_rowmajor
)

def demBones_simple(frames_vertices: np.ndarray, parents: np.ndarray, nnz: int = 4, n_iters: int = 10):
    """
    简化的DemBones调用，避免超时和线程问题
    """
    F, N, _ = frames_vertices.shape
    K = parents.shape[0]
    
    print(f"DemBones: {F} frames, {N} vertices, {K} bones")
    print(f"Parameters: {n_iters} iterations, {nnz} weights/vertex")
    
    # 验证输入数据
    if not np.isfinite(frames_vertices).all():
        print("❌ Non-finite values in vertices")
        raise ValueError("vertices contains non-finite values")
    
    if frames_vertices.min() < -10 or frames_vertices.max() > 10:
        print("❌ Vertices out of reasonable range")
        raise ValueError("vertices out of range")
    
    # 验证parents数组
    if parents.min() < -1 or parents.max() >= K:
        print("❌ Invalid parent indices")
        raise ValueError("invalid parent indices")
    
    try:
        dem = pdb.DemBonesExtWrapper()
        
        # 先设置数据，再设置参数
        # 准备数据 - 确保数据类型和范围正确
        rest_pose = frames_vertices[0].T.copy().astype(np.float64)  # (3, N)
        anim_poses = frames_vertices.transpose(0,2,1).copy().astype(np.float64)  # (F,3,N)
        anim_poses = anim_poses.reshape(3, -1)  # (3, F·N)
        
        # 验证数据形状
        if rest_pose.shape != (3, N):
            raise ValueError(f"rest_pose shape {rest_pose.shape} != (3, {N})")
        if anim_poses.shape != (3, F*N):
            raise ValueError(f"anim_poses shape {anim_poses.shape} != (3, {F*N})")
        
        print(f"Rest pose: {rest_pose.shape}, range [{rest_pose.min():.3f}, {rest_pose.max():.3f}]")
        print(f"Anim poses: {anim_poses.shape}, range [{anim_poses.min():.3f}, {anim_poses.max():.3f}]")
        
        # 首先设置数据
        print("Setting data...")
        dem.set_rest_pose(rest_pose)
        dem.animated_poses = anim_poses
        dem.set_target_vertices('animated', anim_poses)
        dem.parents = parents.copy().astype(np.int32)
        
        # 然后设置参数 - 必须在设置数据之后
        print("Setting parameters...")
        dem.num_iterations = max(1, min(n_iters, 20))  # 限制迭代数
        dem.max_nonzeros_per_vertex = max(1, min(nnz, 4))  # 限制权重数
        dem.weights_smoothness = 1e-3  # 增加平滑度
        dem.weights_sparseness = 1e-5  # 适中的稀疏度
        
        # 设置收敛参数
        dem.tolerance = 1e-4  # 宽松的收敛条件
        
        # 验证设置
        print(f"Expected setup: {N} vertices, {K} bones, {F} frames")
        print(f"DemBones reports: {dem.num_vertices} vertices, {dem.num_bones} bones")
        
        if dem.num_vertices != N or dem.num_bones != K:
            print(f"❌ DemBones setup mismatch!")
            print(f"  Expected: {N} vertices, {K} bones")
            print(f"  Got: {dem.num_vertices} vertices, {dem.num_bones} bones")
            raise RuntimeError("DemBones setup validation failed")
        
        # 计算
        print("Running DemBones computation...")
        start_time = time.time()
        
        success = dem.compute()
        
        elapsed_time = time.time() - start_time
        
        if not success:
            print(f"❌ DemBones compute() returned False after {elapsed_time:.2f}s")
            raise RuntimeError("DemBones computation failed (returned False)")
        
        print(f"✓ DemBones completed in {elapsed_time:.2f}s")
        
        # 获取结果
        try:
            rest_pose_result = dem._dem_bones.get_rest_pose()
            if rest_pose_result.size == 0:
                raise RuntimeError("Empty rest pose result")
            rest_pose_result = _as_rowmajor(rest_pose_result)  # (N,3)
            
            weights = dem.get_weights()
            if weights.size == 0:
                raise RuntimeError("Empty weights result")
            weights = weights.T.copy()  # (N,K)
            
            # 验证和修复权重
            weights = np.maximum(weights, 0)  # 确保非负
            row_sums = weights.sum(axis=1, keepdims=True)
            zero_rows = row_sums.flatten() < 1e-8
            if zero_rows.any():
                print(f"Warning: {zero_rows.sum()} vertices have zero weights, fixing...")
                weights[zero_rows, 0] = 1.0  # 给第一个骨骼分配权重
                row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / row_sums
            
            # 获取变换矩阵
            transforms = dem.get_animated_transformation()  # (K*F, 4, 4)
            if transforms.size == 0:
                raise RuntimeError("Empty transforms result")
            transforms = transforms.reshape(F, K, 4, 4).astype(np.float32)
            
            print(f"✓ Results obtained:")
            print(f"  Rest pose: {rest_pose_result.shape}")
            print(f"  Weights: {weights.shape}, range [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"  Transforms: {transforms.shape}")
            
            return rest_pose_result, weights, transforms
            
        except Exception as e:
            print(f"❌ Error extracting results: {e}")
            raise RuntimeError(f"Failed to extract DemBones results: {e}")
        
    except Exception as e:
        print(f"❌ DemBones computation failed: {e}")
        raise

class FixedVVProcessor:
    """
    修复版VV处理器，专注于DemBones的稳定运行
    """
    
    def __init__(self, folder_path, output_dir=None):
        self.folder_path = folder_path
        self.output_dir = output_dir or os.path.join(folder_path, 'fixed_vv_processing')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据存储
        self.all_mesh_data = []
        self.rest_pose_idx = None
        self.unified_vertices = None
        self.skinning_weights = None
        self.bone_transforms = None
        self.parents = None
        
        # 加载模型
        self._load_neural_marionette()
    
    def _load_neural_marionette(self):
        """加载NeuralMarionette模型"""
        print("Loading NeuralMarionette...")
        
        exp_dir = 'pretrained/aist'
        opt_path = os.path.join(exp_dir, 'opt.pickle')
        with open(opt_path, 'rb') as f:
            opt = pickle.load(f)
        opt.Ttot = 1
        
        ckpt_path = os.path.join(exp_dir, 'aist_pretrained.pth')
        checkpoint = torch.load(ckpt_path)
        network = NeuralMarionette(opt).cuda()
        network.load_state_dict(checkpoint)
        network.eval()
        network.anneal(1)
        
        self.network = network
        self.opt = opt
        
        print("✓ NeuralMarionette loaded")
    
    def step1_process_frames(self, max_frames=30):
        """步骤1: 处理所有帧，限制数量避免DemBones超载"""
        print(f"\n=== STEP 1: Processing Frames (max {max_frames}) ===")
        
        obj_files = sorted(glob.glob(os.path.join(self.folder_path, "*.obj")))
        if not obj_files:
            raise ValueError(f"No .obj files found in {self.folder_path}")
        
        # 智能选择帧
        if len(obj_files) > max_frames:
            indices = np.linspace(0, len(obj_files)-1, max_frames, dtype=int)
            selected_files = [obj_files[i] for i in indices]
            print(f"Selected {max_frames} frames from {len(obj_files)} total")
        else:
            selected_files = obj_files
            print(f"Processing all {len(selected_files)} frames")
        
        # 处理每个文件
        for i, obj_file in enumerate(selected_files):
            print(f"Processing {i+1}/{len(selected_files)}: {os.path.basename(obj_file)}")
            
            try:
                mesh_data = process_single_mesh(obj_file, self.network, self.opt, self.output_dir)
                self.all_mesh_data.append(mesh_data)
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        if not self.all_mesh_data:
            raise RuntimeError("No meshes were successfully processed")
        
        print(f"✓ Processed {len(self.all_mesh_data)} frames")
        return len(self.all_mesh_data)
    
    def step2_detect_rest_pose(self):
        """步骤2: 智能检测最佳rest pose"""
        print("\n=== STEP 2: Detecting Best Rest Pose ===")
        
        scores = []
        for i, data in enumerate(self.all_mesh_data):
            joints = data['joints']
            
            # 简化的评分系统
            joint_spread = np.linalg.norm(joints - joints.mean(axis=0), axis=1).std()
            R_rotations = data['R']
            rotation_regularity = 1.0 / (np.mean([np.linalg.norm(R - np.eye(3)) for R in R_rotations]) + 0.1)
            
            score = joint_spread + rotation_regularity
            scores.append(score)
            
            print(f"  Frame {i:2d}: spread={joint_spread:.3f}, regularity={rotation_regularity:.3f}, score={score:.3f}")
        
        self.rest_pose_idx = np.argmax(scores)
        rest_data = self.all_mesh_data[self.rest_pose_idx]
        
        print(f"✓ Selected frame {self.rest_pose_idx} as rest pose")
        print(f"  Rest pose: {len(rest_data['pts_raw'])} vertices, {len(rest_data['joints'])} joints")
        
        return self.rest_pose_idx
    
    def step3_unify_topology(self, target_vertices=None):
        """步骤3: 统一拓扑，可选择减少顶点数"""
        print("\n=== STEP 3: Unifying Topology ===")
        
        if self.rest_pose_idx is None:
            raise RuntimeError("Must detect rest pose first")
        
        template_data = self.all_mesh_data[self.rest_pose_idx]
        template_pts = template_data['pts_norm']
        
        # 可选的顶点减少
        if target_vertices and len(template_pts) > target_vertices:
            print(f"Reducing vertices: {len(template_pts)} → {target_vertices}")
            indices = np.linspace(0, len(template_pts)-1, target_vertices, dtype=int)
            template_pts = template_pts[indices]
            self.vertex_indices = indices
        else:
            target_vertices = len(template_pts)
            self.vertex_indices = np.arange(target_vertices)
        
        print(f"Template: {target_vertices} vertices")
        
        # 统一所有帧
        unified_frames = []
        for i, data in enumerate(self.all_mesh_data):
            current_pts = data['pts_norm']
            
            if len(current_pts) == target_vertices and target_vertices == len(template_pts):
                unified_frames.append(current_pts)
                print(f"  Frame {i}: No remapping needed")
            else:
                # 使用最近邻重采样
                if hasattr(self, 'vertex_indices'):
                    # 使用预选的顶点索引
                    if len(current_pts) > max(self.vertex_indices):
                        remapped = current_pts[self.vertex_indices]
                    else:
                        # 回退到最近邻
                        tree = cKDTree(current_pts)
                        _, idx = tree.query(template_pts, k=1)
                        remapped = current_pts[idx]
                else:
                    tree = cKDTree(current_pts)
                    _, idx = tree.query(template_pts, k=1)
                    remapped = current_pts[idx]
                
                unified_frames.append(remapped)
                print(f"  Frame {i}: Remapped {len(current_pts)} → {target_vertices}")
        
        self.unified_vertices = np.stack(unified_frames, axis=0)
        print(f"✓ Unified topology: {self.unified_vertices.shape}")
        
        return self.unified_vertices.shape
    
    def step4_demBones_skinning(self):
        """步骤4: 使用DemBones计算蒙皮"""
        print("\n=== STEP 4: DemBones Skinning ===")
        
        if self.unified_vertices is None:
            raise RuntimeError("Must unify topology first")
        
        # 获取父节点
        rest_data = self.all_mesh_data[self.rest_pose_idx]
        self.parents = sanitize_parents(rest_data['parents'])
        
        F, N, _ = self.unified_vertices.shape
        K = len(self.parents)
        
        print(f"Input: {F} frames, {N} vertices, {K} bones")
        
        # 使用保守参数确保DemBones成功
        try:
            rest_pose, self.skinning_weights, self.bone_transforms = demBones_simple(
                self.unified_vertices, 
                self.parents, 
                nnz=4,      # 保守的权重数
                n_iters=15  # 适中的迭代数
            )
            
            print(f"✓ DemBones successful!")
            print(f"  Skinning weights: {self.skinning_weights.shape}")
            print(f"  Bone transforms: {self.bone_transforms.shape}")
            print(f"  Weight range: [{self.skinning_weights.min():.4f}, {self.skinning_weights.max():.4f}]")
            
        except Exception as e:
            print(f"❌ DemBones failed: {e}")
            raise RuntimeError(f"DemBones computation failed: {e}")
        
        # 保存结果
        results = {
            'rest_pose': rest_pose,
            'skinning_weights': self.skinning_weights,
            'bone_transforms': self.bone_transforms,
            'parents': self.parents,
            'unified_vertices': self.unified_vertices,
            'rest_pose_idx': self.rest_pose_idx
        }
        
        results_path = os.path.join(self.output_dir, 'demBones_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✓ Results saved to {results_path}")
    
    def step5_interpolate(self, from_frame, to_frame, num_interp):
        """步骤5: 生成插值"""
        print(f"\n=== STEP 5: Interpolation {from_frame} → {to_frame} ({num_interp} frames) ===")
        
        if self.skinning_weights is None or self.bone_transforms is None:
            raise RuntimeError("Must compute skinning first")
        
        # 插值骨骼变换
        start_transforms = self.bone_transforms[from_frame]
        end_transforms = self.bone_transforms[to_frame]
        
        interpolated_frames = []
        
        for i in range(num_interp):
            t = (i + 1) / (num_interp + 1)
            print(f"  Generating frame {i+1}/{num_interp} (t={t:.3f})")
            
            # 线性插值变换矩阵
            interp_transforms = start_transforms * (1 - t) + end_transforms * t
            
            # 应用蒙皮变换
            rest_vertices = self.unified_vertices[self.rest_pose_idx]
            deformed_vertices = self._apply_skinning(rest_vertices, interp_transforms)
            
            interpolated_frames.append(deformed_vertices)
        
        # 保存插值结果
        for i, vertices in enumerate(interpolated_frames):
            output_file = os.path.join(
                self.output_dir, 
                f"fixed_interp_{from_frame:03d}_{to_frame:03d}_{i:03d}.obj"
            )
            self._save_mesh(vertices, output_file)
            print(f"    Saved: {os.path.basename(output_file)}")
        
        print(f"✓ Generated {len(interpolated_frames)} interpolated frames")
        return len(interpolated_frames)
    
    def _apply_skinning(self, rest_vertices, bone_transforms):
        """应用蒙皮变换"""
        N, _ = rest_vertices.shape
        K = len(self.parents)
        
        deformed_vertices = np.zeros_like(rest_vertices)
        
        for v in range(N):
            vertex = rest_vertices[v]
            weights = self.skinning_weights[v]
            
            # 应用加权骨骼变换
            transformed_vertex = np.zeros(3)
            for k in range(K):
                if weights[k] > 1e-6:  # 只处理有意义的权重
                    # 应用骨骼变换
                    homogeneous_vertex = np.append(vertex, 1.0)
                    transformed = bone_transforms[k] @ homogeneous_vertex
                    transformed_vertex += weights[k] * transformed[:3]
            
            deformed_vertices[v] = transformed_vertex
        
        return deformed_vertices
    
    def _save_mesh(self, vertices, output_file):
        """保存网格"""
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        
        # 生成简单的三角面
        if len(vertices) > 3:
            try:
                mesh = mesh.compute_convex_hull()[0]
            except:
                pass
        
        o3d.io.write_triangle_mesh(output_file, mesh)

def main():
    parser = argparse.ArgumentParser(description='Fixed VV Interpolation Pipeline')
    parser.add_argument('folder_path', help='Path to folder containing .obj files')
    parser.add_argument('--from_frame', type=int, required=True, help='Start frame index')
    parser.add_argument('--to_frame', type=int, required=True, help='End frame index')
    parser.add_argument('--num_interp', type=int, default=5, help='Number of interpolation frames')
    parser.add_argument('--max_frames', type=int, default=20, help='Max frames to process')
    parser.add_argument('--max_vertices', type=int, default=None, help='Max vertices per frame')
    parser.add_argument('--output_dir', help='Output directory')
    
    args = parser.parse_args()
    
    print("🔧 Fixed VV Pipeline - DemBones Focus")
    print(f"📁 Input: {args.folder_path}")
    print(f"🎯 Range: {args.from_frame} → {args.to_frame}")
    print(f"📊 Interpolation: {args.num_interp} frames")
    print(f"⚙️  Max frames: {args.max_frames}")
    if args.max_vertices:
        print(f"⚙️  Max vertices: {args.max_vertices}")
    
    start_time = time.time()
    
    try:
        processor = FixedVVProcessor(args.folder_path, args.output_dir)
        
        # 执行pipeline
        processor.step1_process_frames(args.max_frames)
        processor.step2_detect_rest_pose()
        processor.step3_unify_topology(args.max_vertices)
        processor.step4_demBones_skinning()
        processor.step5_interpolate(args.from_frame, args.to_frame, args.num_interp)
        
        elapsed = time.time() - start_time
        print(f"\n🎉 Pipeline completed successfully! Time: {elapsed:.1f}s")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
