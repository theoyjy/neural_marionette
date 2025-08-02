#!/usr/bin/env python3
"""
生成关键帧对脚本
从DFAUST数据中每隔k帧抽取首尾帧，保存OBJ和skeleton信息
"""

import os
import h5py
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import argparse
import json
from pathlib import Path

def load_dfaust_data(hdf5_path, subject_id="50002", sequence_id="jumping_jacks"):
    """Load DFAUST data, reference official script"""
    with h5py.File(hdf5_path, 'r') as f:
        # Construct sequence key name
        sidseq = f"{subject_id}_{sequence_id}"
        
        if sidseq not in f:
            raise ValueError(f"Sequence {sidseq} not found in {hdf5_path}")
        
        print(f"Loading DFAUST data for {sidseq}")

        # Read data in the same way as the official script
        # verts = f[sidseq].value.transpose([2, 0, 1])  # Old version h5py
        verts = f[sidseq][:].transpose([2, 0, 1])  # 新版本h5py
        faces = f['faces'][:]  # All sequences share the same faces
        
        # Create simple joints (use part of the vertices as joints)
        T, V, _ = verts.shape
        num_joints = min(20, V)  # 最多20个关节
        joints = verts[:, :num_joints, :]  # (T, J, 3)
        
        # 创建简单的父子关系
        parents = [-1] + list(range(num_joints-1))  # 第一个是根节点
        
        return verts, faces, joints, parents

def extract_keyframe_pairs(vertices, faces, joints, parents, k=10, max_pairs=None):
    """每隔k帧抽取首尾帧对"""
    T = vertices.shape[0]
    keyframe_pairs = []
    
    for start_idx in range(0, T - k, k):
        end_idx = start_idx + k
        
        # 如果指定了max_pairs且已经生成了足够的pairs，则停止
        if max_pairs is not None and len(keyframe_pairs) >= max_pairs:
            break
        
        # 保存起始帧
        start_vertices = vertices[start_idx]
        start_joints = joints[start_idx]
        
        # 保存结束帧
        end_vertices = vertices[end_idx]
        end_joints = joints[end_idx]
        
        # 计算T-pose骨长
        bone_lengths = []
        for i, parent in enumerate(parents):
            if parent >= 0:  # 不是根节点
                bone_vec = start_joints[i] - start_joints[parent]
                bone_length = np.linalg.norm(bone_vec)
                bone_lengths.append(bone_length)
        
        # 提取中间帧的顶点数据用于蒙皮优化
        gt_frame_data = []
        if end_idx > start_idx + 1:
            # 选择start和end之间的一些关键帧
            intermediate_indices = list(range(start_idx + 1, end_idx))
            for idx in intermediate_indices:
                if idx < len(vertices):
                    gt_frame_data.append(vertices[idx])
        
        pair_info = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'start_vertices': start_vertices,
            'end_vertices': end_vertices,
            'start_joints': start_joints,
            'end_joints': end_joints,
            'faces': faces,
            'parents': parents,
            'bone_lengths': bone_lengths,
            'gt_frames': gt_frame_data,  # 实际的顶点数据
            'gt_frame_indices': list(range(start_idx + 1, end_idx))  # 原始索引
        }
        
        keyframe_pairs.append(pair_info)
    
    return keyframe_pairs

def save_keyframe_pair(pair_info, output_dir, pair_idx):
    """保存关键帧对和蒙皮优化所需的中间帧"""
    pair_dir = Path(output_dir) / f"pair_{pair_idx:03d}"
    pair_dir.mkdir(exist_ok=True)
    
    # 计算实际的帧间距离
    k = pair_info['end_idx'] - pair_info['start_idx']
    
    # 保存起始帧OBJ 
    start_mesh = trimesh.Trimesh(
        vertices=pair_info['start_vertices'],
        faces=pair_info['faces']
    )
    # start_mesh.export(pair_dir / "start_frame.obj")  # 保持兼容性
    start_mesh.export(pair_dir / "frame_000.obj")    # 用于volumetric pipeline索引（始终是0）
    
    # 保存结束帧OBJ 
    end_mesh = trimesh.Trimesh(
        vertices=pair_info['end_vertices'],
        faces=pair_info['faces']
    )
    # end_mesh.export(pair_dir / "end_frame.obj")      # 保持兼容性
    end_mesh.export(pair_dir / f"frame_{k:03d}.obj")  # 用于volumetric pipeline索引（使用实际的k值）
    
    # 为蒙皮权重优化生成额外的中间帧
    # 这些帧将用于 LBS 学习和权重优化
    if 'gt_frames' in pair_info and len(pair_info['gt_frames']) > 0:
        gt_frames = pair_info['gt_frames']
        
        # 选择最多5个中间帧用于优化
        max_optimization_frames = min(5, len(gt_frames))
        
        # 如果有很多中间帧，均匀采样
        if len(gt_frames) > max_optimization_frames:
            step = len(gt_frames) // max_optimization_frames
            selected_frames = gt_frames[::step][:max_optimization_frames]
        else:
            selected_frames = gt_frames
        
        
        # 保存中间优化帧
        saved_count = 0
        for frame_num, vertices in enumerate(selected_frames, start=1):
            try:
                # 确保顶点数据是正确的形状
                if vertices.shape[0] > 0 and vertices.shape[1] == 3:
                    optimization_mesh = trimesh.Trimesh(
                        vertices=vertices,
                        faces=pair_info['faces'],
                        validate=False  # 跳过验证以避免潜在问题
                    )
                    optimization_mesh.export(pair_dir / f"frame_{frame_num:03d}.obj")
                    saved_count += 1
                else:
                    print(f"    警告: 中间帧 {frame_num} 的顶点数据形状异常: {vertices.shape}")
            except Exception as e:
                print(f"    警告: 保存优化帧 {frame_num} 失败: {e}")
                continue
        
        print(f"  保存了 {2 + saved_count} 个帧用于蒙皮优化: start, end + {saved_count} 中间帧")
    else:
        print(f"  只保存了起始和结束帧（GT帧数据不足）")
    
    # 保存skeleton信息，确保所有numpy类型都转换为Python原生类型
    def convert_to_serializable(obj):
        """将numpy类型转换为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    skeleton_info = {
        'start_joints': convert_to_serializable(pair_info['start_joints']),
        'end_joints': convert_to_serializable(pair_info['end_joints']),
        'parents': convert_to_serializable(pair_info['parents']),
        'bone_lengths': convert_to_serializable(pair_info['bone_lengths']),
        'start_idx': convert_to_serializable(pair_info['start_idx']),
        'end_idx': convert_to_serializable(pair_info['end_idx']),
        'gt_frame_indices': convert_to_serializable(pair_info['gt_frame_indices'])
    }
    
    with open(pair_dir / "skeleton.json", 'w') as f:
        json.dump(skeleton_info, f, indent=2)
    
    return pair_dir

def main():
    parser = argparse.ArgumentParser(description="生成关键帧对")
    parser.add_argument("--hdf5_path", type=str, 
                       default="evaluation/data/dfaust/registrations_m.hdf5",
                       help="DFAUST HDF5文件路径")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/data/dfaust/keyframe_pairs",
                       help="输出基础目录")
    parser.add_argument("--k", type=int, default=10,
                       help="每隔k帧抽取一对")
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="最大生成的关键帧对数量（None表示不限制）")
    parser.add_argument("--subject_id", type=str, default="50002",
                       help="DFAUST subject ID")
    parser.add_argument("--sequence_id", type=str, default="jumping_jacks",
                       help="DFAUST sequence ID")
    
    args = parser.parse_args()
    
    # 从hdf5路径提取数据库名
    database_name = Path(args.hdf5_path).stem
    
    # 创建具体的输出目录，包含database_name、subject_sequence和k值信息
    specific_output_dir = Path(args.output_dir) / database_name / f"{args.subject_id}_{args.sequence_id}_k{args.k}"
    specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Load DFAUST data: {args.hdf5_path}")
    print(f"Output directory: {specific_output_dir}")
    vertices, faces, joints, parents = load_dfaust_data(
        args.hdf5_path, args.subject_id, args.sequence_id
    )
    
    print(f"Data shape: vertices={vertices.shape}, joints={joints.shape}")
    
    # 生成关键帧对
    keyframe_pairs = extract_keyframe_pairs(vertices, faces, joints, parents, args.k, args.max_pairs)
    max_pairs_info = f" (限制为{args.max_pairs})" if args.max_pairs else ""
    print(f"Generated {len(keyframe_pairs)} keyframe pairs{max_pairs_info}")
    
    # 保存关键帧对
    output_dir = specific_output_dir
    
    pair_dirs = []
    for i, pair_info in enumerate(keyframe_pairs):
        pair_dir = save_keyframe_pair(pair_info, output_dir, i)
        pair_dirs.append(str(pair_dir))
        print(f"Save keyframe pair {i}: {pair_dir}")
    
    # 保存所有关键帧对的索引
    with open(output_dir / "pairs_index.json", 'w') as f:
        json.dump({
            'pair_dirs': pair_dirs,
            'total_pairs': len(keyframe_pairs),
            'k': args.k,
            'subject_id': args.subject_id,
            'sequence_id': args.sequence_id,
            'database_name': database_name,
            'hdf5_path': str(args.hdf5_path)
        }, f, indent=2)
    
    print(f"All keyframe pairs saved to: {output_dir}")

if __name__ == "__main__":
    main() 