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
import random
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

def save_all_frames(vertices, faces, joints, parents, output_dir):
    """保存所有帧的OBJ和骨骼信息"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # get all frame_XXX.obj files under output_dir
    extracted_indices = []
    for obj_file in output_dir.glob("frame_*.obj"):
        extracted_indices.append(int(obj_file.stem.split("_")[-1]))

    for t in range(vertices.shape[0]):
        if t not in extracted_indices:
            frame_mesh = trimesh.Trimesh(vertices=vertices[t], faces=faces)
            frame_mesh.export(output_dir / f"frame_{t:03d}.obj")

    # 保存骨骼信息
    skeleton_info = {
        'joints': joints.tolist(),
        'parents': parents
    }
    with open(output_dir / "skeleton.json", 'w') as f:
        json.dump(skeleton_info, f)

def extract_keyframe_pairs(vertices, faces, joints, parents, k=10, max_pairs=None):
    """每隔k帧抽取首尾帧对"""
    T = vertices.shape[0]
    keyframe_pairs = []
    
    #pick max_pairs frames from start_range
    start_range = random.sample(list(range(0, T - k, 1)), max_pairs)

    for start_idx in start_range:
        end_idx = start_idx + k
        
        # 如果指定了max_pairs且已经生成了足够的pairs，则停止
        if max_pairs is not None and len(keyframe_pairs) >= max_pairs:
            break
        
        # 保存起始帧
        # start_vertices = vertices[start_idx]
        start_joints = joints[start_idx]
        
        # 保存结束帧
        # end_vertices = vertices[end_idx]
        end_joints = joints[end_idx]
        
        # 计算T-pose骨长
        bone_lengths = []
        for i, parent in enumerate(parents):
            if parent >= 0:  # 不是根节点
                bone_vec = start_joints[i] - start_joints[parent]
                bone_length = np.linalg.norm(bone_vec)
                bone_lengths.append(bone_length)
        
        # 提取用于蒙皮优化的帧数据，使用与Interpolate.py中双参考帧优化相同的策略
        gt_frame_data = []
        intermediate_indices = []
        
        # 使用与Interpolate.py相同的max_optimize_frames默认值
        max_optimize_frames = 5
        
        # 为start frame选择优化帧（范围[-5, +5]）
        start_available_frames = []
        for i in range(max(0, start_idx - 5), min(len(vertices), start_idx + 6)):
            start_available_frames.append(i)
        
        # 为end frame选择优化帧（范围[-5, +5]）
        end_available_frames = []
        for i in range(max(0, end_idx - 5), min(len(vertices), end_idx + 6)):
            end_available_frames.append(i)
        
        # 为start frame优化帧选择策略（与Interpolate.py保持一致）
        start_optimize_frames = []
        if len(start_available_frames) > max_optimize_frames:
            center_idx = start_available_frames.index(start_idx)
            half_range = max_optimize_frames // 2
            
            start_idx_range = max(0, center_idx - half_range)
            end_idx_range = min(len(start_available_frames), center_idx + half_range + 1)
            start_optimize_frames = start_available_frames[start_idx_range:end_idx_range]
            
            # 如果还不够max_optimize_frames个，从两边补充
            while len(start_optimize_frames) < max_optimize_frames and (start_idx_range > 0 or end_idx_range < len(start_available_frames)):
                if start_idx_range > 0:
                    start_idx_range -= 1
                    start_optimize_frames.insert(0, start_available_frames[start_idx_range])
                if len(start_optimize_frames) < max_optimize_frames and end_idx_range < len(start_available_frames):
                    start_optimize_frames.append(start_available_frames[end_idx_range])
                    end_idx_range += 1
        else:
            start_optimize_frames = start_available_frames
        
        # 为end frame优化帧选择策略（与Interpolate.py保持一致）
        end_optimize_frames = []
        if len(end_available_frames) > max_optimize_frames:
            center_idx = end_available_frames.index(end_idx)
            half_range = max_optimize_frames // 2
            
            start_idx_range = max(0, center_idx - half_range)
            end_idx_range = min(len(end_available_frames), center_idx + half_range + 1)
            end_optimize_frames = end_available_frames[start_idx_range:end_idx_range]
            
            # 如果还不够max_optimize_frames个，从两边补充
            while len(end_optimize_frames) < max_optimize_frames and (start_idx_range > 0 or end_idx_range < len(end_available_frames)):
                if start_idx_range > 0:
                    start_idx_range -= 1
                    end_optimize_frames.insert(0, end_available_frames[start_idx_range])
                if len(end_optimize_frames) < max_optimize_frames and end_idx_range < len(end_available_frames):
                    end_optimize_frames.append(end_available_frames[end_idx_range])
                    end_idx_range += 1
        else:
            end_optimize_frames = end_available_frames
        
        # 合并两组优化帧并去重
        all_optimize_frames = sorted(list(set(start_optimize_frames + end_optimize_frames)))
        
        # 收集优化帧的顶点数据
        # for idx in all_optimize_frames:
        #     if idx < len(vertices):
        #         gt_frame_data.append(vertices[idx])
        
        pair_info = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            # 'start_vertices': start_vertices,
            # 'end_vertices': end_vertices,
            'start_joints': start_joints,
            'end_joints': end_joints,
            # 'faces': faces,
            'parents': parents,
            'bone_lengths': bone_lengths,
            # 'gt_frames': gt_frame_data,  # 实际的顶点数据
            'gt_frame_indices': all_optimize_frames,  # 原始索引
            'start_optimize_frames': start_optimize_frames,  # start frame的优化帧索引
            'end_optimize_frames': end_optimize_frames,  # end frame的优化帧索引
        }
        
        keyframe_pairs.append(pair_info)
    
    return keyframe_pairs

def save_keyframe_pair(pair_info, output_dir, pair_idx, k_value):
    """保存关键帧对和蒙皮优化所需的中间帧"""
    # pair_dir = Path(output_dir) / f"pair_{pair_idx:03d}"
    # pair_dir.mkdir(parents=True, exist_ok=True)
    
    # # 计算实际的帧间距离
    # k = pair_info['end_idx'] - pair_info['start_idx']
    
    # # 保存起始帧OBJ 
    # start_mesh = trimesh.Trimesh(
    #     vertices=pair_info['start_vertices'],
    #     faces=pair_info['faces']
    # )
    # # start_mesh.export(pair_dir / "start_frame.obj")  # 保持兼容性
    # start_mesh.export(pair_dir / f"frame_{pair_info['start_idx']:03d}.obj")    # 用于volumetric pipeline索引（始终是0）

    # # 保存结束帧OBJ 
    # end_mesh = trimesh.Trimesh(
    #     vertices=pair_info['end_vertices'],
    #     faces=pair_info['faces']
    # )
    # # end_mesh.export(pair_dir / "end_frame.obj")      # 保持兼容性
    # end_mesh.export(pair_dir / f"frame_{pair_info['end_idx']:03d}.obj")  # 用于volumetric pipeline索引（使用实际的k值）

    # # 为蒙皮权重优化生成额外的中间帧
    # # 这些帧将用于 LBS 学习和权重优化，使用与Interpolate.py相同的策略
    # if 'gt_frames' in pair_info and len(pair_info['gt_frames']) > 0:
    #     gt_frames = pair_info['gt_frames']
    #     gt_frame_indices = pair_info['gt_frame_indices']
        
    #     # 输出优化帧选择信息
    #     print(f"  双参考帧优化策略:")
    #     print(f"    - Start frame {pair_info['start_idx']} 优化帧: {pair_info['start_optimize_frames']}")
    #     print(f"    - End frame {pair_info['end_idx']} 优化帧: {pair_info['end_optimize_frames']}")
    #     print(f"    - 合并去重后的所有优化帧: {pair_info['gt_frame_indices']}")
        
    #     # 直接保存所有优化帧，无需再次采样
    #     # 因为已经在extract_keyframe_pairs中按照Interpolate.py的逻辑选择了
    #     saved_count = 0
    #     for i, (frame_idx, vertices) in enumerate(zip(gt_frame_indices, gt_frames)):
    #         try:
    #             # 确保顶点数据是正确的形状
    #             if vertices.shape[0] > 0 and vertices.shape[1] == 3:
    #                 optimization_mesh = trimesh.Trimesh(
    #                     vertices=vertices,
    #                     faces=pair_info['faces'],
    #                     validate=False  # 跳过验证以避免潜在问题
    #                 )
    #                 # 保存所有优化帧，使用实际的帧索引
    #                 # 这样与Interpolate.py的双参考帧优化策略完全一致
    #                 optimization_mesh.export(pair_dir / f"opt_frame_{frame_idx:03d}.obj")
    #                 saved_count += 1
    #             else:
    #                 print(f"    警告: 优化帧 {frame_idx} 的顶点数据形状异常: {vertices.shape}")
    #         except Exception as e:
    #             print(f"    警告: 保存优化帧 {frame_idx} 失败: {e}")
    #             continue
        
    #     print(f"  保存了 {2 + saved_count} 个帧用于蒙皮优化: start, end + {saved_count} 优化帧")
    #     print(f"  优化帧索引范围: {min(gt_frame_indices)} 到 {max(gt_frame_indices)}")
    # else:
    #     print(f"  只保存了起始和结束帧（优化帧数据不足）")
    
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
        'gt_frame_indices': convert_to_serializable(pair_info['gt_frame_indices']),
        'start_optimize_frames': convert_to_serializable(pair_info['start_optimize_frames']),
        'end_optimize_frames': convert_to_serializable(pair_info['end_optimize_frames']),
    }
    
    # 创建k值子目录
    k_dir = Path(output_dir) / f"k{k_value}"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    pair_info_path = k_dir / f"pair_{pair_idx:03d}.json"

    with open(pair_info_path, 'w') as f:
        json.dump(skeleton_info, f, indent=2)
    
    return pair_info_path

def main():
    parser = argparse.ArgumentParser(description="Generate keyframe pairs")
    parser.add_argument("--hdf5_path", type=str, 
                       default="evaluation/data/dfaust/registrations_m.hdf5",
                       help="DFAUST HDF5 file path")
    parser.add_argument("--output_dir", type=str, 
                       default="evaluation/data/dfaust/keyframe_pairs",
                       help="Output base directory")
    parser.add_argument("--k", type=int, default=10,
                       help="Extract one keyframe pair every k frames")
    parser.add_argument("--max_pairs", type=int, default=None,
                       help="Maximum number of keyframe pairs to generate (None means no limit)")
    parser.add_argument("--subject_id", type=str, default="50002",
                       help="DFAUST subject ID")
    parser.add_argument("--sequence_id", type=str, default="jumping_jacks",
                       help="DFAUST sequence ID")
    
    args = parser.parse_args()
    
    # Extract database name from hdf5 path
    database_name = Path(args.hdf5_path).stem
    
    # Create specific output directory, including database_name, subject_sequence, and k value information
    specific_output_dir = Path(args.output_dir) / database_name / f"{args.subject_id}_{args.sequence_id}"
    specific_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading DFAUST data: {args.hdf5_path}")
    print(f"Output directory: {specific_output_dir}")
    vertices, faces, joints, parents = load_dfaust_data(
        args.hdf5_path, args.subject_id, args.sequence_id
    )
    
    print(f"Data shape: vertices={vertices.shape}, joints={joints.shape}")
    
    save_all_frames(vertices, faces, joints, parents, specific_output_dir)
    
    # 生成关键帧对
    keyframe_pairs = extract_keyframe_pairs(vertices, faces, joints, parents, args.k, args.max_pairs)
    max_pairs_info = f" (limited to {args.max_pairs})" if args.max_pairs else ""
    print(f"Generated {len(keyframe_pairs)} keyframe pairs{max_pairs_info}")
    
    # 保存关键帧对
    output_dir = specific_output_dir

    pair_info_paths = []
    for i, pair_info in enumerate(keyframe_pairs):
        pair_info_path = save_keyframe_pair(pair_info, output_dir, i, args.k)
        pair_info_paths.append(str(pair_info_path))
        print(f"Saved keyframe pair {i}: {pair_info_path}")

    # 创建k值子目录并保存索引文件
    k_dir = Path(output_dir) / f"k{args.k}"
    k_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存所有关键帧对的索引到k值子目录
    with open(k_dir / "pairs_index.json", 'w') as f:
        json.dump({
            'pair_info_paths': pair_info_paths,
            'total_pairs': len(keyframe_pairs),
            'k': args.k,
            'subject_id': args.subject_id,
            'sequence_id': args.sequence_id,
            'database_name': database_name,
            'hdf5_path': str(args.hdf5_path)
        }, f, indent=2)
    
    print(f"All keyframe pairs saved to: {output_dir}/k{args.k}")

if __name__ == "__main__":
    main() 