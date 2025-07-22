#!/usr/bin/env python3

import pickle
import numpy as np
import os

# 检查原始数据质量
data_folder = r'D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons'

# 加载一个原始mesh数据
with open(f'{data_folder}/Frame_00001_textured_hd_t_s_c_data.pkl', 'rb') as f:
    original_data = pickle.load(f)

print('=== 原始数据质量 ===')
print(f'pts_norm shape: {original_data["pts_norm"].shape}')
print(f'pts_norm range: [{original_data["pts_norm"].min():.6f}, {original_data["pts_norm"].max():.6f}]')
print(f'pts_norm mean: {original_data["pts_norm"].mean(axis=0)}')
print(f'vertices precision (std): {original_data["pts_norm"].std(axis=0)}')

# 加载skeleton-driven处理后的数据
if os.path.exists(f'{data_folder}/skeleton_driven_results.pkl'):
    with open(f'{data_folder}/skeleton_driven_results.pkl', 'rb') as f:
        skeleton_results = pickle.load(f)
    
    skeleton_data = skeleton_results['unified_vertices']
    print('\n=== Skeleton-driven处理后数据 ===')
    print(f'unified_vertices shape: {skeleton_data.shape}')
    print(f'unified_vertices range: [{skeleton_data.min():.6f}, {skeleton_data.max():.6f}]')
    print(f'unified_vertices mean: {skeleton_data.mean(axis=(0,1))}')
    print(f'vertices precision (std): {skeleton_data.std(axis=(0,1))}')
    
    # 比较第一帧
    frame_0 = skeleton_data[0]
    print(f'\nFrame 0 比较:')
    print(f'Original vertices count: {original_data["pts_norm"].shape[0]}')
    print(f'Skeleton-driven count: {frame_0.shape[0]}')
    print(f'Difference in detail: {abs(original_data["pts_norm"].shape[0] - frame_0.shape[0])}')
    
    # 检查最近邻数据
    if os.path.exists(f'{data_folder}/nearest_neighbor_results.pkl'):
        with open(f'{data_folder}/nearest_neighbor_results.pkl', 'rb') as f:
            nn_results = pickle.load(f)
        nn_data = nn_results['unified_vertices']
        print(f'\n=== Nearest Neighbor处理后数据 ===')
        print(f'nearest_neighbor shape: {nn_data.shape}')
        print(f'nearest_neighbor range: [{nn_data.min():.6f}, {nn_data.max():.6f}]')
        print(f'nearest_neighbor mean: {nn_data.mean(axis=(0,1))}')
        print(f'vertices precision (std): {nn_data.std(axis=(0,1))}')
        
        # 检查直接对应关系
        # 找到Frame_00001对应的索引
        frame_names = skeleton_results['frame_names']
        if 'Frame_00001_textured_hd_t_s_c' in frame_names:
            frame_idx = frame_names.index('Frame_00001_textured_hd_t_s_c')
            skeleton_frame = skeleton_data[frame_idx]
            nn_frame = nn_data[frame_idx]
            
            print(f'\n=== 同一帧对比分析 (Frame_00001) ===')
            print(f'Original (31419 vertices) vs Skeleton-driven (32140 vertices)')
            print(f'Skeleton-driven vs Nearest Neighbor差异:')
            print(f'  Mean difference: {np.mean(np.abs(skeleton_frame - nn_frame)):.6f}')
            print(f'  Max difference: {np.max(np.abs(skeleton_frame - nn_frame)):.6f}')
            print(f'  STD difference: {np.std(skeleton_frame - nn_frame):.6f}')
else:
    print('\nSkeleton-driven结果文件未找到')
