#!/usr/bin/env python3
"""
DemBones修复测试
===============

专门修复DemBones调用问题的最小测试脚本
"""

import numpy as np
import py_dem_bones as pdb
from GenerateSkel import _as_rowmajor, sanitize_parents
import pickle
import os
import time

def test_demBones_minimal():
    """使用最小数据测试DemBones"""
    
    # 创建超简单的测试数据
    print("创建测试数据...")
    
    # 3帧，4个顶点，3个骨骼
    F, N, K = 3, 4, 3
    
    # 顶点数据：简单的四面体变形
    frames_vertices = np.array([
        # Frame 0
        [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        # Frame 1  
        [[-0.8, 0, 0], [0.8, 0, 0], [0, 0.8, 0], [0, 0, 0.8]],
        # Frame 2
        [[-0.6, 0, 0], [0.6, 0, 0], [0, 0.6, 0], [0, 0, 0.6]]
    ], dtype=np.float32)
    
    # 父节点：简单的线性链
    parents = np.array([-1, 0, 1], dtype=np.int32)
    
    print(f"测试数据: {F} frames, {N} vertices, {K} bones")
    print(f"Vertices shape: {frames_vertices.shape}")
    print(f"Parents: {parents}")
    
    try:
        # 创建DemBones实例
        dem = pdb.DemBonesExtWrapper()
        
        # 准备数据 - 完全按照原始代码格式
        rest_pose = frames_vertices[0].T.astype(np.float64)  # (3, N)
        anim_poses = frames_vertices.transpose(0,2,1).astype(np.float64)  # (F,3,N)
        anim_poses = anim_poses.reshape(3, -1)  # (3, F·N)
        
        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {anim_poses.shape}")
        
        # 设置数据 - 完全按照原始代码
        print("设置DemBones数据...")
        dem.set_rest_pose(rest_pose)
        dem.animated_poses = anim_poses
        dem.set_target_vertices('animated', anim_poses)
        print(f"  Rest pose: {rest_pose.shape}")
        print(f"  Animated poses: {anim_poses.shape}")
        
        # 正确设置骨骼层次结构
        print("设置骨骼层次结构...")
        # 首先设置骨骼名称
        bone_names = [f"bone_{i}" for i in range(K)]
        dem.set_bone_names(*bone_names)
        
        # 然后设置父子关系
        for i in range(K):
            parent_idx = parents[i]
            if parent_idx >= 0:
                dem.set_parent_bone(i, parent_idx)
            else:
                dem.set_parent_bone(i, None)  # 根骨骼
        
        # 设置参数 - 极保守
        print("设置DemBones参数...")
        dem.num_iterations = 5
        dem.max_nonzeros_per_vertex = 2
        dem.weights_smoothness = 1e-3
        dem.weights_sparseness = 1e-5
        
        # 额外调试：尝试设置更多参数
        try:
            dem.tolerance = 1e-3
            dem.max_line_search_iterations = 3
            print("  设置了收敛参数")
        except:
            print("  收敛参数不可用")
        
        # 检查设置状态
        print(f"DemBones状态检查:")
        print(f"  num_vertices: {dem.num_vertices}")
        print(f"  num_bones: {dem.num_bones}")
        print(f"  num_frames: {dem.num_frames}")
        print(f"  num_targets: {dem.num_targets}")
        print(f"  num_iterations: {dem.num_iterations}")
        print(f"  max_nonzeros_per_vertex: {dem.max_nonzeros_per_vertex}")
        
        # 深度调试：检查所有属性
        print("\n深度调试 - 检查所有可用属性:")
        for attr in sorted(dir(dem)):
            if not attr.startswith('_') and not callable(getattr(dem, attr)):
                try:
                    value = getattr(dem, attr)
                    print(f"  {attr}: {value}")
                except:
                    print(f"  {attr}: <不可访问>")
        
        # 尝试获取父节点信息
        try:
            parent_bones = dem.parent_bones
            print(f"\n父节点信息: {parent_bones}")
        except:
            print("\n无法获取父节点信息")
        
        # 尝试获取骨骼名称
        try:
            bone_names_check = dem.get_bone_names()
            print(f"骨骼名称: {bone_names_check}")
        except:
            print("无法获取骨骼名称")
        
        # 计算前最后检查
        print(f"\n计算前最终检查:")
        print(f"  顶点数匹配: rest_pose={rest_pose.shape[1]}, dem.num_vertices={dem.num_vertices}")
        print(f"  骨骼数匹配: parents={len(parents)}, dem.num_bones={dem.num_bones}")
        print(f"  目标数: {dem.num_targets}")
        
        # 计算
        print("\n开始DemBones计算...")
        start_time = time.time()
        
        try:
            success = dem.compute()
            elapsed = time.time() - start_time
            print(f"计算完成，耗时 {elapsed:.2f}s，结果: {success}")
        except Exception as compute_error:
            elapsed = time.time() - start_time
            print(f"计算异常，耗时 {elapsed:.2f}s: {compute_error}")
            
            # 尝试获取更详细的错误信息
            try:
                print("尝试获取详细错误信息...")
                import traceback
                traceback.print_exc()
            except:
                pass
            
            return False, None
        
        if success:
            print("✅ DemBones计算成功!")
            
            # 获取结果
            rest_result = dem._dem_bones.get_rest_pose()
            rest_result = _as_rowmajor(rest_result)
            
            weights = dem.get_weights()
            weights = weights.T.copy()
            
            transforms = dem.get_animated_transformation()
            transforms = transforms.reshape(F, K, 4, 4)
            
            print(f"✅ 结果:")
            print(f"  Rest pose: {rest_result.shape}")
            print(f"  Weights: {weights.shape}")
            print(f"  Transforms: {transforms.shape}")
            print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
            
            return True, (rest_result, weights, transforms)
        else:
            print("❌ DemBones计算失败 (返回False)")
            
            # 尝试诊断为什么返回False
            print("\n失败诊断:")
            try:
                # 检查是否有部分结果
                rest_result = dem._dem_bones.get_rest_pose()
                print(f"  Rest pose size: {rest_result.size}")
            except:
                print("  无法获取rest pose")
            
            try:
                weights = dem.get_weights()
                print(f"  Weights size: {weights.size}")
            except:
                print("  无法获取weights")
            
            return False, None
            
    except Exception as e:
        print(f"❌ DemBones异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_with_real_data():
    """使用真实的测试数据"""
    print("\n测试真实数据...")
    
    # 加载一些现有的数据
    data_files = []
    test_dir = "test_vv_data/fixed_vv_processing"
    
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('_data.pkl'):
                data_files.append(os.path.join(test_dir, file))
    
    if len(data_files) < 2:
        print("❌ 需要至少2个数据文件")
        return False
    
    # 加载数据
    mesh_data = []
    for file in data_files[:3]:  # 只用前3个
        with open(file, 'rb') as f:
            data = pickle.load(f)
            mesh_data.append(data)
    
    if len(mesh_data) < 2:
        print("❌ 加载的数据不足")
        return False
    
    print(f"加载了 {len(mesh_data)} 个数据文件")
    
    # 构建frames_vertices
    frames_vertices = []
    for data in mesh_data:
        pts = data['pts_norm']  # 已归一化的点
        # 只取前100个顶点进行测试
        pts = pts[:100]
        frames_vertices.append(pts)
    
    frames_vertices = np.array(frames_vertices, dtype=np.float32)
    parents = sanitize_parents(mesh_data[0]['parents'])
    
    print(f"真实数据: {frames_vertices.shape}, parents: {len(parents)}")
    
    try:
        # 测试DemBones
        dem = pdb.DemBonesExtWrapper()
        
        F, N, _ = frames_vertices.shape
        K = len(parents)
        
        # 准备数据
        rest_pose = frames_vertices[0].T.astype(np.float64)
        anim_poses = frames_vertices.transpose(0,2,1).astype(np.float64)
        anim_poses = anim_poses.reshape(3, -1)
        
        # 设置数据 - 按照原始代码
        dem.set_rest_pose(rest_pose)
        dem.animated_poses = anim_poses  
        dem.set_target_vertices('animated', anim_poses)
        
        # 正确设置骨骼层次结构
        bone_names = [f"bone_{i}" for i in range(K)]
        dem.set_bone_names(*bone_names)
        
        for i in range(K):
            parent_idx = parents[i]
            if parent_idx >= 0:
                dem.set_parent_bone(i, parent_idx)
            else:
                dem.set_parent_bone(i, None)
        
        # 设置参数
        dem.num_iterations = 8
        dem.max_nonzeros_per_vertex = 3
        dem.weights_smoothness = 1e-3
        dem.weights_sparseness = 1e-5
        
        print(f"DemBones状态: {dem.num_vertices}v, {dem.num_bones}b, {dem.num_frames}f")
        
        # 计算
        print("计算中...")
        success = dem.compute()
        
        if success:
            print("✅ 真实数据测试成功!")
            return True
        else:
            print("❌ 真实数据测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 真实数据测试异常: {e}")
        return False

if __name__ == "__main__":
    print("🔧 DemBones修复测试")
    print("=" * 50)
    
    # 测试1：最小数据
    success1, result = test_demBones_minimal()
    
    # 测试2：真实数据
    success2 = test_with_real_data()
    
    print(f"\n📊 测试结果:")
    print(f"  最小数据测试: {'✅ 成功' if success1 else '❌ 失败'}")
    print(f"  真实数据测试: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 DemBones修复成功！可以继续完善pipeline")
    else:
        print("\n🔧 需要进一步调试DemBones问题")
