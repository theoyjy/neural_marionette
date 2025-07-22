#!/usr/bin/env python3
"""
使用正确的DemBonesExtWrapper API进行测试
"""

import numpy as np
import py_dem_bones as pdb
import time

def test_correct_demBones_api():
    """使用DemBonesExtWrapper进行正确的测试"""
    print("🔧 DemBonesExtWrapper 正确API测试")
    print("=" * 50)
    
    # 创建测试数据
    n_frames = 3
    n_vertices = 4  
    n_bones = 3
    
    vertices = np.array([
        [[-1, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1]],    # frame 0
        [[-0.8, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0, 0.8]], # frame 1  
        [[-0.6, 0, 0], [0.6, 0, 0], [0, 0, 0], [0, 0, 0.6]]  # frame 2
    ], dtype=np.float64)
    
    parents = np.array([-1, 0, 1], dtype=np.int32)
    
    print(f"测试数据: {n_frames} frames, {n_vertices} vertices, {n_bones} bones")
    
    try:
        # 使用正确的包装器
        dem = pdb.DemBonesExtWrapper()
        print("✅ 创建DemBonesExtWrapper成功")
        
        # 按照GenerateSkel.py的方式设置参数
        dem.num_iterations = 5
        dem.max_nonzeros_per_vertex = 2  # 注意这里是max_nonzeros_per_vertex而不是nnz
        dem.weights_smoothness = 1e-4
        # dem.weights_sparseness = 1e-6  # 检查是否有这个属性
        dem.num_vertices = n_vertices
        dem.num_bones = n_bones
        
        print(f"✅ 参数设置: vertices={dem.num_vertices}, bones={dem.num_bones}, iterations={dem.num_iterations}")
        
        # 准备数据 - 完全按照GenerateSkel.py的格式
        rest_pose = vertices[0].T.astype(np.float64)  # (3, N)
        anim_poses = vertices.transpose(0,2,1).astype(np.float64)  # (F,3,N)
        anim_poses = anim_poses.reshape(3, -1)  # (3, F·N)
        
        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {anim_poses.shape}")
        
        # 设置数据 - 按照原始代码的方式
        dem.set_rest_pose(rest_pose)
        print(f"✅ Rest pose设置成功")
        
        # 设置animated poses - 尝试原始代码的两种方式
        if hasattr(dem, 'animated_poses'):
            dem.animated_poses = anim_poses
            print("✅ animated_poses属性设置成功")
        
        dem.set_target_vertices('animated', anim_poses)
        print("✅ target_vertices设置成功")
        
        # 设置骨骼层次结构
        bone_names = [f"bone_{i}" for i in range(n_bones)]
        dem.set_bone_names(*bone_names)
        print(f"✅ 骨骼名称设置: {bone_names}")
        
        for i in range(n_bones):
            parent_idx = parents[i]
            if parent_idx >= 0:
                dem.set_parent_bone(i, parent_idx)
            else:
                dem.set_parent_bone(i, None)  # 根骨骼
        
        print(f"✅ 骨骼层次设置完成")
        
        # 检查最终状态
        print(f"📊 最终状态:")
        print(f"  num_vertices: {dem.num_vertices}")
        print(f"  num_bones: {dem.num_bones}")
        print(f"  num_targets: {dem.num_targets}")
        print(f"  bone_names: {dem.bone_names}")
        print(f"  parent_bones: {dem.parent_bones}")
        
        # 计算
        print("🚀 开始计算...")
        start_time = time.time()
        try:
            success = dem.compute()
            elapsed = time.time() - start_time
            
            print(f"计算结果: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
            
            if success:
                print("🎉 DemBones计算成功！")
                
                # 获取结果
                try:
                    weights = dem.get_weights()
                    print(f"  权重矩阵shape: {weights.shape}")
                    print(f"  权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
                    
                    # 检查权重是否合理（每行和为1）
                    row_sums = weights.sum(axis=1)
                    print(f"  权重行和范围: [{row_sums.min():.3f}, {row_sums.max():.3f}]")
                    
                    return True, weights
                    
                except Exception as e:
                    print(f"❌ 获取权重失败: {e}")
                    return True, None
            else:
                print("❌ DemBones计算失败")
                return False, None
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 计算异常 (耗时 {elapsed:.2f}s): {e}")
            import traceback
            traceback.print_exc()
            return False, None
            
    except Exception as e:
        print(f"❌ 设置阶段异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 50)
    print("🧪 真实数据测试")
    
    try:
        # 加载真实数据
        from GenerateSkel import load_sequence_data
        
        data_files = [
            "d:/Code/neural_marionette/data/demo/source/000.npz",
            "d:/Code/neural_marionette/data/demo/source/001.npz", 
            "d:/Code/neural_marionette/data/demo/source/002.npz"
        ]
        
        print("加载真实数据...")
        frames_vertices, bones_data = load_sequence_data(data_files)
        F, N, _ = frames_vertices.shape
        K = len(bones_data)
        
        print(f"真实数据: {F} frames, {N} vertices, {K} bones")
        
        # 使用DemBonesExtWrapper
        dem = pdb.DemBonesExtWrapper()
        
        # 保守参数
        dem.num_iterations = 3
        dem.max_nonzeros_per_vertex = 4
        dem.weights_smoothness = 1e-3
        dem.num_vertices = N
        dem.num_bones = K
        
        # 准备数据
        rest_pose = frames_vertices[0].T.astype(np.float64)
        anim_poses = frames_vertices.transpose(0,2,1).astype(np.float64)
        anim_poses = anim_poses.reshape(3, -1)
        
        # 设置数据
        dem.set_rest_pose(rest_pose)
        dem.set_target_vertices('animated', anim_poses)
        
        # 骨骼设置
        bone_names = [f"bone_{i}" for i in range(K)]
        dem.set_bone_names(*bone_names)
        
        parents = np.array([bone['parent'] for bone in bones_data], dtype=np.int32)
        for i in range(K):
            if parents[i] >= 0:
                dem.set_parent_bone(i, parents[i])
            else:
                dem.set_parent_bone(i, None)
        
        print("🚀 真实数据计算中...")
        start_time = time.time()
        success = dem.compute()
        elapsed = time.time() - start_time
        
        print(f"真实数据结果: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
        
        return success
        
    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")
        return False

if __name__ == "__main__":
    # 测试最小数据
    success, weights = test_correct_demBones_api()
    
    if success:
        print("\n✅ 最小数据测试成功，继续测试真实数据...")
        real_success = test_with_real_data()
        
        if real_success:
            print("\n🎉 所有测试成功！DemBones问题已解决！")
        else:
            print("\n⚠️ 最小数据成功，但真实数据失败，需要调整参数")
    else:
        print("\n❌ 最小数据测试失败，需要进一步调试")
