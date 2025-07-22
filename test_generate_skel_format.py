#!/usr/bin/env python3
"""
使用GenerateSkel.py的数据格式来测试DemBones
"""

import numpy as np
import py_dem_bones as pdb
import time

def test_with_generate_skel_format():
    """使用GenerateSkel.py完全相同的数据格式"""
    print("🔧 使用GenerateSkel.py数据格式测试")
    print("=" * 50)
    
    try:
        # 导入GenerateSkel的辅助函数
        from GenerateSkel import load_sequence_data, fix_bone_parenting
        
        # 加载实际数据
        data_files = [
            "d:/Code/neural_marionette/data/demo/source/000.npz",
            "d:/Code/neural_marionette/data/demo/source/001.npz"  # 只用2帧
        ]
        
        print("📁 加载数据...")
        frames_vertices, bones_data = load_sequence_data(data_files)
        F, N, _ = frames_vertices.shape
        K = len(bones_data)
        
        print(f"数据规模: {F} frames, {N} vertices, {K} bones")
        
        # 提取少量顶点进行测试
        n_test_vertices = 10  # 只测试10个顶点
        test_vertices = frames_vertices[:, :n_test_vertices, :]
        
        print(f"测试规模: {F} frames, {n_test_vertices} vertices, {K} bones")
        
        # 完全按照GenerateSkel.py的方式处理
        parents = np.array([bone['parent'] for bone in bones_data], dtype=np.int32)
        parents = fix_bone_parenting(parents)  # 修复self-parenting
        
        print(f"父节点: {parents[:10]}...")  # 显示前10个
        
        # 使用和GenerateSkel.py相同的包装器和设置
        dem = pdb.DemBonesExtWrapper()
        
        # 完全相同的参数设置
        n_iters = 3  # 更少的迭代
        nnz = 4
        
        dem.num_iterations = n_iters
        dem.max_nonzeros_per_vertex = nnz
        dem.weights_smoothness = 1e-4
        # dem.weights_sparseness = 1e-6  # 如果有的话
        dem.num_vertices = n_test_vertices
        dem.num_bones = K
        
        print(f"参数设置: iterations={n_iters}, nnz={nnz}, vertices={n_test_vertices}, bones={K}")
        
        # 完全相同的数据准备
        rest_pose = test_vertices[0].T.astype(np.float64)  # (3, N)
        anim_poses = test_vertices.transpose(0,2,1).astype(np.float64)  # (F,3,N)
        anim_poses = anim_poses.reshape(3, -1)  # (3, F·N)
        
        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {anim_poses.shape}")
        
        # 完全相同的数据设置顺序
        dem.set_rest_pose(rest_pose)
        
        # 尝试设置animated_poses属性（如果存在）
        if hasattr(dem, 'animated_poses'):
            dem.animated_poses = anim_poses
        
        dem.set_target_vertices('animated', anim_poses)
        
        # 设置parents - 这可能是关键差异
        if hasattr(dem, 'parents'):
            dem.parents = parents.astype(np.int32)
            print("✅ 设置parents属性")
        else:
            # 或者使用骨骼名称和层次结构
            bone_names = [f"bone_{i}" for i in range(K)]
            dem.set_bone_names(*bone_names)
            
            for i in range(K):
                if parents[i] >= 0 and parents[i] < K:
                    dem.set_parent_bone(i, parents[i])
                else:
                    dem.set_parent_bone(i, None)
            print("✅ 设置骨骼层次结构")
        
        print(f"最终状态: vertices={dem.num_vertices}, bones={dem.num_bones}, targets={dem.num_targets}")
        
        # 尝试计算，但有时间限制
        print("🚀 开始计算（限时测试）...")
        
        start_time = time.time()
        
        # 创建一个简单的计算线程
        import threading
        
        result = [None]
        exception = [None]
        
        def compute_thread():
            try:
                result[0] = dem.compute()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=compute_thread)
        thread.daemon = True
        thread.start()
        
        # 等待最多30秒
        thread.join(timeout=30)
        
        elapsed = time.time() - start_time
        
        if thread.is_alive():
            print(f"⏰ 计算超时 (30秒)，线程仍在运行")
            return False
        elif exception[0]:
            print(f"❌ 计算异常 (耗时 {elapsed:.2f}s): {exception[0]}")
            return False
        elif result[0]:
            print(f"✅ 计算成功 (耗时 {elapsed:.2f}s)")
            
            # 获取结果
            weights = dem.get_weights()
            print(f"权重矩阵: {weights.shape}")
            print(f"权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
            
            return True
        else:
            print(f"❌ 计算失败 (耗时 {elapsed:.2f}s)")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_working_case():
    """尝试找到最小的工作用例"""
    print("\n🧪 寻找最小工作用例")
    print("=" * 30)
    
    # 最简单：1个顶点，1个骨骼，静态
    try:
        dem = pdb.DemBonesExtWrapper()
        
        # 单个顶点，两个"帧"（实际相同）
        vertices = np.array([
            [[0, 0, 0]],  # frame 0
            [[0, 0, 0]]   # frame 1 - 相同位置
        ], dtype=np.float64)
        
        rest_pose = vertices[0].T  # (3, 1)
        
        # 最小设置
        dem.num_bones = 1
        dem.num_iterations = 1
        dem.max_nonzeros_per_vertex = 1
        dem.weight_smoothness = 0.0
        
        dem.set_rest_pose(rest_pose)
        dem.set_target_vertices("static", vertices[1].T)
        
        dem.set_bone_names("root")
        dem.set_parent_bone(0, None)
        
        print(f"最小设置: {dem.num_vertices}v, {dem.num_bones}b, {dem.num_targets}t")
        
        # 快速计算测试
        import threading
        result = [None]
        
        def quick_compute():
            result[0] = dem.compute()
        
        thread = threading.Thread(target=quick_compute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=10)  # 10秒超时
        
        if thread.is_alive():
            print("❌ 最小用例也超时")
            return False
        elif result[0]:
            print("✅ 最小用例成功!")
            return True
        else:
            print("❌ 最小用例失败")
            return False
            
    except Exception as e:
        print(f"❌ 最小用例异常: {e}")
        return False

if __name__ == "__main__":
    print("开始DemBones完整测试...")
    
    # 先尝试最小用例
    minimal_success = test_minimal_working_case()
    
    if minimal_success:
        print("\n✅ 最小用例成功，继续实际数据测试...")
        full_success = test_with_generate_skel_format()
        
        if full_success:
            print("\n🎉 完整测试成功！DemBones问题已解决！")
        else:
            print("\n⚠️ 最小用例成功但完整数据失败，需要调整规模或参数")
    else:
        print("\n❌ 连最小用例都失败，DemBones可能有根本性问题")
        print("建议检查:")
        print("1. DemBones版本兼容性")
        print("2. 依赖库版本")
        print("3. 数据类型和精度")
        print("4. 是否需要额外的初始化步骤")
