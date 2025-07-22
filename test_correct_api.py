#!/usr/bin/env python3
"""
基于basic_example.py的正确DemBones使用方法
"""

import numpy as np
import py_dem_bones as pdb
import time

def test_basic_demBones_api():
    """使用basic_example.py中展示的正确API"""
    print("🔧 使用basic_example.py的正确DemBones API")
    print("=" * 50)
    
    try:
        # 创建简单的测试数据 - 基于basic_example.py的模式
        def create_simple_mesh():
            """创建一个简单的4个顶点的网格"""
            vertices = np.array([
                [-1, -1, 0],  # 0
                [1, -1, 0],   # 1
                [1, 1, 0],    # 2
                [-1, 1, 0]    # 3
            ], dtype=np.float64)
            return vertices
        
        def create_deformed_mesh(scale_y):
            """变形网格 - 沿Y轴拉伸"""
            vertices = create_simple_mesh()
            deformed = vertices.copy()
            deformed[:, 1] *= scale_y
            return deformed
        
        # 按照basic_example.py的方式创建数据
        rest_pose = create_simple_mesh()  # (4, 3)
        animated_poses = np.vstack([
            create_deformed_mesh(1.2),  # Frame 1
            create_deformed_mesh(1.5),  # Frame 2
        ])  # (8, 3) - 2 frames * 4 vertices
        
        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {animated_poses.shape}")
        
        # 使用basic_example.py中的DemBones类（不是ExtWrapper）
        dem_bones = pdb.DemBones()
        
        # 完全按照basic_example.py设置参数
        dem_bones.nIters = 5  # 减少迭代次数
        dem_bones.nInitIters = 2
        dem_bones.nTransIters = 2
        dem_bones.nWeightsIters = 2
        dem_bones.nnz = 2  # 每个顶点最多2个骨骼
        dem_bones.weightsSmooth = 1e-4
        
        # 设置数据 - 完全按照basic_example.py的格式
        dem_bones.nV = 4  # 4 vertices
        dem_bones.nB = 2  # 2 bones
        dem_bones.nF = 2  # 2 frames
        dem_bones.nS = 1  # 1 subject
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(2, dtype=np.int32)  # 2 frames
        dem_bones.u = rest_pose  # Rest pose
        dem_bones.v = animated_poses  # Animated poses
        
        print(f"设置完成:")
        print(f"  nV={dem_bones.nV}, nB={dem_bones.nB}, nF={dem_bones.nF}, nS={dem_bones.nS}")
        print(f"  iterations={dem_bones.nIters}, nnz={dem_bones.nnz}")
        print(f"  fStart={dem_bones.fStart}")
        print(f"  subjectID={dem_bones.subjectID}")
        
        # 计算骨骼分解
        print("🚀 开始计算...")
        start_time = time.time()
        
        try:
            success = dem_bones.compute()
            elapsed = time.time() - start_time
            
            print(f"计算完成: {'✅ 成功' if success else '❌ 失败'} (耗时 {elapsed:.2f}s)")
            
            if success:
                # 获取结果
                weights = dem_bones.get_weights()
                transformations = dem_bones.get_transformations()
                
                print(f"权重矩阵 shape: {weights.shape}")
                print(f"变换矩阵 shape: {transformations.shape}")
                
                print("\n权重矩阵:")
                print(weights)
                
                print("\n变换矩阵 (第一帧):")
                print(transformations[0])
                
                return True, weights, transformations
            else:
                print("计算返回失败")
                return False, None, None
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ 计算异常 (耗时 {elapsed:.2f}s): {e}")
            import traceback
            traceback.print_exc()
            return False, None, None
            
    except Exception as e:
        print(f"❌ 设置异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_with_real_data():
    """使用真实数据测试正确的API"""
    print("\n" + "=" * 50)
    print("🧪 真实数据测试（正确API）")
    
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
        
        # 使用少量顶点测试
        n_test_vertices = 50
        test_frames = frames_vertices[:, :n_test_vertices, :]
        
        print(f"测试规模: {F} frames, {n_test_vertices} vertices, {K} bones")
        
        # 按照basic_example.py的格式准备数据
        rest_pose = test_frames[0]  # (N, 3)
        animated_poses = test_frames[1:].reshape(-1, 3)  # ((F-1)*N, 3)
        
        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {animated_poses.shape}")
        
        # 使用正确的DemBones API
        dem_bones = pdb.DemBones()
        
        # 保守的参数设置
        dem_bones.nIters = 3
        dem_bones.nInitIters = 1
        dem_bones.nTransIters = 1
        dem_bones.nWeightsIters = 1
        dem_bones.nnz = 4
        dem_bones.weightsSmooth = 1e-3
        
        # 设置数据
        dem_bones.nV = n_test_vertices
        dem_bones.nB = min(K, 10)  # 限制骨骼数量
        dem_bones.nF = F - 1  # animated frames (不包括rest pose)
        dem_bones.nS = 1
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(F - 1, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        print(f"真实数据设置:")
        print(f"  nV={dem_bones.nV}, nB={dem_bones.nB}, nF={dem_bones.nF}")
        
        # 计算
        print("🚀 真实数据计算...")
        start_time = time.time()
        
        # 使用线程超时控制
        import threading
        result = [None]
        exception = [None]
        
        def compute_thread():
            try:
                result[0] = dem_bones.compute()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=compute_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout=60)  # 60秒超时
        
        elapsed = time.time() - start_time
        
        if thread.is_alive():
            print(f"⏰ 真实数据计算超时 (60s)")
            return False
        elif exception[0]:
            print(f"❌ 真实数据计算异常 (耗时 {elapsed:.2f}s): {exception[0]}")
            return False
        elif result[0]:
            print(f"✅ 真实数据计算成功 (耗时 {elapsed:.2f}s)")
            
            weights = dem_bones.get_weights()
            print(f"真实数据权重矩阵: {weights.shape}")
            
            return True
        else:
            print(f"❌ 真实数据计算失败 (耗时 {elapsed:.2f}s)")
            return False
            
    except Exception as e:
        print(f"❌ 真实数据测试异常: {e}")
        return False

if __name__ == "__main__":
    print("基于basic_example.py的DemBones测试...")
    
    # 测试基本API
    success, weights, transformations = test_basic_demBones_api()
    
    if success:
        print("\n✅ 基本API测试成功！")
        print("DemBones问题已解决，现在可以用于真实数据")
        
        # 测试真实数据
        real_success = test_with_real_data()
        
        if real_success:
            print("\n🎉 完整测试成功！DemBones现在可以正常工作了！")
        else:
            print("\n⚠️ 基本测试成功，但真实数据需要进一步优化参数")
    else:
        print("\n❌ 基本API测试失败")
        print("需要检查DemBones安装或版本问题")
