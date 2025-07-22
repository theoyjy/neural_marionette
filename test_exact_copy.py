#!/usr/bin/env python3
"""
完全复制basic_example.py的工作方式
"""

import numpy as np
import py_dem_bones as pdb

def test_exact_copy_of_basic_example():
    """完全复制basic_example.py"""
    print("🔧 完全复制basic_example.py")
    print("=" * 50)
    
    # 完全复制basic_example.py的函数
    def create_cube():
        """Create a simple cube mesh."""
        vertices = np.array([
            [-1, -1, -1],  # 0
            [1, -1, -1],   # 1
            [1, 1, -1],    # 2
            [-1, 1, -1],   # 3
            [-1, -1, 1],   # 4
            [1, -1, 1],    # 5
            [1, 1, 1],     # 6
            [-1, 1, 1]     # 7
        ], dtype=np.float64)
        return vertices

    def create_deformed_cube(scale_y):
        """Create a deformed cube by stretching it along the y-axis."""
        vertices = create_cube()
        deformed = vertices.copy()
        deformed[:, 1] *= scale_y
        return deformed
    
    try:
        # 完全按照basic_example.py创建数据
        rest_pose = create_cube()
        animated_poses = np.vstack([
            create_deformed_cube(1.2),  # Frame 1
            create_deformed_cube(1.5),  # Frame 2
            create_deformed_cube(1.8)   # Frame 3
        ])

        print(f"Rest pose shape: {rest_pose.shape}")
        print(f"Animated poses shape: {animated_poses.shape}")

        # Create DemBones instance
        dem_bones = pdb.DemBones()

        # 完全相同的参数设置
        dem_bones.nIters = 20
        dem_bones.nInitIters = 10
        dem_bones.nTransIters = 5
        dem_bones.nWeightsIters = 3
        dem_bones.nnz = 4
        dem_bones.weightsSmooth = 1e-4

        # 完全相同的数据设置
        dem_bones.nV = 8  # 8 vertices in a cube
        dem_bones.nB = 2  # 2 bones
        dem_bones.nF = 3  # 3 frames
        dem_bones.nS = 1  # 1 subject
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(3, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses

        print("数据设置完成，开始计算...")

        # Compute skinning decomposition
        success = dem_bones.compute()
        
        print(f"计算结果: {'✅ 成功' if success else '❌ 失败'}")

        if success:
            # Get results
            weights = dem_bones.get_weights()
            transformations = dem_bones.get_transformations()

            print("✅ 权重矩阵:")
            print(weights)
            print("\n✅ 骨骼变换:")
            print(transformations)
            
            return True, weights, transformations
        else:
            print("❌ 计算失败")
            return False, None, None
            
    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def adapt_for_real_data():
    """将工作的API适配到真实数据"""
    print("\n" + "=" * 50)
    print("🔧 适配真实数据")
    
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
        
        # 使用少量数据测试
        n_test = 20  # 20个顶点
        n_frames = 3  # 3帧
        n_bones = 8   # 8个骨骼
        
        test_data = frames_vertices[:n_frames, :n_test, :]
        
        print(f"测试数据: {n_frames} frames, {n_test} vertices, {n_bones} bones")
        
        # 按照工作的格式准备数据
        rest_pose = test_data[0]  # (n_test, 3)
        animated_poses = test_data[1:].reshape(-1, 3)  # ((n_frames-1)*n_test, 3)
        
        print(f"Rest pose: {rest_pose.shape}")
        print(f"Animated poses: {animated_poses.shape}")
        
        # 使用工作的API
        dem_bones = pdb.DemBones()
        
        # 适中的参数
        dem_bones.nIters = 10
        dem_bones.nInitIters = 5
        dem_bones.nTransIters = 3
        dem_bones.nWeightsIters = 2
        dem_bones.nnz = 4
        dem_bones.weightsSmooth = 1e-4
        
        # 设置数据
        dem_bones.nV = n_test
        dem_bones.nB = n_bones
        dem_bones.nF = n_frames - 1  # animated frames
        dem_bones.nS = 1
        dem_bones.fStart = np.array([0], dtype=np.int32)
        dem_bones.subjectID = np.zeros(n_frames - 1, dtype=np.int32)
        dem_bones.u = rest_pose
        dem_bones.v = animated_poses
        
        print("真实数据设置完成，开始计算...")
        
        # 计算（带超时）
        import threading
        import time
        
        result = [None]
        exception = [None]
        
        def compute_thread():
            try:
                result[0] = dem_bones.compute()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=compute_thread)
        thread.daemon = True
        
        start_time = time.time()
        thread.start()
        thread.join(timeout=120)  # 2分钟超时
        elapsed = time.time() - start_time
        
        if thread.is_alive():
            print(f"⏰ 真实数据计算超时 (120s)")
            return False
        elif exception[0]:
            print(f"❌ 真实数据计算异常 (耗时 {elapsed:.2f}s): {exception[0]}")
            return False
        elif result[0]:
            print(f"✅ 真实数据计算成功 (耗时 {elapsed:.2f}s)")
            
            weights = dem_bones.get_weights()
            transformations = dem_bones.get_transformations()
            
            print(f"权重矩阵: {weights.shape}")
            print(f"变换矩阵: {transformations.shape}")
            
            return True
        else:
            print(f"❌ 真实数据计算失败 (耗时 {elapsed:.2f}s)")
            return False
            
    except Exception as e:
        print(f"❌ 真实数据适配异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("测试完全复制basic_example.py...")
    
    # 完全复制basic_example.py
    success, weights, transformations = test_exact_copy_of_basic_example()
    
    if success:
        print("\n✅ 完全复制成功！DemBones API正常工作")
        print("现在尝试适配真实数据...")
        
        real_success = adapt_for_real_data()
        
        if real_success:
            print("\n🎉 真实数据适配成功！DemBones问题完全解决！")
        else:
            print("\n⚠️ 基本API成功，真实数据需要调整参数或规模")
    else:
        print("\n❌ 连基本复制都失败，环境可能有问题")
