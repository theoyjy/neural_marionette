import numpy as np
import time
try:
    import py_dem_bones as pdb
    print("✅ 成功导入 py_dem_bones")
except ImportError as e:
    print(f"❌ 无法导入 py_dem_bones: {e}")
    exit(1)

def safe_demBones_test():
    """安全的DemBones测试，避免崩溃"""
    print("=== 安全DemBones测试（小数据集）===")
    
    # 使用最小的测试数据
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    # 创建2帧变形
    frame1 = rest_vertices + [0.3, 0, 0]
    frame2 = rest_vertices + [0, 0.3, 0]
    
    animated_poses = np.concatenate([frame1, frame2], axis=0)
    
    print(f"Rest pose: {rest_vertices.shape}")
    print(f"Animated poses: {animated_poses.shape}")
    
    try:
        # 初始化DemBones - 使用最保守的设置
        db = pdb.DemBones()
        
        N = 4  # 顶点数
        K = 2  # 骨骼数
        F = 3  # 总帧数
        
        db.nV = N
        db.nB = K
        db.nF = F - 1
        db.nS = 1
        db.fStart = np.array([0], dtype=np.int32)
        db.subjectID = np.zeros(F - 1, dtype=np.int32)
        db.u = rest_vertices
        db.v = animated_poses
        
        # 最保守的参数，避免崩溃
        db.nIters = 10
        db.nInitIters = 3
        db.nWeightsIters = 2
        db.nTransIters = 2
        
        print(f"配置: nV={N}, nB={K}, nF={F-1}")
        print("🚀 开始计算（保守参数）...")
        
        start_time = time.time()
        db.compute()
        elapsed = time.time() - start_time
        
        print(f"✅ 计算完成 (耗时 {elapsed:.2f}s)")
        
        # 获取结果
        weights = db.get_weights()
        transforms = db.get_transformations()
        rest_skel = db.get_rest_pose()
        
        print(f"\n📊 结果:")
        print(f"Weights: {weights.shape}")
        print(f"Transforms: {transforms.shape}")
        print(f"Rest skeleton: {rest_skel.shape}")
        
        # 显示详细内容
        if weights.size > 0:
            print(f"\nWeights内容:\n{weights}")
        if transforms.size > 0:
            print(f"\nTransforms shape详细: {transforms.shape}")
            if len(transforms.shape) == 3 and transforms.shape[0] > 0 and transforms.shape[1] > 0:
                print(f"第0帧第0骨骼变换:\n{transforms[0, 0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_even_smaller():
    """测试最小数据集"""
    print("\n=== 最小数据集测试 ===")
    
    # 只有3个顶点的三角形
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0]
    ], dtype=np.float32)
    
    # 1帧变形
    frame1 = rest_vertices + [0.2, 0.2, 0]
    
    try:
        db = pdb.DemBones()
        db.nV = 3
        db.nB = 1  # 只有1个骨骼
        db.nF = 1
        db.nS = 1
        db.fStart = np.array([0], dtype=np.int32)
        db.subjectID = np.zeros(1, dtype=np.int32)
        db.u = rest_vertices
        db.v = frame1
        
        # 最小参数
        db.nIters = 5
        db.nInitIters = 1
        db.nWeightsIters = 1
        db.nTransIters = 1
        
        print(f"最小配置: 3顶点, 1骨骼, 1帧")
        
        start_time = time.time()
        db.compute()
        elapsed = time.time() - start_time
        
        print(f"✅ 最小测试完成 (耗时 {elapsed:.2f}s)")
        
        weights = db.get_weights()
        transforms = db.get_transformations()
        
        print(f"最小测试结果: weights={weights.shape}, transforms={transforms.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 最小测试失败: {e}")
        return False

def debug_crash_issue():
    """调试崩溃问题"""
    print("\n=== 调试崩溃问题 ===")
    
    # 逐步增加复杂度
    test_configs = [
        (3, 1, 1),   # 3顶点, 1骨骼, 1帧
        (4, 1, 1),   # 4顶点, 1骨骼, 1帧  
        (4, 2, 1),   # 4顶点, 2骨骼, 1帧
        (4, 2, 2),   # 4顶点, 2骨骼, 2帧
        (10, 2, 2),  # 10顶点, 2骨骼, 2帧
        (20, 3, 3),  # 20顶点, 3骨骼, 3帧
    ]
    
    for nV, nB, nF in test_configs:
        print(f"\n测试配置: {nV}顶点, {nB}骨骼, {nF}帧")
        
        try:
            # 生成测试数据
            rest_vertices = np.random.randn(nV, 3).astype(np.float32) * 0.5
            animated_frames = []
            for f in range(nF):
                frame = rest_vertices + 0.1 * f * np.random.randn(nV, 3).astype(np.float32)
                animated_frames.append(frame)
            
            animated_poses = np.concatenate(animated_frames, axis=0)
            
            # DemBones设置
            db = pdb.DemBones()
            db.nV = nV
            db.nB = nB
            db.nF = nF
            db.nS = 1
            db.fStart = np.array([0], dtype=np.int32)
            db.subjectID = np.zeros(nF, dtype=np.int32)
            db.u = rest_vertices
            db.v = animated_poses
            
            # 保守参数
            db.nIters = 5
            db.nInitIters = 2
            db.nWeightsIters = 1
            db.nTransIters = 1
            
            # 尝试计算
            start_time = time.time()
            db.compute()
            elapsed = time.time() - start_time
            
            weights = db.get_weights()
            transforms = db.get_transformations()
            
            print(f"   ✅ 成功: weights={weights.shape}, transforms={transforms.shape}, 耗时={elapsed:.2f}s")
            
        except Exception as e:
            print(f"   ❌ 失败于配置({nV}, {nB}, {nF}): {e}")
            break  # 停止在第一个失败的配置

if __name__ == "__main__":
    print("开始DemBones稳定性测试...")
    
    # 1. 安全测试
    success1 = safe_demBones_test()
    
    # 2. 最小测试
    success2 = test_even_smaller()
    
    # 3. 调试崩溃问题
    debug_crash_issue()
    
    print(f"\n=== 总结 ===")
    print(f"安全测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"最小测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("🎉 DemBones基本功能正常，崩溃可能是由于数据规模过大")
        print("📝 建议：在实际使用中限制顶点数量或使用采样")
    else:
        print("⚠️ DemBones存在基础问题，需要进一步调试")
