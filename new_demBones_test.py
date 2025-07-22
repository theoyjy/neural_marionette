import numpy as np
import time
try:
    import py_dem_bones as pdb
    print("✅ 成功导入 py_dem_bones")
except ImportError as e:
    print(f"❌ 无法导入 py_dem_bones: {e}")
    exit(1)

def test_demBones_correct_api():
    """使用正确的API测试DemBones"""
    print("=== 使用正确API测试DemBones ===")
    
    # 创建简单测试数据
    # 4个顶点的正方形
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    # 创建2帧变形
    frame1 = rest_vertices + [0.5, 0, 0]  # 整体右移
    frame2 = rest_vertices + [0, 0.5, 0]  # 整体上移
    
    # 按照成功脚本的格式准备数据
    rest_pose = rest_vertices  # (N, 3)
    animated_poses = np.concatenate([frame1, frame2], axis=0)  # (2*N, 3)
    
    print(f"Rest pose shape: {rest_pose.shape}")
    print(f"Animated poses shape: {animated_poses.shape}")
    
    # 初始化DemBones - 使用成功脚本的方式
    db = pdb.DemBones()
    
    # 使用成功脚本的参数设置方式
    N = rest_pose.shape[0]  # 顶点数
    K = 2  # 骨骼数
    F = 3  # 总帧数（包括rest pose）
    
    # 关键：使用属性而不是方法设置
    db.nV = N
    db.nB = K
    db.nF = F - 1  # 动画帧数（不包括rest pose）
    db.nS = 1      # 主题数
    db.fStart = np.array([0], dtype=np.int32)
    db.subjectID = np.zeros(F - 1, dtype=np.int32)
    db.u = rest_pose
    db.v = animated_poses
    
    # 迭代参数
    db.nIters = 30
    db.nInitIters = 10
    db.nWeightsIters = 5
    db.nTransIters = 5
    
    print(f"DemBones配置:")
    print(f"  nV={db.nV}, nB={db.nB}, nF={db.nF}, nS={db.nS}")
    print(f"  rest pose shape: {db.u.shape}")
    print(f"  animated poses shape: {db.v.shape}")
    print(f"  fStart: {db.fStart}")
    print(f"  subjectID: {db.subjectID}")
    
    # 计算
    print("🚀 开始计算...")
    try:
        start_time = time.time()
        db.compute()
        elapsed = time.time() - start_time
        print(f"✅ 计算完成 (耗时 {elapsed:.2f}s)")
        
        # 获取结果
        weights = db.get_weights()
        transforms = db.get_transformations()
        
        print(f"\n📊 结果分析:")
        print(f"权重矩阵shape: {weights.shape}")
        print(f"变换矩阵shape: {transforms.shape}")
        
        if weights.size > 0:
            print(f"权重矩阵内容:\n{weights}")
            print(f"权重值范围: [{weights.min():.4f}, {weights.max():.4f}]")
            
            # 根据成功脚本，权重应该是(K, N)格式，需要转置为(N, K)
            if weights.shape[0] == K and weights.shape[1] == N:
                print("✅ 权重矩阵格式正确：(K, N)")
                weights_T = weights.T  # 转置为(N, K)
                print(f"转置后权重矩阵shape: {weights_T.shape}")
                print(f"转置后权重矩阵:\n{weights_T}")
                
                # 检查权重归一化
                row_sums = weights_T.sum(axis=1)
                print(f"每个顶点权重和: {row_sums}")
                print(f"权重和是否接近1: {np.allclose(row_sums, 1.0, atol=0.1)}")
                
            return weights, transforms
        else:
            print("❌ 权重矩阵为空")
            return None, None
            
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_single_frame():
    """测试单帧情况"""
    print("\n=== 测试单帧情况 ===")
    
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    frame1 = rest_vertices + [0.3, 0.3, 0]
    
    db = pdb.DemBones()
    db.nV = 4
    db.nB = 1
    db.nF = 1
    db.nS = 1
    db.fStart = np.array([0], dtype=np.int32)
    db.subjectID = np.zeros(1, dtype=np.int32)
    db.u = rest_vertices
    db.v = frame1
    
    print(f"单帧测试: nV={db.nV}, nB={db.nB}, nF={db.nF}")
    
    try:
        db.compute()
        weights = db.get_weights()
        print(f"单帧权重shape: {weights.shape}")
        print(f"单帧权重内容: {weights}")
        return weights
    except Exception as e:
        print(f"单帧测试失败: {e}")
        return None

if __name__ == "__main__":
    # 主要测试
    weights, transforms = test_demBones_correct_api()
    
    # 单帧测试
    single_weights = test_single_frame()
    
    print("\n=== 总结 ===")
    if weights is not None:
        print(f"✅ DemBones API调用成功!")
        print(f"权重矩阵格式: {weights.shape}")
        print("📝 关键发现:")
        print("   - 必须使用属性设置而不是方法（nV, nB, nF等）")
        print("   - 必须设置u（rest pose）和v（animated poses）")
        print("   - 必须设置fStart和subjectID")
        print("   - 权重矩阵格式为(K, N)，需要转置为(N, K)使用")
    else:
        print("❌ DemBones API调用失败")
