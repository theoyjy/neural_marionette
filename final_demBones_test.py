import numpy as np
import time
try:
    import py_dem_bones as pdb
    print("✅ 成功导入 py_dem_bones")
except ImportError as e:
    print(f"❌ 无法导入 py_dem_bones: {e}")
    exit(1)

def test_demBones_complete_SSDR():
    """测试DemBones完整的SSDR输出：Rest Pose Skeleton + Skinning Weights + Per-frame Bone Transforms"""
    print("=== DemBones完整SSDR测试 ===")
    print("提取：1. Bind Pose Skeleton  2. Skinning Weights  3. Per-frame Bone Transforms")
    
    # 创建有意义的变形数据
    # 8个顶点的立方体
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 顶面
    ], dtype=np.float32)
    
    # 创建3帧明显不同的变形
    frame1 = rest_vertices.copy()
    frame1[:4, 0] -= 0.8  # 左半部分大幅向左移动
    
    frame2 = rest_vertices.copy()
    frame2[4:, 0] += 0.8  # 右半部分大幅向右移动
    
    frame3 = rest_vertices.copy()
    frame3[:4, 1] += 0.6  # 左半部分向上移动
    frame3[4:, 1] -= 0.6  # 右半部分向下移动
    
    # 组合所有帧
    all_frames = np.concatenate([frame1, frame2, frame3], axis=0)  # (24, 3)
    
    print(f"Rest pose shape: {rest_vertices.shape}")
    print(f"Animated poses shape: {all_frames.shape}")
    print(f"变形幅度范围: {np.linalg.norm(all_frames.reshape(3, 8, 3) - rest_vertices[None, :, :], axis=2).max():.3f}")
    
    # 初始化DemBones - 使用正确的API
    db = pdb.DemBones()
    
    # 使用正确的属性设置方式（不是旧的nS方式）
    N = rest_vertices.shape[0]  # 8个顶点
    K = 2  # 2个骨骼  
    F = 4  # 总帧数（包括rest pose）
    
    db.nV = N
    db.nB = K
    db.nF = F - 1  # 动画帧数（不包括rest pose）
    db.nS = 1      # 主题数
    db.fStart = np.array([0], dtype=np.int32)
    db.subjectID = np.zeros(F - 1, dtype=np.int32)
    db.u = rest_vertices  # Rest pose (N, 3)
    db.v = all_frames     # Animated poses (3*N, 3)
    
    # 更保守的参数设置，避免崩溃
    db.nInitIters = 10
    db.nIters = 30
    db.nWeightsIters = 5
    db.nTransIters = 5
    db.weightsSmooth = 0.01
    
    print(f"DemBones配置: nV={N}, nB={K}, nF={F-1}")
    print(f"迭代参数: init={db.nInitIters}, total={db.nIters}, weights={db.nWeightsIters}, trans={db.nTransIters}")
    print(f"数据格式: rest_pose={db.u.shape}, animated_poses={db.v.shape}")
    
    # 计算
    print("🚀 开始计算...")
    try:
        start_time = time.time()
        db.compute()
        elapsed = time.time() - start_time
        print(f"✅ 计算完成 (耗时 {elapsed:.2f}s)")
        
        # 获取完整的SSDR结果
        weights = db.get_weights()           # Skinning Weights W
        transforms = db.get_transformations() # Per-frame Bone Transforms B_t
        rest_pose_skel = db.get_rest_pose()  # Rest/Bind Pose Skeleton
        
        print(f"\n📊 完整SSDR结果分析:")
        print(f"1. Skinning Weights shape: {weights.shape}")
        print(f"2. Per-frame Transforms shape: {transforms.shape}")  
        print(f"3. Rest Pose Skeleton shape: {rest_pose_skel.shape if hasattr(rest_pose_skel, 'shape') else type(rest_pose_skel)}")
        
        # 详细分析每个组件
        print(f"\n🦴 1. Rest/Bind Pose Skeleton 分析:")
        if hasattr(rest_pose_skel, 'shape'):
            print(f"   骨骼姿态数据: {rest_pose_skel.shape}")
            print(f"   骨骼姿态内容: {rest_pose_skel}")
        else:
            print(f"   骨骼姿态类型: {type(rest_pose_skel)}")
        
        print(f"\n⚖️ 2. Skinning Weights W 分析:")
        print(f"   权重矩阵shape: {weights.shape}")
        if weights.size > 0:
            print(f"   权重矩阵内容:\n{weights}")
            print(f"   权重值范围: [{weights.min():.4f}, {weights.max():.4f}]")
            print(f"   非零权重比例: {np.count_nonzero(weights)/weights.size:.2%}")
            
            # 检查权重稀疏性和归一化
            if len(weights.shape) == 2:
                if weights.shape[0] == db.nB and weights.shape[1] == db.nS:
                    # (nB, nS) 格式
                    vertex_sums = np.sum(weights, axis=0)
                    print(f"   每个顶点权重和: {vertex_sums}")
                    print(f"   权重归一化检查: {np.allclose(vertex_sums, 1.0, atol=0.01)}")
                elif weights.shape[0] == db.nS and weights.shape[1] == db.nB:
                    # (nS, nB) 格式  
                    vertex_sums = np.sum(weights, axis=1)
                    print(f"   每个顶点权重和: {vertex_sums}")
                    print(f"   权重归一化检查: {np.allclose(vertex_sums, 1.0, atol=0.01)}")
        
        print(f"\n🎯 3. Per-frame Bone Transforms B_t 分析:")
        print(f"   变换矩阵shape: {transforms.shape}")
        if transforms.size > 0:
            print(f"   变换矩阵内容预览:")
            if len(transforms.shape) == 3:  # (nF, nB, 4, 4) 或类似格式
                print(f"   第0帧第0骨骼变换:\n{transforms[0, 0] if transforms.shape[1] > 0 else 'N/A'}")
                print(f"   第0帧第1骨骼变换:\n{transforms[0, 1] if transforms.shape[1] > 1 else 'N/A'}")
            else:
                print(f"   变换数据: {transforms}")
        
        # 验证LBS参数完整性
        print(f"\n✅ LBS参数完整性检查:")
        has_weights = weights.size > 0
        has_transforms = transforms.size > 0 
        has_skeleton = rest_pose_skel is not None
        
        print(f"   ✅ Skinning Weights: {'有效' if has_weights else '无效'}")
        print(f"   ✅ Bone Transforms: {'有效' if has_transforms else '无效'}")
        print(f"   ✅ Rest Pose Skeleton: {'有效' if has_skeleton else '无效'}")
        
        complete_lbs = has_weights and has_transforms and has_skeleton
        print(f"   🎯 完整LBS参数集: {'✅ 完整' if complete_lbs else '❌ 不完整'}")
        
        return {
            'weights': weights,
            'transforms': transforms,
            'rest_skeleton': rest_pose_skel,
            'complete': complete_lbs
        }
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        return None, None

def test_different_bone_counts():
    """测试不同骨骼数量的情况"""
    print("\n=== 测试不同骨骼数量 ===")
    
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    # 测试1-4个骨骼
    for nB in range(1, 5):
        print(f"\n测试 {nB} 个骨骼:")
        
        # 简单变形
        frame1 = rest_vertices + [0.1 * nB, 0, 0]
        
        db = pdb.DemBones()
        db.nS = 4
        db.nB = nB
        db.nF = 1
        db.nIters = 20
        
        db.set_rest_pose(rest_vertices.flatten())
        db.set_animated_poses(frame1.flatten())
        
        try:
            db.compute()
            weights = db.get_weights()
            print(f"  权重shape: {weights.shape}, 期望: ({nB}, 4) 或 (4, {nB})")
            print(f"  权重内容: {weights}")
        except Exception as e:
            print(f"  失败: {e}")

if __name__ == "__main__":
    # 主要测试：完整SSDR输出
    ssdr_results = test_demBones_complete_SSDR()
    
    # 不同骨骼数量测试
    test_different_bone_counts()
    
    print("\n=== 总结 ===")
    if ssdr_results and ssdr_results['complete']:
        print(f"✅ DemBones SSDR完整输出成功!")
        print(f"📝 获得完整LBS参数集:")
        print(f"   - Skinning Weights: {ssdr_results['weights'].shape}")
        print(f"   - Bone Transforms: {ssdr_results['transforms'].shape}")
        print(f"   - Rest Skeleton: 已提取")
        print("🎯 可用于完整的骨骼动画管道!")
    else:
        print("❌ DemBones SSDR输出不完整，需要进一步调试")
