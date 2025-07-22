import numpy as np
import time
try:
    import py_dem_bones as pdb
    print("✅ 成功导入 py_dem_bones")
except ImportError as e:
    print(f"❌ 无法导入 py_dem_bones: {e}")
    exit(1)

def test_distinct_bone_deformation():
    """测试明显区分的骨骼变形"""
    print("=== 测试明显区分的骨骼变形 ===")
    
    # 创建8个顶点的立方体
    rest_vertices = np.array([
        # 左半部分（应该受骨骼1影响）
        [0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
        # 右半部分（应该受骨骼2影响）
        [2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1]
    ], dtype=np.float32)
    
    # 创建多帧明显不同的变形
    # 帧1：左边绕Y轴旋转，右边保持不动
    frame1 = rest_vertices.copy()
    # 左半部分旋转90度
    cos_theta, sin_theta = 0, 1  # 90度
    for i in range(4):
        x, z = frame1[i, 0], frame1[i, 2]
        frame1[i, 0] = cos_theta * x - sin_theta * z
        frame1[i, 2] = sin_theta * x + cos_theta * z
    
    # 帧2：右边绕Y轴旋转，左边保持不动
    frame2 = rest_vertices.copy()
    # 右半部分旋转-90度
    cos_theta, sin_theta = 0, -1  # -90度
    for i in range(4, 8):
        x, z = frame2[i, 0] - 2, frame2[i, 2]  # 相对于(2,0,0)旋转
        frame2[i, 0] = 2 + cos_theta * x - sin_theta * z
        frame2[i, 2] = sin_theta * x + cos_theta * z
    
    # 帧3：左边上移，右边下移
    frame3 = rest_vertices.copy()
    frame3[:4, 1] += 1.0   # 左边上移
    frame3[4:, 1] -= 1.0   # 右边下移
    
    # 准备数据
    rest_pose = rest_vertices
    animated_poses = np.concatenate([frame1, frame2, frame3], axis=0)
    
    print(f"Rest pose shape: {rest_pose.shape}")
    print(f"Animated poses shape: {animated_poses.shape}")
    
    # 计算变形幅度验证
    deform1 = np.linalg.norm(frame1 - rest_vertices, axis=1)
    deform2 = np.linalg.norm(frame2 - rest_vertices, axis=1)
    deform3 = np.linalg.norm(frame3 - rest_vertices, axis=1)
    print(f"帧1变形幅度: 左半部分={deform1[:4].mean():.3f}, 右半部分={deform1[4:].mean():.3f}")
    print(f"帧2变形幅度: 左半部分={deform2[:4].mean():.3f}, 右半部分={deform2[4:].mean():.3f}")
    print(f"帧3变形幅度: 左半部分={deform3[:4].mean():.3f}, 右半部分={deform3[4:].mean():.3f}")
    
    # 初始化DemBones
    db = pdb.DemBones()
    
    N = rest_pose.shape[0]  # 8个顶点
    K = 2  # 2个骨骼
    F = 4  # 总帧数（包括rest pose）
    
    db.nV = N
    db.nB = K
    db.nF = F - 1  # 3个动画帧
    db.nS = 1
    db.fStart = np.array([0], dtype=np.int32)
    db.subjectID = np.zeros(F - 1, dtype=np.int32)
    db.u = rest_pose
    db.v = animated_poses
    
    # 更积极的参数设置
    db.nIters = 100
    db.nInitIters = 20
    db.nWeightsIters = 15
    db.nTransIters = 15
    db.weightsSmooth = 0.01  # 增加权重平滑
    
    print(f"DemBones配置: nV={N}, nB={K}, nF={F-1}")
    print(f"迭代参数: total={db.nIters}, init={db.nInitIters}, weights={db.nWeightsIters}, trans={db.nTransIters}")
    
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
            
            # 分析每行权重
            for i in range(weights.shape[0]):
                print(f"权重行{i+1}: {weights[i]}")
                left_weights = weights[i, :4].mean()   # 左半部分平均权重
                right_weights = weights[i, 4:].mean()  # 右半部分平均权重
                print(f"  左半部分平均权重: {left_weights:.4f}")
                print(f"  右半部分平均权重: {right_weights:.4f}")
            
            # 检查是否识别出了多个骨骼
            if weights.shape[0] == K:
                print(f"✅ 成功识别出{K}个骨骼!")
                return weights, transforms
            else:
                print(f"⚠️  只识别出{weights.shape[0]}个骨骼，期望{K}个")
                
                # 尝试理解为什么只有一个骨骼
                print("\n🔍 分析原因:")
                unique_weights = np.unique(weights)
                print(f"权重唯一值: {unique_weights}")
                if len(unique_weights) == 1:
                    print("所有权重相同，可能是算法收敛到平凡解")
                
            return weights, transforms
        else:
            print("❌ 权重矩阵为空")
            return None, None
            
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_extreme_deformation():
    """测试极端变形情况"""
    print("\n=== 测试极端变形情况 ===")
    
    # 只有4个顶点，2个骨骼
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0],  # 左边2个顶点（骨骼1）
        [3, 0, 0], [4, 0, 0]   # 右边2个顶点（骨骼2）
    ], dtype=np.float32)
    
    # 极端变形：左边向左，右边向右
    frame1 = rest_vertices.copy()
    frame1[:2, 0] -= 2.0   # 左边大幅向左
    frame1[2:, 0] += 2.0   # 右边大幅向右
    
    db = pdb.DemBones()
    db.nV = 4
    db.nB = 2
    db.nF = 1
    db.nS = 1
    db.fStart = np.array([0], dtype=np.int32)
    db.subjectID = np.zeros(1, dtype=np.int32)
    db.u = rest_vertices
    db.v = frame1
    
    db.nIters = 50
    db.nInitIters = 15
    db.nWeightsIters = 10
    db.nTransIters = 10
    
    print(f"极端变形测试: {db.nV}顶点, {db.nB}骨骼")
    print(f"变形幅度: 左边={np.linalg.norm(frame1[:2] - rest_vertices[:2], axis=1).mean():.3f}")
    print(f"变形幅度: 右边={np.linalg.norm(frame1[2:] - rest_vertices[2:], axis=1).mean():.3f}")
    
    try:
        db.compute()
        weights = db.get_weights()
        print(f"极端变形权重shape: {weights.shape}")
        print(f"极端变形权重内容:\n{weights}")
        return weights
    except Exception as e:
        print(f"极端变形测试失败: {e}")
        return None

if __name__ == "__main__":
    # 明显区分的骨骼变形测试
    weights, transforms = test_distinct_bone_deformation()
    
    # 极端变形测试
    extreme_weights = test_extreme_deformation()
    
    print("\n=== 总结 ===")
    if weights is not None and weights.shape[0] > 1:
        print(f"✅ 成功识别出多个骨骼! 权重矩阵: {weights.shape}")
        print("🎯 DemBones权重矩阵格式确认:")
        print(f"   - 实际格式: {weights.shape}")
        print(f"   - 期望格式: (nB, nV) = ({weights.shape[0]}, {weights.shape[1]})")
        print("   - 在管道中使用时需要转置为(nV, nB)格式")
    else:
        print("⚠️  DemBones倾向于收敛到单骨骼解决方案")
        print("📝 可能的解决方案:")
        print("   1. 增加变形的多样性和幅度")
        print("   2. 调整DemBones参数（迭代次数、平滑参数等）")
        print("   3. 使用更多的训练帧")
        print("   4. 在管道中适应单骨骼或少骨骼的结果")
