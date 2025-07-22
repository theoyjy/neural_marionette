import numpy as np
import demBones

def create_meaningful_deformation():
    """创建有意义的骨骼变形数据"""
    # 简单的8顶点立方体
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # 底面
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # 顶面
    ], dtype=np.float32)
    
    print(f"Rest pose vertices:\n{rest_vertices}")
    
    # 创建两帧有意义的变形
    # 帧1：骨骼1影响左半部分，骨骼2影响右半部分
    frame1 = rest_vertices.copy()
    frame1[:4, 0] -= 0.5  # 左半部分向左移动（骨骼1）
    frame1[4:, 0] += 0.5  # 右半部分向右移动（骨骼2）
    
    # 帧2：不同的变形
    frame2 = rest_vertices.copy()
    frame2[:4, 1] += 0.3  # 左半部分向上移动（骨骼1）
    frame2[4:, 1] -= 0.3  # 右半部分向下移动（骨骼2）
    
    # 组合所有帧
    all_frames = np.concatenate([frame1, frame2], axis=0)  # (16, 3)
    
    print(f"Frame 1 vertices:\n{frame1}")
    print(f"Frame 2 vertices:\n{frame2}")
    print(f"All frames shape: {all_frames.shape}")
    
    # 计算变形幅度
    deformation1 = np.linalg.norm(frame1 - rest_vertices, axis=1)
    deformation2 = np.linalg.norm(frame2 - rest_vertices, axis=1)
    print(f"Frame 1 deformation magnitudes: {deformation1}")
    print(f"Frame 2 deformation magnitudes: {deformation2}")
    
    return rest_vertices, all_frames

def test_demBones_with_better_data():
    """使用更好的数据测试DemBones"""
    print("=== 测试有意义的骨骼变形数据 ===")
    
    rest_vertices, animated_vertices = create_meaningful_deformation()
    
    # 初始化DemBones
    db = demBones.DemBones()
    db.nS = 8    # 8个顶点
    db.nB = 2    # 2个骨骼
    db.nF = 2    # 2帧
    db.nInitIters = 15  # 增加初始化迭代次数
    db.nIters = 50      # 增加总迭代次数
    db.nWeightsIters = 5 # 增加权重迭代次数
    db.nTransIters = 5   # 增加变换迭代次数
    
    # 设置数据
    db.set_rest_pose(rest_vertices.flatten())
    db.set_animated_poses(animated_vertices.flatten())
    
    print(f"DemBones参数设置:")
    print(f"  顶点数: {db.nS}, 骨骼数: {db.nB}, 帧数: {db.nF}")
    print(f"  迭代次数: init={db.nInitIters}, total={db.nIters}")
    print(f"  权重迭代: {db.nWeightsIters}, 变换迭代: {db.nTransIters}")
    
    # 计算
    print("开始计算...")
    try:
        db.compute()
        print("✓ 计算完成!")
        
        # 获取结果
        weights = db.get_weights()
        transforms = db.get_transformations()
        
        print(f"\n权重矩阵 shape: {weights.shape}")
        print(f"变换矩阵 shape: {transforms.shape}")
        
        if weights.size > 0:
            weights_2d = weights.reshape(-1, db.nS)
            print(f"重塑后权重矩阵 shape: {weights_2d.shape}")
            print(f"权重矩阵内容:\n{weights_2d}")
            
            # 检查权重的合理性
            print(f"\n权重分析:")
            for i in range(weights_2d.shape[0]):
                print(f"  骨骼{i+1}权重: min={weights_2d[i].min():.3f}, "
                      f"max={weights_2d[i].max():.3f}, "
                      f"mean={weights_2d[i].mean():.3f}")
            
            # 检查每个顶点的权重和
            if weights_2d.shape[0] == db.nB:
                vertex_sums = np.sum(weights_2d, axis=0)
                print(f"  每个顶点权重和: {vertex_sums}")
                print(f"  权重和是否归一化: {np.allclose(vertex_sums, 1.0)}")
        
        return weights, transforms
        
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        return None, None

def test_single_bone():
    """测试单骨骼情况"""
    print("\n=== 测试单骨骼情况 ===")
    
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    # 单骨骼刚体变换
    frame1 = rest_vertices + [0.5, 0, 0]  # 平移
    all_frames = frame1
    
    db = demBones.DemBones()
    db.nS = 4
    db.nB = 1
    db.nF = 1
    
    db.set_rest_pose(rest_vertices.flatten())
    db.set_animated_poses(all_frames.flatten())
    
    print(f"单骨骼测试: {db.nS}顶点, {db.nB}骨骼, {db.nF}帧")
    
    try:
        db.compute()
        weights = db.get_weights()
        print(f"单骨骼权重 shape: {weights.shape}")
        print(f"单骨骼权重内容: {weights}")
        return weights
    except Exception as e:
        print(f"单骨骼测试失败: {e}")
        return None

if __name__ == "__main__":
    # 测试有意义的变形数据
    weights, transforms = test_demBones_with_better_data()
    
    # 测试单骨骼情况
    single_weights = test_single_bone()
    
    print("\n=== 总结 ===")
    if weights is not None:
        print(f"多骨骼权重矩阵: {weights.shape}")
    if single_weights is not None:
        print(f"单骨骼权重矩阵: {single_weights.shape}")
