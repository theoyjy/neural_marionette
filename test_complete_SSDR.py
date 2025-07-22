import numpy as np
import time
try:
    import py_dem_bones as pdb
    print("✅ 成功导入 py_dem_bones")
except ImportError as e:
    print(f"❌ 无法导入 py_dem_bones: {e}")
    exit(1)

def test_demBones_correct_SSDR_API():
    """使用正确的API测试DemBones完整SSDR输出"""
    print("=== 使用正确API测试DemBones SSDR ===")
    
    # 创建测试数据
    rest_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
    ], dtype=np.float32)
    
    # 创建2帧变形
    frame1 = rest_vertices + [0.5, 0, 0]
    frame2 = rest_vertices + [0, 0.5, 0]
    
    # 使用成功脚本的API方式
    rest_pose = rest_vertices  # (N, 3)
    animated_poses = np.concatenate([frame1, frame2], axis=0)  # (2*N, 3)
    
    print(f"Rest pose shape: {rest_pose.shape}")
    print(f"Animated poses shape: {animated_poses.shape}")
    
    # 使用成功脚本的设置方式
    db = pdb.DemBones()
    
    N = rest_pose.shape[0]  # 4个顶点
    K = 2  # 2个骨骼
    F = 3   # 总帧数（包括rest pose）
    
    # 关键：使用属性而不是方法设置
    db.nV = N
    db.nB = K
    db.nF = F - 1  # 动画帧数
    db.nS = 1      # 主题数
    db.fStart = np.array([0], dtype=np.int32)
    db.subjectID = np.zeros(F - 1, dtype=np.int32)
    db.u = rest_pose     # Rest pose
    db.v = animated_poses # Animated poses
    
    # 参数设置
    db.nIters = 50
    db.nInitIters = 10
    db.nWeightsIters = 8
    db.nTransIters = 8
    
    print(f"DemBones配置: nV={N}, nB={K}, nF={F-1}")
    
    try:
        print("🚀 开始SSDR计算...")
        start_time = time.time()
        db.compute()
        elapsed = time.time() - start_time
        print(f"✅ 计算完成 (耗时 {elapsed:.2f}s)")
        
        # 提取完整SSDR结果
        print(f"\n🔍 提取SSDR三大组件:")
        
        # 1. Skinning Weights
        weights = db.get_weights()
        print(f"1️⃣ Skinning Weights shape: {weights.shape}")
        
        # 2. Per-frame Bone Transforms  
        transforms = db.get_transformations()
        print(f"2️⃣ Bone Transforms shape: {transforms.shape}")
        
        # 3. Rest/Bind Pose Skeleton
        # 尝试不同的方法获取骨骼信息
        try:
            rest_skel = db.get_rest_pose()
            print(f"3️⃣ Rest Skeleton (get_rest_pose) shape: {rest_skel.shape if hasattr(rest_skel, 'shape') else type(rest_skel)}")
        except:
            rest_skel = None
            print(f"3️⃣ Rest Skeleton (get_rest_pose): 方法不存在")
        
        # 尝试获取骨骼层级信息
        try:
            # 检查db对象的所有属性，寻找骨骼相关信息
            bone_attrs = [attr for attr in dir(db) if 'bone' in attr.lower() or 'skel' in attr.lower() or 'm' == attr or 'u' == attr]
            print(f"🦴 可能的骨骼相关属性: {bone_attrs}")
            
            # 检查m属性（可能是骨骼变换）
            if hasattr(db, 'm') and hasattr(db.m, 'shape'):
                print(f"   db.m shape: {db.m.shape}")
                print(f"   db.m 内容: {db.m}")
                
        except Exception as e:
            print(f"获取骨骼属性失败: {e}")
        
        # 详细分析结果
        print(f"\n📊 详细SSDR分析:")
        
        # 分析Skinning Weights
        if weights.size > 0:
            print(f"\n⚖️ Skinning Weights 详细分析:")
            print(f"   Shape: {weights.shape}")
            print(f"   内容:\n{weights}")
            print(f"   值范围: [{weights.min():.4f}, {weights.max():.4f}]")
            
            # 判断格式并检查归一化
            if len(weights.shape) == 2:
                rows, cols = weights.shape
                if rows == K and cols == N:
                    print(f"   格式: (nB={K}, nV={N}) - 标准格式")
                    vertex_sums = weights.sum(axis=0)
                    print(f"   每个顶点权重和: {vertex_sums}")
                    print(f"   归一化检查: {np.allclose(vertex_sums, 1.0, atol=0.01)}")
                elif rows == N and cols == K:
                    print(f"   格式: (nV={N}, nB={K}) - 转置格式")
                    vertex_sums = weights.sum(axis=1)
                    print(f"   每个顶点权重和: {vertex_sums}")
                    print(f"   归一化检查: {np.allclose(vertex_sums, 1.0, atol=0.01)}")
                else:
                    print(f"   格式: ({rows}, {cols}) - 非标准格式")
        else:
            print(f"\n⚠️ Skinning Weights 为空")
        
        # 分析Bone Transforms
        if transforms.size > 0:
            print(f"\n🎯 Bone Transforms 详细分析:")
            print(f"   Shape: {transforms.shape}")
            print(f"   期望格式: (nF={F-1}, nB={K}, 4, 4)")
            
            if len(transforms.shape) == 3:
                nf, nb, mat_size = transforms.shape
                print(f"   实际: ({nf}帧, {nb}骨骼, {mat_size}x{mat_size}变换)")
                if nf > 0 and nb > 0:
                    print(f"   第0帧第0骨骼变换:\n{transforms[0, 0]}")
            else:
                print(f"   非标准变换格式: {transforms.shape}")
        else:
            print(f"\n⚠️ Bone Transforms 为空")
        
        # 总结SSDR完整性
        has_weights = weights.size > 0
        has_transforms = transforms.size > 0
        has_skeleton = rest_skel is not None
        
        print(f"\n✅ SSDR完整性总结:")
        print(f"   Skinning Weights W: {'✅ 有效' if has_weights else '❌ 无效'}")
        print(f"   Bone Transforms B_t: {'✅ 有效' if has_transforms else '❌ 无效'}")
        print(f"   Rest Pose Skeleton: {'✅ 有效' if has_skeleton else '❌ 无效'}")
        
        complete = has_weights and has_transforms
        print(f"   🎯 SSDR完整度: {'✅ 完整' if complete else '❌ 部分/失败'}")
        
        return {
            'weights': weights,
            'transforms': transforms, 
            'rest_skeleton': rest_skel,
            'complete': complete
        }
        
    except Exception as e:
        print(f"❌ SSDR计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_demBones_correct_SSDR_API()
    
    if results and results['complete']:
        print(f"\n🎉 DemBones SSDR完整输出成功!")
        print(f"✅ 获得完整LBS参数集合，可用于骨骼动画管道")
    else:
        print(f"\n⚠️ DemBones SSDR输出不完整")
        print(f"📝 可能需要调整数据格式或参数设置")
