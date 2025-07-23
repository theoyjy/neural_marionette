#!/usr/bin/env python3
"""
最简单的LBS权重优化测试

这个版本只测试最基本的功能，没有复杂的优化
"""

import numpy as np
import sys
from pathlib import Path
from Skinning import InverseMeshCanonicalizer

def test_basic_lbs():
    """
    最基本的LBS测试 - 只测试LBS变换本身
    """
    print("=" * 80)
    print("基本LBS变换测试")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        # 创建统一化器
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        # 加载网格序列
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        print(f"顶点数: {len(canonicalizer.reference_mesh.vertices)}")
        print(f"关节数: {canonicalizer.num_joints}")
        
        # 设置rest pose
        rest_vertices = np.asarray(canonicalizer.reference_mesh.vertices)
        
        # 计算归一化参数并归一化顶点
        rest_norm_params = canonicalizer.compute_mesh_normalization_params(canonicalizer.reference_mesh)
        rest_vertices_norm = canonicalizer.normalize_mesh_vertices(rest_vertices, rest_norm_params)
        
        # 创建简单的测试权重（基于距离）
        keypoints = canonicalizer.keypoints[0, :, :3]  # reference frame的关节点
        
        print("创建基于距离的权重...")
        distances = np.linalg.norm(rest_vertices_norm[:, None, :] - keypoints[None, :, :], axis=2)  # [V, J]
        weights = np.exp(-distances**2 / (2 * 0.1**2))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)  # 归一化
        
        print(f"权重矩阵形状: {weights.shape}")
        print(f"权重统计: min={np.min(weights):.4f}, max={np.max(weights):.4f}, mean={np.mean(weights):.4f}")
        
        # 测试LBS变换
        print("\n测试LBS变换...")
        rest_transforms = canonicalizer.transforms[0]  # reference frame
        target_transforms = canonicalizer.transforms[1]  # target frame
        
        # 计算相对变换
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(canonicalizer.num_joints):
            try:
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            except:
                relative_transforms[j] = np.eye(4)
        
        # 应用LBS变换
        transformed_vertices = canonicalizer.apply_lbs_transform(rest_vertices_norm, weights, relative_transforms)
        
        print(f"变换结果形状: {transformed_vertices.shape}")
        print(f"变换差异统计:")
        diff = transformed_vertices - rest_vertices_norm
        print(f"  - 平均差异: {np.mean(np.linalg.norm(diff, axis=1)):.6f}")
        print(f"  - 最大差异: {np.max(np.linalg.norm(diff, axis=1)):.6f}")
        
        # 与真实目标比较
        print("\n与真实目标比较...")
        target_mesh = canonicalizer.mesh_files[1]
        target_mesh_obj = __import__('open3d').io.read_triangle_mesh(str(target_mesh))
        target_vertices = np.asarray(target_mesh_obj.vertices)
        target_norm_params = canonicalizer.compute_mesh_normalization_params(target_mesh_obj)
        target_vertices_norm = canonicalizer.normalize_mesh_vertices(target_vertices, target_norm_params)
        
        error = np.linalg.norm(transformed_vertices - target_vertices_norm, axis=1)
        print(f"与真实目标的误差:")
        print(f"  - 平均误差: {np.mean(error):.6f}")
        print(f"  - 最大误差: {np.max(error):.6f}")
        print(f"  - 标准差: {np.std(error):.6f}")
        
        print("\n✅ 基本LBS变换测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 基本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_optimization():
    """
    简单优化测试 - 只优化很少的顶点
    """
    print("=" * 80)
    print("简单优化测试")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 设置rest pose
        canonicalizer.rest_pose_vertices = np.asarray(canonicalizer.reference_mesh.vertices)
        canonicalizer.rest_pose_transforms = canonicalizer.transforms[0]
        
        # 只使用前100个顶点进行测试
        num_test_vertices = 100
        rest_vertices = canonicalizer.rest_pose_vertices[:num_test_vertices]
        
        # 计算归一化
        rest_norm_params = canonicalizer.frame_normalization_params[0]
        rest_vertices_norm = canonicalizer.normalize_mesh_vertices(rest_vertices, rest_norm_params)
        
        # 目标数据
        target_frame = 1
        target_mesh = __import__('open3d').io.read_triangle_mesh(str(canonicalizer.mesh_files[target_frame]))
        target_vertices = np.asarray(target_mesh.vertices)[:num_test_vertices]
        target_norm_params = canonicalizer.compute_mesh_normalization_params(target_mesh)
        target_vertices_norm = canonicalizer.normalize_mesh_vertices(target_vertices, target_norm_params)
        
        # 相对变换
        rest_transforms = canonicalizer.transforms[0]
        target_transforms = canonicalizer.transforms[target_frame]
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(canonicalizer.num_joints):
            try:
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            except:
                relative_transforms[j] = np.eye(4)
        
        # 初始权重（基于距离）
        keypoints = canonicalizer.keypoints[0, :, :3]
        distances = np.linalg.norm(rest_vertices_norm[:, None, :] - keypoints[None, :, :], axis=2)
        weights_init = np.exp(-distances**2 / (2 * 0.1**2))
        weights_init = weights_init / (np.sum(weights_init, axis=1, keepdims=True) + 1e-8)
        
        print(f"测试顶点数: {num_test_vertices}")
        print(f"关节数: {canonicalizer.num_joints}")
        print(f"初始权重形状: {weights_init.shape}")
        
        # 简单的梯度下降优化
        print("\n开始简单梯度下降优化...")
        weights = weights_init.copy()
        learning_rate = 0.001
        num_iterations = 50
        
        for iteration in range(num_iterations):
            # 前向传播
            predicted = canonicalizer.apply_lbs_transform(rest_vertices_norm, weights, relative_transforms)
            
            # 计算损失
            error = predicted - target_vertices_norm
            loss = np.mean(np.sum(error**2, axis=1))
            
            if iteration % 10 == 0:
                print(f"  迭代 {iteration}: 损失 = {loss:.6f}")
            
            # 简单的梯度近似（有限差分）
            gradient = np.zeros_like(weights)
            eps = 1e-6
            
            for i in range(min(10, num_test_vertices)):  # 只计算前10个顶点的梯度
                for j in range(canonicalizer.num_joints):
                    # 前向差分
                    weights_plus = weights.copy()
                    weights_plus[i, j] += eps
                    # 重新归一化
                    weights_plus[i] = weights_plus[i] / (np.sum(weights_plus[i]) + 1e-8)
                    
                    predicted_plus = canonicalizer.apply_lbs_transform(rest_vertices_norm, weights_plus, relative_transforms)
                    loss_plus = np.mean(np.sum((predicted_plus - target_vertices_norm)**2, axis=1))
                    
                    gradient[i, j] = (loss_plus - loss) / eps
            
            # 更新权重
            weights -= learning_rate * gradient
            
            # 确保非负并重新归一化
            weights = np.maximum(weights, 0)
            weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
        
        # 最终评估
        final_predicted = canonicalizer.apply_lbs_transform(rest_vertices_norm, weights, relative_transforms)
        final_error = np.mean(np.linalg.norm(final_predicted - target_vertices_norm, axis=1))
        
        print(f"\n✅ 简单优化完成！")
        print(f"最终误差: {final_error:.6f}")
        print(f"权重稀疏度: {np.mean(weights > 0.01):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单优化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_optimization_issue():
    """
    调试优化问题
    """
    print("=" * 80)
    print("调试优化问题")
    print("=" * 80)
    
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    try:
        canonicalizer = InverseMeshCanonicalizer(
            skeleton_data_dir=skeleton_data_dir,
            reference_frame_idx=0
        )
        
        canonicalizer.load_mesh_sequence(mesh_folder_path)
        
        # 设置rest pose
        canonicalizer.rest_pose_vertices = np.asarray(canonicalizer.reference_mesh.vertices)
        canonicalizer.rest_pose_transforms = canonicalizer.transforms[0]
        
        print(f"Rest vertices形状: {canonicalizer.rest_pose_vertices.shape}")
        print(f"变换矩阵形状: {canonicalizer.transforms.shape}")
        
        # 测试单个优化函数的组件
        target_frame = 1
        
        # 检查每个步骤
        print("\n1. 检查网格加载...")
        target_mesh = __import__('open3d').io.read_triangle_mesh(str(canonicalizer.mesh_files[target_frame]))
        print(f"目标网格顶点数: {len(target_mesh.vertices)}")
        
        print("\n2. 检查归一化...")
        target_norm_params = canonicalizer.compute_mesh_normalization_params(target_mesh)
        print(f"归一化参数: {target_norm_params}")
        
        print("\n3. 检查相对变换计算...")
        rest_transforms = canonicalizer.transforms[0]
        target_transforms = canonicalizer.transforms[target_frame]
        print(f"Rest变换矩阵第一个关节:")
        print(rest_transforms[0])
        print(f"目标变换矩阵第一个关节:")
        print(target_transforms[0])
        
        # 检查矩阵可逆性
        print("\n4. 检查矩阵可逆性...")
        for j in range(min(5, canonicalizer.num_joints)):
            det = np.linalg.det(rest_transforms[j][:3, :3])
            print(f"关节 {j} 行列式: {det}")
            if abs(det) < 1e-6:
                print(f"  警告: 关节 {j} 的变换矩阵接近奇异!")
        
        print("\n5. 测试LBS损失函数...")
        # 创建一个小的测试案例
        test_vertices = 5
        test_rest = canonicalizer.rest_pose_vertices[:test_vertices]
        test_target = np.asarray(target_mesh.vertices)[:test_vertices]
        
        # 归一化
        rest_norm = canonicalizer.normalize_mesh_vertices(test_rest, canonicalizer.frame_normalization_params[0])
        target_norm = canonicalizer.normalize_mesh_vertices(test_target, target_norm_params)
        
        # 计算相对变换
        relative_transforms = np.zeros_like(target_transforms)
        for j in range(canonicalizer.num_joints):
            if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                rest_inv = np.linalg.inv(rest_transforms[j])
                relative_transforms[j] = target_transforms[j] @ rest_inv
            else:
                relative_transforms[j] = np.eye(4)
        
        # 创建测试权重
        test_weights = np.random.rand(test_vertices, canonicalizer.num_joints)
        test_weights = test_weights / (np.sum(test_weights, axis=1, keepdims=True) + 1e-8)
        
        print(f"测试权重形状: {test_weights.shape}")
        print(f"测试rest顶点形状: {rest_norm.shape}")
        print(f"测试target顶点形状: {target_norm.shape}")
        
        # 测试损失函数
        test_weights_flat = test_weights.flatten()
        loss = canonicalizer.compute_lbs_loss(test_weights_flat, rest_norm, target_norm, relative_transforms)
        print(f"测试损失: {loss}")
        
        print("\n✅ 调试完成！组件看起来正常工作")
        return True
        
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 基本LBS变换测试")
    print("2. 简单优化测试")
    print("3. 调试优化问题")
    
    try:
        choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == "1":
            success = test_basic_lbs()
        elif choice == "2":
            success = test_simple_optimization()
        elif choice == "3":
            success = debug_optimization_issue()
        else:
            print("无效选择，默认运行基本测试")
            success = test_basic_lbs()
        
        if success:
            print("\n🎉 测试成功完成！")
        else:
            print("\n❌ 测试失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试出错: {e}")
        sys.exit(1)
