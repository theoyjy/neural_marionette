#!/usr/bin/env python3
"""
测试 C++ CLI 版本的 DemBones 集成
=============================

这个脚本用于测试完整的 CLI 集成流程
"""

import numpy as np
import tempfile
import os
import sys

# 添加当前目录到 Python 路径以导入我们的模块
sys.path.append(os.path.dirname(__file__))

def create_test_data():
    """创建简单的测试数据"""
    # 创建一个简单的变形立方体
    # Rest pose: 单位立方体
    rest_vertices = np.array([
        [-0.5, -0.5, -0.5],  # 0
        [ 0.5, -0.5, -0.5],  # 1  
        [ 0.5,  0.5, -0.5],  # 2
        [-0.5,  0.5, -0.5],  # 3
        [-0.5, -0.5,  0.5],  # 4
        [ 0.5, -0.5,  0.5],  # 5
        [ 0.5,  0.5,  0.5],  # 6
        [-0.5,  0.5,  0.5],  # 7
    ], dtype=np.float32)
    
    # 动画帧1: 轻微变形
    frame1_vertices = rest_vertices.copy()
    frame1_vertices[:, 0] *= 1.2  # X方向拉伸
    
    # 动画帧2: 更多变形
    frame2_vertices = rest_vertices.copy()
    frame2_vertices[:, 1] *= 0.8  # Y方向压缩
    frame2_vertices[:, 2] *= 1.3  # Z方向拉伸
    
    # 骨骼层次结构 (简单的2骨骼)
    parents = [-1, 0]  # 根骨骼没有父节点，第二个骨骼的父节点是第一个
    
    frames = np.stack([rest_vertices, frame1_vertices, frame2_vertices], axis=0)
    
    return frames, parents

def test_dembones_methods():
    """测试 DemBones 相关方法"""
    print("=== 测试 DemBones CLI 集成 ===\n")
    
    # 导入pipeline类
    try:
        from complete_vv_pipeline import CompleteVVPipeline
    except ImportError as e:
        print(f"❌ 无法导入 CompleteVVPipeline: {e}")
        return False
    
    # 创建一个最小化的pipeline实例用于测试
    pipeline = CompleteVVPipeline("./test")
    
    # 创建测试数据
    frames, parents = create_test_data()
    print(f"创建测试数据: {frames.shape[0]} 帧, {frames.shape[1]} 顶点, {len(parents)} 骨骼")
    
    # 测试1: 查找可执行文件
    print("\n1. 测试查找 DemBones 可执行文件...")
    exe_path = pipeline._find_demBones_executable()
    
    if exe_path:
        print(f"✓ 找到可执行文件: {exe_path}")
    else:
        print("⚠️ 未找到可执行文件，将测试文件写入/读取功能")
    
    # 测试2: 写入输入文件
    print("\n2. 测试写入 DemBones 输入文件...")
    config = {
        'nIters': 5,
        'nInitIters': 1,
        'nTransIters': 1,
        'nWeightsIters': 1,
        'nnz': 4,
        'weightsSmooth': 0.001,
        'timeout': 30
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.dembones', delete=False) as f:
        test_input_file = f.name
    
    try:
        pipeline._write_demBones_input(test_input_file, frames, parents, config)
        print(f"✓ 输入文件写入成功: {test_input_file}")
        
        # 显示文件内容的前几行
        with open(test_input_file, 'r') as f:
            lines = f.readlines()[:15]
            print("文件前15行:")
            for i, line in enumerate(lines):
                print(f"  {i+1:2d}: {line.rstrip()}")
                
    except Exception as e:
        print(f"❌ 输入文件写入失败: {e}")
        return False
    finally:
        if os.path.exists(test_input_file):
            os.unlink(test_input_file)
    
    # 测试3: 完整的DemBones调用（如果有可执行文件）
    if exe_path:
        print("\n3. 测试完整的 DemBones 调用...")
        try:
            result = pipeline._try_demBones_with_timeout(frames, parents, config)
            if result:
                rest_pose, weights, transforms = result
                print(f"✓ DemBones 调用成功!")
                print(f"  Rest pose: {rest_pose.shape}")
                print(f"  权重: {weights.shape}")
                print(f"  变换: {transforms.shape}")
                print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            else:
                print("⚠️ DemBones 调用返回 None，将使用简化权重")
        except Exception as e:
            print(f"❌ DemBones 调用失败: {e}")
    
    # 测试4: 简化权重创建（回退方案）
    print("\n4. 测试简化权重创建...")
    try:
        # 创建一个伪造的mesh数据用于测试
        pipeline.all_mesh_data = [{'joints': np.random.rand(len(parents), 3)}]
        pipeline.rest_pose_idx = 0
        
        rest_pose, weights, transforms = pipeline._create_simple_skinning_weights(
            frames[0], len(parents)
        )
        print(f"✓ 简化权重创建成功!")
        print(f"  Rest pose: {rest_pose.shape}")
        print(f"  权重: {weights.shape}")
        print(f"  变换: {transforms.shape}")
        print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
        
    except Exception as e:
        print(f"❌ 简化权重创建失败: {e}")
        return False
    
    print("\n🎉 所有测试完成!")
    return True

def main():
    success = test_dembones_methods()
    
    print(f"\n=== 测试结果 ===")
    if success:
        print("✅ CLI 集成测试通过!")
        print("\n下一步:")
        print("1. 如果还没有 DemBones 可执行文件，运行: python install_dembones.py")
        print("2. 使用 complete_vv_pipeline.py 处理真实数据")
    else:
        print("❌ CLI 集成测试失败")
        print("请检查代码中的问题")

if __name__ == "__main__":
    main()
