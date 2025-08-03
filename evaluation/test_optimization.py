#!/usr/bin/env python3
"""
评估优化效果测试脚本
对比快速模式和精确模式的性能差异
"""

import time
import numpy as np
from utils_mesh import *

def generate_test_data():
    """生成测试用的网格数据"""
    # 生成两个相似的点云
    np.random.seed(42)
    vertices_a = np.random.randn(5000, 3) * 0.5
    vertices_b = vertices_a + np.random.randn(5000, 3) * 0.01  # 添加小扰动
    
    return vertices_a, vertices_b

def test_chamfer_distance():
    """测试Chamfer距离计算优化"""
    print("=== Chamfer距离计算测试 ===")
    vertices_a, vertices_b = generate_test_data()
    
    # 精确模式
    start_time = time.time()
    chamfer_exact = compute_chamfer_distance(vertices_a, vertices_b, subsample_ratio=1.0)
    exact_time = time.time() - start_time
    
    # 快速模式
    start_time = time.time()
    chamfer_fast = compute_chamfer_distance(vertices_a, vertices_b, subsample_ratio=0.05)
    fast_time = time.time() - start_time
    
    speedup = exact_time / fast_time if fast_time > 0 else float('inf')
    error = abs(chamfer_exact - chamfer_fast) / chamfer_exact * 100
    
    print(f"精确模式: {chamfer_exact:.6f} ({exact_time:.3f}s)")
    print(f"快速模式: {chamfer_fast:.6f} ({fast_time:.3f}s)")
    print(f"加速比: {speedup:.1f}x")
    print(f"相对误差: {error:.2f}%")
    print()

def test_arap_error():
    """测试ARAP误差计算优化"""
    print("=== ARAP误差计算测试 ===")
    vertices_a, vertices_b = generate_test_data()
    
    # 精确模式
    start_time = time.time()
    arap_exact = compute_arap_error(vertices_a, vertices_b, sample_ratio=0.1)
    exact_time = time.time() - start_time
    
    # 快速模式
    start_time = time.time()
    arap_fast = compute_arap_error(vertices_a, vertices_b, sample_ratio=0.02)
    fast_time = time.time() - start_time
    
    speedup = exact_time / fast_time if fast_time > 0 else float('inf')
    if arap_exact > 0:
        error = abs(arap_exact - arap_fast) / arap_exact * 100
    else:
        error = 0
    
    print(f"精确模式: {arap_exact:.6f} ({exact_time:.3f}s)")
    print(f"快速模式: {arap_fast:.6f} ({fast_time:.3f}s)")
    print(f"加速比: {speedup:.1f}x")
    print(f"相对误差: {error:.2f}%")
    print()

def test_self_intersection():
    """测试自碰撞检测优化"""
    print("=== 自碰撞检测测试 ===")
    try:
        import trimesh
        # 创建简单的立方体网格
        mesh = trimesh.creation.box()
        
        # 精确模式
        start_time = time.time()
        intersect_exact = compute_self_intersection_count(mesh, fast_mode=False)
        exact_time = time.time() - start_time
        
        # 快速模式
        start_time = time.time()
        intersect_fast = compute_self_intersection_count(mesh, fast_mode=True)
        fast_time = time.time() - start_time
        
        speedup = exact_time / fast_time if fast_time > 0 else float('inf')
        
        print(f"精确模式: {intersect_exact} 个碰撞 ({exact_time:.3f}s)")
        print(f"快速模式: {intersect_fast} 个碰撞 ({fast_time:.3f}s)")
        print(f"加速比: {speedup:.1f}x")
        print()
        
    except ImportError:
        print("Trimesh未安装，跳过自碰撞测试")
        print()

def test_overall_performance():
    """测试整体性能"""
    print("=== 整体性能测试 ===")
    vertices_a, vertices_b = generate_test_data()
    
    # 模拟完整的评估流程
    def run_evaluation(fast_mode=True):
        start_time = time.time()
        
        # Chamfer距离
        subsample_ratio = 0.05 if fast_mode else 1.0
        chamfer = compute_chamfer_distance(vertices_a, vertices_b, subsample_ratio=subsample_ratio)
        
        # ARAP误差
        sample_ratio = 0.02 if fast_mode else 0.1
        arap = compute_arap_error(vertices_a, vertices_b, sample_ratio=sample_ratio)
        
        # 法向一致性（无优化）
        normals_a = np.random.randn(*vertices_a.shape)
        normals_b = np.random.randn(*vertices_b.shape)
        normals = compute_normal_consistency(normals_a, normals_b)
        
        return time.time() - start_time
    
    # 测试多次取平均
    exact_times = []
    fast_times = []
    
    for i in range(3):
        print(f"运行测试 {i+1}/3...")
        exact_times.append(run_evaluation(fast_mode=False))
        fast_times.append(run_evaluation(fast_mode=True))
    
    exact_avg = np.mean(exact_times)
    fast_avg = np.mean(fast_times)
    speedup = exact_avg / fast_avg if fast_avg > 0 else float('inf')
    
    print(f"精确模式平均时间: {exact_avg:.3f}s")
    print(f"快速模式平均时间: {fast_avg:.3f}s")
    print(f"整体加速比: {speedup:.1f}x")
    print()

def main():
    """主测试函数"""
    print("评估系统优化效果测试")
    print("=" * 50)
    print()
    
    test_chamfer_distance()
    test_arap_error()
    test_self_intersection()
    test_overall_performance()
    
    print("测试完成！")
    print()
    print("建议:")
    print("- 如果加速比低于预期，检查CPU核数和内存")
    print("- 如果误差过大，可以调整采样比例参数")
    print("- 生产环境中建议使用快速模式以获得最佳性能")

if __name__ == "__main__":
    main()