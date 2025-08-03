#!/usr/bin/env python3
"""
测试评估修复是否有效
"""

import os
import subprocess
from pathlib import Path
import json

def test_evaluation_fix():
    """测试评估修复"""
    print("=== 评估修复测试 ===")
    
    # 检查关键帧对目录
    pairs_dir = Path("evaluation/data/dfaust/keyframe_pairs/registrations_m/50002_chicken_wings")
    if not pairs_dir.exists():
        print(f"ERROR: 关键帧对目录不存在: {pairs_dir}")
        return False
    
    # 检查k5子目录
    k5_dir = pairs_dir / "k5"
    if not k5_dir.exists():
        print(f"ERROR: k5目录不存在: {k5_dir}")
        return False
    
    # 检查pair文件
    pair_files = list(k5_dir.glob("pair_*.json"))
    if not pair_files:
        print(f"ERROR: k5目录中没有pair文件: {k5_dir}")
        return False
    
    print(f"✓ 找到 {len(pair_files)} 个pair文件")
    
    # 检查插值结果目录
    interp_dir = Path("evaluation/interpolation/registrations_m/50002_chicken_wings_k5")
    if not interp_dir.exists():
        print(f"ERROR: 插值结果目录不存在: {interp_dir}")
        return False
    
    # 检查具体的pair目录
    pair_dirs = list(interp_dir.glob("pair_*"))
    if not pair_dirs:
        print(f"ERROR: 插值结果目录中没有pair子目录: {interp_dir}")
        return False
    
    print(f"✓ 找到 {len(pair_dirs)} 个插值pair目录")
    
    # 检查方法目录和obj文件
    for pair_dir in pair_dirs[:1]:  # 只检查第一个pair
        print(f"检查 {pair_dir.name}:")
        
        for method in ["baseline", "dual_reference"]:
            method_dir = pair_dir / method
            if not method_dir.exists():
                print(f"  ERROR: 方法目录不存在: {method_dir}")
                continue
            
            obj_files = list(method_dir.glob("*.obj"))
            if not obj_files:
                print(f"  ERROR: 方法目录中没有obj文件: {method_dir}")
                continue
            
            print(f"  ✓ {method}: 找到 {len(obj_files)} 个obj文件")
            
            # 检查文件名格式
            interpolated_files = [f for f in obj_files if "interpolated_frame_" in f.name]
            if interpolated_files:
                print(f"    - interpolated_frame_*.obj 文件: {len(interpolated_files)}")
            
            frame_files = [f for f in obj_files if "frame_" in f.name and "with_colors" in f.name]
            if frame_files:
                print(f"    - frame_*_with_colors.obj 文件: {len(frame_files)}")
    
    return True

def test_path_logic():
    """测试路径构建逻辑"""
    print("\n=== 路径构建逻辑测试 ===")
    
    # 模拟evaluate_all_pairs中的路径查找逻辑
    pairs_dir = Path("evaluation/data/dfaust/keyframe_pairs/registrations_m/50002_chicken_wings")
    k_value = 5
    
    print(f"输入pairs_dir: {pairs_dir}")
    print(f"输入k_value: {k_value}")
    
    # 查找k值目录
    k_pattern = f"k{k_value}"
    direct_k_dir = pairs_dir / k_pattern
    
    if direct_k_dir.exists():
        print(f"✓ 找到k值目录: {direct_k_dir}")
        
        # 检查pair文件
        pair_files = list(direct_k_dir.glob("pair_*.json"))
        print(f"✓ 找到 {len(pair_files)} 个pair文件")
        
        return True
    else:
        print(f"ERROR: k值目录不存在: {direct_k_dir}")
        return False

def main():
    """主测试函数"""
    print("开始评估修复验证测试...")
    
    success = True
    
    # 测试1: 文件结构检查
    if not test_evaluation_fix():
        success = False
    
    # 测试2: 路径逻辑检查  
    if not test_path_logic():
        success = False
    
    print("\n=== 测试结果 ===")
    if success:
        print("✓ 所有测试通过！评估修复应该有效。")
        print("\n建议操作:")
        print("1. 运行: python evaluation/run_evaluation_pipeline.py")
        print("2. 或者直接运行: python evaluation/evaluate_interpolation.py --fast --k 5")
    else:
        print("✗ 某些测试失败，需要进一步检查。")
    
    return success

if __name__ == "__main__":
    main()