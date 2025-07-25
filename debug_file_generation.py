#!/usr/bin/env python3
"""
调试文件生成问题
"""

import numpy as np
from pathlib import Path

def debug_interpolation_parameters():
    """调试插值参数计算"""
    print("🔍 调试插值参数计算")
    print("=" * 40)
    
    # 测试不同的num_interpolate值
    test_cases = [1, 3, 5, 10]
    
    for num_interpolate in test_cases:
        t_values = np.linspace(0, 1, num_interpolate + 2)[1:-1]
        print(f"num_interpolate = {num_interpolate}")
        print(f"  t_values = {t_values}")
        print(f"  len(t_values) = {len(t_values)}")
        print(f"  t_values.shape = {t_values.shape}")
        print()

def debug_output_directory():
    """调试输出目录结构"""
    print("🔍 调试输出目录结构")
    print("=" * 40)
    
    output_base = Path("output")
    if output_base.exists():
        pipeline_dirs = list(output_base.glob("pipeline_*"))
        print(f"找到 {len(pipeline_dirs)} 个pipeline目录")
        
        for pipeline_dir in pipeline_dirs:
            print(f"\n📁 Pipeline目录: {pipeline_dir}")
            
            # 检查子目录
            skeleton_dir = pipeline_dir / "skeleton_prediction"
            skinning_dir = pipeline_dir / "skinning_weights"
            interpolation_dir = pipeline_dir / "interpolation_results"
            
            print(f"  - 骨骼数据: {'✅' if skeleton_dir.exists() else '❌'} {skeleton_dir}")
            print(f"  - 蒙皮权重: {'✅' if skinning_dir.exists() else '❌'} {skinning_dir}")
            print(f"  - 插值结果: {'✅' if interpolation_dir.exists() else '❌'} {interpolation_dir}")
            
            # 检查文件数量
            if interpolation_dir.exists():
                obj_files = list(interpolation_dir.glob("*.obj"))
                png_files = list(interpolation_dir.glob("*.png"))
                npy_files = list(interpolation_dir.glob("*.npy"))
                
                print(f"    - OBJ文件: {len(obj_files)} 个")
                print(f"    - PNG文件: {len(png_files)} 个")
                print(f"    - NPY文件: {len(npy_files)} 个")
                
                if obj_files:
                    print(f"    - OBJ文件列表:")
                    for obj_file in sorted(obj_files):
                        print(f"      * {obj_file.name}")
            
            if skinning_dir.exists():
                weight_files = list(skinning_dir.glob("*.npz"))
                print(f"    - 权重文件: {len(weight_files)} 个")
                for weight_file in weight_files:
                    print(f"      * {weight_file.name}")

def debug_recent_run():
    """调试最近的运行结果"""
    print("🔍 调试最近的运行结果")
    print("=" * 40)
    
    # 查找最新的pipeline目录
    output_base = Path("output")
    if output_base.exists():
        pipeline_dirs = list(output_base.glob("pipeline_*"))
        if pipeline_dirs:
            latest_pipeline = max(pipeline_dirs, key=lambda x: x.stat().st_mtime)
            print(f"📁 最新Pipeline目录: {latest_pipeline}")
            
            # 检查修改时间
            mtime = latest_pipeline.stat().st_mtime
            print(f"⏰ 修改时间: {mtime}")
            
            # 检查interpolation_results目录
            interpolation_dir = latest_pipeline / "interpolation_results"
            if interpolation_dir.exists():
                print(f"\n📊 插值结果目录内容:")
                
                # 列出所有文件
                all_files = list(interpolation_dir.rglob("*"))
                print(f"  总文件数: {len(all_files)}")
                
                # 按类型分类
                obj_files = [f for f in all_files if f.suffix == '.obj']
                png_files = [f for f in all_files if f.suffix == '.png']
                npy_files = [f for f in all_files if f.suffix == '.npy']
                
                print(f"  - OBJ文件: {len(obj_files)} 个")
                print(f"  - PNG文件: {len(png_files)} 个")
                print(f"  - NPY文件: {len(npy_files)} 个")
                
                if obj_files:
                    print(f"\n📋 OBJ文件详情:")
                    for obj_file in sorted(obj_files):
                        size = obj_file.stat().st_size
                        mtime = obj_file.stat().st_mtime
                        print(f"  - {obj_file.name} (大小: {size} bytes, 时间: {mtime})")
            else:
                print("❌ 插值结果目录不存在")
        else:
            print("❌ 没有找到pipeline目录")

if __name__ == "__main__":
    print("🔍 开始调试文件生成问题")
    print("=" * 60)
    
    # 调试插值参数
    debug_interpolation_parameters()
    
    # 调试输出目录
    debug_output_directory()
    
    # 调试最近的运行
    debug_recent_run()
    
    print("\n✅ 调试完成") 