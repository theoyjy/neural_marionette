#!/usr/bin/env python3
"""
调试纹理插值问题
"""

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def debug_texture_interpolation():
    """调试纹理插值问题"""
    print("=" * 60)
    print("调试纹理插值问题")
    print("=" * 60)
    
    # 检查原始纹理文件
    texture_folder = Path("D:/Code/VVEditor/Rafa_Approves_hd_4k")
    
    # 查找纹理文件
    texture_files = sorted(list(texture_folder.glob("Frame_*_textured_hd_t_s_c.jpg")))
    print(f"📁 找到 {len(texture_files)} 个纹理文件")
    
    if len(texture_files) < 2:
        print("❌ 纹理文件不足，无法测试插值")
        return
    
    # 加载两个纹理文件进行测试
    start_texture_file = texture_files[0]
    end_texture_file = texture_files[-1]
    
    print(f"📁 起始纹理: {start_texture_file.name}")
    print(f"📁 结束纹理: {end_texture_file.name}")
    
    # 加载纹理
    start_texture = cv2.imread(str(start_texture_file))
    end_texture = cv2.imread(str(end_texture_file))
    
    if start_texture is None or end_texture is None:
        print("❌ 无法加载纹理文件")
        return
    
    # 转换颜色空间
    start_texture_rgb = cv2.cvtColor(start_texture, cv2.COLOR_BGR2RGB)
    end_texture_rgb = cv2.cvtColor(end_texture, cv2.COLOR_BGR2RGB)
    
    print(f"✅ 起始纹理尺寸: {start_texture_rgb.shape}")
    print(f"✅ 结束纹理尺寸: {end_texture_rgb.shape}")
    
    # 检查纹理内容
    print(f"📊 起始纹理统计:")
    print(f"  - 最小值: {start_texture_rgb.min()}")
    print(f"  - 最大值: {start_texture_rgb.max()}")
    print(f"  - 均值: {start_texture_rgb.mean():.2f}")
    print(f"  - 标准差: {start_texture_rgb.std():.2f}")
    
    print(f"📊 结束纹理统计:")
    print(f"  - 最小值: {end_texture_rgb.min()}")
    print(f"  - 最大值: {end_texture_rgb.max()}")
    print(f"  - 均值: {end_texture_rgb.mean():.2f}")
    print(f"  - 标准差: {end_texture_rgb.std():.2f}")
    
    # 测试不同的插值参数
    test_t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print(f"\n🧪 测试纹理插值:")
    for t in test_t_values:
        # 使用cv2.addWeighted进行插值
        interpolated = cv2.addWeighted(start_texture_rgb, 1 - t, end_texture_rgb, t, 0)
        
        print(f"  t={t:.2f}:")
        print(f"    - 尺寸: {interpolated.shape}")
        print(f"    - 最小值: {interpolated.min()}")
        print(f"    - 最大值: {interpolated.max()}")
        print(f"    - 均值: {interpolated.mean():.2f}")
        print(f"    - 标准差: {interpolated.std():.2f}")
        
        # 保存插值结果用于检查
        output_dir = Path("debug_texture_output")
        output_dir.mkdir(exist_ok=True)
        
        # 转换回BGR并保存
        interpolated_bgr = cv2.cvtColor(interpolated, cv2.COLOR_RGB2BGR)
        output_file = output_dir / f"interpolated_t_{t:.2f}.jpg"
        cv2.imwrite(str(output_file), interpolated_bgr)
        print(f"    - 保存到: {output_file}")
    
    # 保存原始纹理用于对比
    cv2.imwrite(str(output_dir / "original_start.jpg"), start_texture)
    cv2.imwrite(str(output_dir / "original_end.jpg"), end_texture)
    
    print(f"\n✅ 调试完成，结果保存在: {output_dir}")
    print("请检查生成的纹理文件，看看插值是否正确")

if __name__ == "__main__":
    debug_texture_interpolation() 