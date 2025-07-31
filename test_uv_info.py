#!/usr/bin/env python3
"""
测试mesh的UV信息

检查mesh是否包含UV坐标，以及如何正确使用它们
"""

import os
import sys
from pathlib import Path
import numpy as np
import open3d as o3d

def test_mesh_uv_info():
    """测试mesh的UV信息"""
    print("🧪 测试mesh的UV信息")
    
    # 测试mesh文件
    mesh_file = "D:/Code/VVEditor/Rafa_Approves_hd_4k/Frame_00001_textured_hd_t_s_c.obj"
    
    if not os.path.exists(mesh_file):
        print(f"❌ mesh文件不存在: {mesh_file}")
        return False
    
    print(f"📁 加载mesh: {mesh_file}")
    
    # 加载mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    print(f"✅ mesh加载成功")
    print(f"  - 顶点数: {len(mesh.vertices)}")
    print(f"  - 面数: {len(mesh.triangles)}")
    print(f"  - 顶点颜色: {len(mesh.vertex_colors)}")
    
    # 检查UV信息
    if hasattr(mesh, 'triangle_uvs') and len(mesh.triangle_uvs) > 0:
        print(f"✅ 找到UV信息: {len(mesh.triangle_uvs)} 个UV坐标")
        print(f"  - UV坐标范围: {np.min(mesh.triangle_uvs, axis=0)} 到 {np.max(mesh.triangle_uvs, axis=0)}")
        
        # 显示前几个UV坐标
        print(f"  - 前5个UV坐标:")
        for i in range(min(5, len(mesh.triangle_uvs))):
            print(f"    {i}: {mesh.triangle_uvs[i]}")
    else:
        print(f"❌ 没有找到UV信息")
    
    # 检查其他可能的UV属性
    uv_attrs = []
    for attr in dir(mesh):
        if 'uv' in attr.lower() or 'tex' in attr.lower():
            uv_attrs.append(attr)
    
    if uv_attrs:
        print(f"✅ 找到其他UV相关属性: {uv_attrs}")
        for attr in uv_attrs:
            value = getattr(mesh, attr)
            if hasattr(value, '__len__'):
                print(f"  - {attr}: {len(value)} 个元素")
            else:
                print(f"  - {attr}: {value}")
    else:
        print(f"❌ 没有找到其他UV相关属性")
    
    return True

def main():
    """主测试函数"""
    print("="*60)
    print("Mesh UV信息测试")
    print("="*60)
    
    if test_mesh_uv_info():
        print("\n✅ 测试完成")
    else:
        print("\n❌ 测试失败")
    
    print("="*60)

if __name__ == "__main__":
    main() 