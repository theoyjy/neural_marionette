#!/usr/bin/env python3
"""
清理脚本 - 删除不需要的旧版本文件
"""

import os
import shutil

def main():
    """删除不需要的脚本文件"""
    
    # 需要删除的文件列表
    files_to_remove = [
        "fixed_pipeline_complete.py",
        "pipeline_vv_interpolation.py",
        "fast_vv_pipeline.py",
        "NMSkel_Inter_LBS.py",
        "NMSkel_DemSkin.py", 
        "NM_AnyMate_Interpolation.py",
        "NM_IatentMotion_predict.py",
        "vis_generation.py",
        "vis_interpolation.py", 
        "vis_retarget.py"
    ]
    
    # 需要删除的文件夹列表
    folders_to_remove = [
        "pipeline",
        "results", 
        "test_vv_data/vv_processing",
        "test_vv_data/fast_vv_processing",
        "test_small"
    ]
    
    removed_files = []
    removed_folders = []
    
    # 删除文件
    for file_name in files_to_remove:
        if os.path.exists(file_name):
            try:
                os.remove(file_name)
                removed_files.append(file_name)
                print(f"✓ 删除文件: {file_name}")
            except Exception as e:
                print(f"❌ 删除文件失败 {file_name}: {e}")
        else:
            print(f"⚠️ 文件不存在: {file_name}")
    
    # 删除文件夹
    for folder_name in folders_to_remove:
        if os.path.exists(folder_name):
            try:
                shutil.rmtree(folder_name)
                removed_folders.append(folder_name)
                print(f"✓ 删除文件夹: {folder_name}")
            except Exception as e:
                print(f"❌ 删除文件夹失败 {folder_name}: {e}")
        else:
            print(f"⚠️ 文件夹不存在: {folder_name}")
    
    print(f"\n🧹 清理完成!")
    print(f"删除了 {len(removed_files)} 个文件")
    print(f"删除了 {len(removed_folders)} 个文件夹")
    
    print(f"\n📁 现在主要文件结构:")
    print(f"  complete_vv_pipeline.py - ⭐ 主要的完整管道入口")
    print(f"  GenerateSkel.py - 核心骨骼生成模块")
    print(f"  train.py - 训练脚本")
    print(f"  model/ - NeuralMarionette网络模型")
    print(f"  utils/ - 工具函数")
    print(f"  data/ - 数据文件")
    print(f"  pretrained/ - 预训练模型")

if __name__ == "__main__":
    main()
