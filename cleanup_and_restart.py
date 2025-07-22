#!/usr/bin/env python3
"""
Neural Marionette Pipeline 清理和重构脚本
清理所有中间数据，重新设计正确的骨骼驱动逻辑
"""

import os
import shutil
import glob

def clean_generated_data():
    """清理所有生成的中间数据"""
    print("🧹 开始清理生成的数据...")
    
    # 定义要清理的文件和目录
    cleanup_targets = [
        # 生成的Python文件
        "advanced_*.py",
        "corrected_*.py", 
        "final_*.py",
        "simplified_*.py",
        "enhanced_*.py",
        "hybrid_*.py",
        "skeleton_driven_*.py",
        "true_topology_*.py",
        "unified_topology_*.py",
        "retrain_dembones_*.py",
        
        # 测试和检查文件
        "check_*.py",
        "test_*.py",
        "force_*.py",
        "inspect_*.py",
        "process_*.py",
        "verify_*.py",
        "direct_*.py",
        "quick_*.py",
        "example_*.py",
        
        # 插值相关文件
        "interpolate.py",
        "mesh_interpolation.py",
        
        # 报告文件 (保留最终报告)
        "INTERPOLATION_*.md",
        "SKELETON_DRIVEN_*.md", 
        "FINAL_TOPOLOGY_*.md",
        
        # 日志文件
        "*.txt",
        
        # 缓存文件
        "__pycache__",
    ]
    
    # 执行清理
    for pattern in cleanup_targets:
        if pattern == "__pycache__":
            # 特殊处理缓存目录
            if os.path.exists(pattern):
                shutil.rmtree(pattern)
                print(f"  ✅ 删除目录: {pattern}")
        else:
            # 处理文件模式
            files = glob.glob(pattern)
            for file in files:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"  ✅ 删除文件: {file}")
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    print(f"  ✅ 删除目录: {file}")
    
    print("🎯 清理完成！")

def clean_external_generated_data():
    """清理外部生成的数据"""
    print("\n🗂️ 清理外部生成数据...")
    
    external_paths = [
        r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
        # 可以添加其他路径
    ]
    
    for path in external_paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"  ✅ 删除目录: {path}")
            except Exception as e:
                print(f"  ⚠️ 无法删除 {path}: {e}")
        else:
            print(f"  ℹ️ 路径不存在: {path}")

def show_remaining_files():
    """显示清理后剩余的重要文件"""
    print("\n📋 剩余的核心文件:")
    
    important_files = [
        "GenerateSkel.py",
        "NMSkel_DemSkin.py", 
        "NMSkel_Inter_LBS.py",
        "NM_AnyMate_Interpolation.py",
        "NM_IatentMotion_predict.py",
        "train.py",
        "vis_*.py",
        "requirements.txt",
        "README.md",
        "FINAL_PROJECT_REPORT.md"
    ]
    
    for pattern in important_files:
        files = glob.glob(pattern)
        for file in files:
            if os.path.isfile(file):
                size = os.path.getsize(file)
                print(f"  📄 {file} ({size} bytes)")

def create_new_pipeline_structure():
    """创建新的pipeline结构"""
    print("\n🏗️ 创建新的pipeline结构...")
    
    # 创建新的目录结构
    new_dirs = [
        "pipeline",
        "pipeline/skeleton",
        "pipeline/interpolation", 
        "pipeline/utils",
        "results",
        "results/skeletons",
        "results/interpolations",
        "results/validation"
    ]
    
    for dir_path in new_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  📁 创建目录: {dir_path}")
        
    # 创建pipeline的__init__.py文件
    with open("pipeline/__init__.py", "w") as f:
        f.write('"""Neural Marionette Pipeline Package"""\n')
        
    print("  ✅ Pipeline结构创建完成")

def main():
    """主函数"""
    print("🚀 Neural Marionette Pipeline 重构")
    print("=" * 50)
    
    # 询问用户确认
    confirm = input("⚠️ 这将删除所有生成的中间数据，确认继续? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("❌ 操作已取消")
        return
        
    # 执行清理
    clean_generated_data()
    clean_external_generated_data()
    
    # 创建新结构
    create_new_pipeline_structure()
    
    # 显示剩余文件
    show_remaining_files()
    
    print("\n🎯 重构准备完成！")
    print("\n下一步:")
    print("1. 重新设计骨骼驱动逻辑")
    print("2. 实现正确的rest pose处理") 
    print("3. 建立真正的骨骼变换系统")
    print("4. 验证插值质量")

if __name__ == "__main__":
    main()
