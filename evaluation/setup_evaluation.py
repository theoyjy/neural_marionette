#!/usr/bin/env python3
"""
评估系统安装和设置脚本
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    try:
        # 使用pip安装requirements.txt中的包
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "evaluation/requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 依赖包安装成功")
            return True
        else:
            print(f"❌ 依赖包安装失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 安装过程出错: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    
    directories = [
        "evaluation/data/keyframe_pairs",
        "evaluation/results/baseline",
        "evaluation/results/dual_reference",
        "evaluation/results/visualizations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {directory}")

def test_installation():
    """测试安装是否成功"""
    print("\n🧪 测试安装...")
    
    try:
        import numpy
        import pandas
        import h5py
        import trimesh
        import matplotlib
        import scipy
        print("✅ 所有依赖包导入成功")
        return True
    except ImportError as e:
        print(f"❌ 依赖包导入失败: {e}")
        return False

def main():
    print("🚀 评估系统安装向导")
    print("=" * 50)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 安装依赖包
    if install_requirements():
        # 创建目录
        create_directories()
        
        # 测试安装
        if test_installation():
            print("\n🎉 安装完成！")
            print("\n📝 使用说明:")
            print("1. 快速测试: python evaluation/run_evaluation_pipeline.py --max_pairs 1 --no_gt")
            print("2. 完整评估: python evaluation/run_evaluation_pipeline.py")
            print("3. 查看结果: evaluation/results/")
            print("\n📚 更多信息请查看: evaluation/README.md")
        else:
            print("\n⚠️ 安装测试失败，请手动检查")
    else:
        print("\n❌ 安装失败，请手动安装依赖包:")
        print("pip install numpy pandas trimesh h5py matplotlib scipy scikit-learn seaborn")

if __name__ == "__main__":
    main() 