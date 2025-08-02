#!/usr/bin/env python3
"""
简单的系统状态检查脚本
"""

import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print(f"Python版本: {sys.version}")
    return True

def check_files():
    """检查必要文件"""
    print("\n检查必要文件...")
    
    files_to_check = [
        "evaluation/data/registrations_m.hdf5",
        "volumetric_interpolation_pipeline.py",
        "evaluation/generate_keyframe_pairs.py",
        "evaluation/evaluate_interpolation.py"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False
    
    return all_exist

def check_packages():
    """检查Python包"""
    print("\n检查Python包...")
    
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("h5py", "h5py"),
        ("trimesh", "trimesh"),
        ("matplotlib", "matplotlib"),
        ("scipy", "scipy")
    ]
    
    all_available = True
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name}")
            all_available = False
    
    return all_available

def main():
    print("🔍 系统状态检查")
    print("=" * 40)
    
    # 检查Python版本
    check_python_version()
    
    # 检查文件
    files_ok = check_files()
    
    # 检查包
    packages_ok = check_packages()
    
    print("\n" + "=" * 40)
    print("📊 检查结果:")
    print(f"文件状态: {'✅ 正常' if files_ok else '❌ 有问题'}")
    print(f"包状态: {'✅ 正常' if packages_ok else '❌ 有问题'}")
    
    if files_ok and packages_ok:
        print("\n🎉 系统准备就绪！")
        print("可以运行: python evaluation/run_evaluation_pipeline.py --max_pairs 1")
    else:
        print("\n⚠️ 系统需要配置")
        if not packages_ok:
            print("请安装缺失的包:")
            print("pip install numpy pandas h5py trimesh matplotlib scipy")
        if not files_ok:
            print("请检查文件路径")

if __name__ == "__main__":
    main() 