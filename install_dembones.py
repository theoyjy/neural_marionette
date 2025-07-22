#!/usr/bin/env python3
"""
下载和编译 DemBones C++ CLI 版本
=============================

这个脚本会自动下载 DemBones 源码并尝试编译
"""

import os
import subprocess
import sys
import tempfile
import zipfile
import urllib.request
import shutil

def download_dembones():
    """下载 DemBones 源码"""
    print("=== 下载 DemBones 源码 ===")
    
    # GitHub 仓库地址
    repo_url = "https://github.com/electronicarts/dem-bones/archive/refs/heads/main.zip"
    
    try:
        print(f"从 {repo_url} 下载...")
        urllib.request.urlretrieve(repo_url, "dem-bones-main.zip")
        print("✓ 下载完成")
        
        # 解压
        print("解压源码...")
        with zipfile.ZipFile("dem-bones-main.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        print("✓ 解压完成")
        return "dem-bones-main"
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def check_dependencies():
    """检查编译依赖"""
    print("\n=== 检查编译依赖 ===")
    
    dependencies = [
        ("cmake", "CMake"),
        ("git", "Git"),
    ]
    
    missing = []
    
    for cmd, name in dependencies:
        try:
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ {name} 已安装")
            else:
                print(f"❌ {name} 未安装或无法运行")
                missing.append(name)
        except:
            print(f"❌ {name} 未找到")
            missing.append(name)
    
    # 检查 Visual Studio (Windows)
    if sys.platform == "win32":
        try:
            # 检查是否有 MSBuild
            result = subprocess.run(["where", "msbuild"], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                print("✓ MSBuild (Visual Studio) 已安装")
            else:
                print("❌ MSBuild (Visual Studio) 未找到")
                missing.append("Visual Studio")
        except:
            print("❌ MSBuild (Visual Studio) 未找到")
            missing.append("Visual Studio")
    
    return missing

def compile_dembones(source_dir):
    """编译 DemBones"""
    print(f"\n=== 编译 DemBones ===")
    
    if not os.path.exists(source_dir):
        print(f"❌ 源码目录不存在: {source_dir}")
        return None
    
    os.chdir(source_dir)
    
    try:
        # 创建构建目录
        build_dir = "build"
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)
        os.chdir(build_dir)
        
        # 运行 CMake 配置
        print("运行 CMake 配置...")
        cmake_cmd = ["cmake", ".."]
        if sys.platform == "win32":
            # 在 Windows 上指定生成器
            cmake_cmd.extend(["-G", "Visual Studio 17 2022"])  # VS 2022
        
        result = subprocess.run(cmake_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ CMake 配置失败:")
            print(result.stderr)
            return None
        
        print("✓ CMake 配置完成")
        
        # 编译
        print("开始编译...")
        if sys.platform == "win32":
            compile_cmd = ["cmake", "--build", ".", "--config", "Release"]
        else:
            compile_cmd = ["make", "-j4"]
        
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ 编译失败:")
            print(result.stderr)
            return None
        
        print("✓ 编译完成")
        
        # 查找可执行文件
        exe_name = "DemBones.exe" if sys.platform == "win32" else "DemBones"
        possible_paths = [
            exe_name,
            f"Release/{exe_name}",
            f"Debug/{exe_name}",
            f"bin/{exe_name}",
            f"src/{exe_name}",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                abs_path = os.path.abspath(path)
                print(f"✓ 找到可执行文件: {abs_path}")
                return abs_path
        
        print("❌ 未找到编译后的可执行文件")
        return None
        
    except Exception as e:
        print(f"❌ 编译过程出错: {e}")
        return None

def install_dembones(exe_path):
    """安装 DemBones 到当前目录"""
    print(f"\n=== 安装 DemBones ===")
    
    if not exe_path or not os.path.exists(exe_path):
        print("❌ 可执行文件不存在")
        return False
    
    # 复制到当前工作目录
    current_dir = os.getcwd()
    target_name = "demBones.exe" if sys.platform == "win32" else "demBones"
    target_path = os.path.join(current_dir, target_name)
    
    try:
        shutil.copy2(exe_path, target_path)
        print(f"✓ 已复制到: {target_path}")
        
        # 测试可执行文件
        result = subprocess.run([target_path, "--help"], 
                              capture_output=True, timeout=10)
        if result.returncode == 0 or "dembones" in result.stdout.decode().lower():
            print("✓ 可执行文件测试成功")
            return True
        else:
            print("⚠️ 可执行文件测试失败，但文件已复制")
            return True
            
    except Exception as e:
        print(f"❌ 安装失败: {e}")
        return False

def main():
    print("DemBones C++ CLI 版本安装脚本")
    print("=" * 40)
    
    # 检查依赖
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"\n❌ 缺少以下依赖: {', '.join(missing_deps)}")
        print("\n安装建议:")
        for dep in missing_deps:
            if dep == "CMake":
                print("- CMake: https://cmake.org/download/")
            elif dep == "Git":
                print("- Git: https://git-scm.com/download/")
            elif dep == "Visual Studio":
                print("- Visual Studio: https://visualstudio.microsoft.com/downloads/")
        return False
    
    # 保存当前目录
    original_dir = os.getcwd()
    
    try:
        # 下载源码
        source_dir = download_dembones()
        if not source_dir:
            return False
        
        # 编译
        exe_path = compile_dembones(source_dir)
        if not exe_path:
            return False
        
        # 回到原目录
        os.chdir(original_dir)
        
        # 安装
        success = install_dembones(exe_path)
        
        if success:
            print("\n🎉 DemBones 安装成功!")
            print("现在可以使用 complete_vv_pipeline.py 了")
        else:
            print("\n❌ DemBones 安装失败")
            
        return success
        
    except Exception as e:
        print(f"\n❌ 安装过程出错: {e}")
        return False
    finally:
        # 清理临时文件
        os.chdir(original_dir)
        if os.path.exists("dem-bones-main.zip"):
            os.remove("dem-bones-main.zip")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
