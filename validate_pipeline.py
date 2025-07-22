#!/usr/bin/env python3
"""
简单的管道验证脚本
"""

import os
import subprocess
import sys

def main():
    print("🧪 Neural Marionette 体积视频插值管道验证")
    print("=" * 60)
    
    # 检查主要文件
    print("📋 检查核心文件...")
    files_to_check = [
        "complete_vv_pipeline.py",
        "GenerateSkel.py",
        "model/neural_marionette.py"
    ]
    
    all_present = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_present = False
    
    if not all_present:
        print("\n❌ 缺少核心文件！")
        return False
    
    # 测试命令行接口
    print(f"\n🔍 测试管道命令行接口...")
    try:
        result = subprocess.run([
            sys.executable, "complete_vv_pipeline.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("  ✓ 命令行接口正常")
        else:
            print(f"  ❌ 命令行接口错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ 测试时出错: {e}")
        return False
    
    # 检查测试数据
    test_dir = "test_vv_data"
    if os.path.exists(test_dir):
        obj_files = [f for f in os.listdir(test_dir) if f.endswith('.obj')]
        print(f"\n📁 发现测试数据: {len(obj_files)} 个OBJ文件")
        
        if len(obj_files) >= 5:
            print("  ✓ 测试数据充足，可以运行完整测试")
            print(f"\n📝 完整测试命令示例:")
            print(f"  python complete_vv_pipeline.py \"{test_dir}\" --start_frame 0 --end_frame 9")
        else:
            print("  ⚠️ 测试数据不足，建议先生成更多测试文件")
    else:
        print(f"\n📁 未发现测试数据目录 {test_dir}")
        print("  提示: 可以使用管道自带的测试数据生成功能")
    
    print(f"\n✨ 管道使用示例:")
    print(f"  # 基本处理")
    print(f"  python complete_vv_pipeline.py \"your_obj_folder\"")
    print(f"")
    print(f"  # 生成插值（从第5帧到第15帧，生成10个插值帧）")
    print(f"  python complete_vv_pipeline.py \"your_obj_folder\" \\")
    print(f"    --interp_from 5 --interp_to 15 --num_interp 10")
    
    print(f"\n🎉 验证完成！管道已准备就绪")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
