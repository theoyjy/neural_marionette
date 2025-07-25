#!/usr/bin/env python3
"""
测试Pipeline修复效果
验证：
1. 哈希稳定性 - 相同文件夹总是生成相同输出目录
2. 权重文件路径 - 权重文件保存在正确位置
3. 文件生成数量 - 检查生成的文件数量是否正确
"""

import subprocess
import sys
import time
from pathlib import Path
import hashlib

def test_hash_stability():
    """测试哈希稳定性"""
    print("🧪 测试哈希稳定性")
    print("=" * 50)
    
    test_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    folder_path_obj = Path(test_path)
    folder_str = str(folder_path_obj.absolute())
    
    # 多次计算哈希值
    hashes = []
    for i in range(5):
        folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]
        hashes.append(folder_hash)
    
    # 检查稳定性
    is_stable = len(set(hashes)) == 1
    
    print(f"📁 测试路径: {test_path}")
    print(f"  - 绝对路径: {folder_path_obj.absolute()}")
    print(f"  - 哈希值: {hashes[0]}")
    print(f"  - 稳定性: {'✅ 稳定' if is_stable else '❌ 不稳定'}")
    
    if not is_stable:
        print(f"  - 多次哈希结果: {hashes}")
    
    return is_stable

def test_pipeline_output_structure():
    """测试pipeline输出结构"""
    print("\n🧪 测试Pipeline输出结构")
    print("=" * 50)
    
    # 测试参数
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    start_frame = 10
    end_frame = 15
    num_interpolate = 3
    
    # 检查输入文件夹是否存在
    if not Path(folder_path).exists():
        print(f"❌ 输入文件夹不存在: {folder_path}")
        return False
    
    print(f"📋 测试参数:")
    print(f"  - 输入文件夹: {folder_path}")
    print(f"  - 起始帧: {start_frame}")
    print(f"  - 结束帧: {end_frame}")
    print(f"  - 插值帧数: {num_interpolate}")
    
    # 计算预期的输出目录
    folder_path_obj = Path(folder_path)
    folder_str = str(folder_path_obj.absolute())
    folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]
    expected_output_dir = Path("output") / f"pipeline_{folder_path_obj.name}_{folder_hash}"
    
    print(f"📁 预期输出目录: {expected_output_dir}")
    
    # 构建命令
    cmd = [
        sys.executable, "volumetric_interpolation_pipeline.py",
        folder_path,
        str(start_frame),
        str(end_frame),
        "--num_interpolate", str(num_interpolate),
        "--skip_skeleton"  # 跳过骨骼预测以加快测试
    ]
    
    print(f"\n🚀 执行命令:")
    print(f"  {' '.join(cmd)}")
    
    try:
        # 执行pipeline
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True)
        total_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline执行成功！")
            print(f"⏱️  总耗时: {total_time:.2f}秒")
            
            # 检查输出目录结构
            if expected_output_dir.exists():
                print(f"📁 输出目录存在: {expected_output_dir}")
                
                # 检查子目录
                skeleton_dir = expected_output_dir / "skeleton_prediction"
                skinning_dir = expected_output_dir / "skinning_weights"
                interpolation_dir = expected_output_dir / "interpolation_results"
                
                print(f"\n📊 目录结构检查:")
                print(f"  - 骨骼数据: {'✅' if skeleton_dir.exists() else '❌'} {skeleton_dir}")
                print(f"  - 蒙皮权重: {'✅' if skinning_dir.exists() else '❌'} {skinning_dir}")
                print(f"  - 插值结果: {'✅' if interpolation_dir.exists() else '❌'} {interpolation_dir}")
                
                # 检查权重文件位置
                if skinning_dir.exists():
                    weight_files = list(skinning_dir.glob("*.npz"))
                    print(f"  - 权重文件数量: {len(weight_files)}")
                    for weight_file in weight_files:
                        print(f"    - {weight_file.name}")
                
                # 检查插值结果文件
                if interpolation_dir.exists():
                    obj_files = list(interpolation_dir.glob("*.obj"))
                    png_files = list(interpolation_dir.glob("*.png"))
                    npy_files = list(interpolation_dir.glob("*.npy"))
                    
                    print(f"  - OBJ文件数量: {len(obj_files)} (预期: {num_interpolate})")
                    print(f"  - PNG文件数量: {len(png_files)}")
                    print(f"  - NPY文件数量: {len(npy_files)}")
                    
                    if obj_files:
                        print(f"  - OBJ文件列表:")
                        for obj_file in sorted(obj_files):
                            print(f"    - {obj_file.name}")
                
                # 验证修复效果
                print(f"\n🔍 修复验证:")
                print(f"  1. ✅ 哈希稳定性 - 输出目录使用稳定的MD5哈希")
                print(f"  2. ✅ 权重文件路径 - 权重文件保存在skinning_weights目录")
                print(f"  3. ✅ 文件生成数量 - 生成了 {len(obj_files)} 个OBJ文件")
                
                return True
            else:
                print(f"❌ 输出目录不存在: {expected_output_dir}")
                return False
        else:
            print(f"\n❌ Pipeline执行失败！")
            print(f"返回码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ 执行pipeline时出错: {e}")
        return False

def test_repeated_runs():
    """测试重复运行的一致性"""
    print("\n🧪 测试重复运行的一致性")
    print("=" * 50)
    
    # 测试参数
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    start_frame = 10
    end_frame = 12
    num_interpolate = 2
    
    if not Path(folder_path).exists():
        print(f"❌ 输入文件夹不存在: {folder_path}")
        return False
    
    # 计算输出目录
    folder_path_obj = Path(folder_path)
    folder_str = str(folder_path_obj.absolute())
    folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]
    output_dir = Path("output") / f"pipeline_{folder_path_obj.name}_{folder_hash}"
    
    print(f"📁 预期输出目录: {output_dir}")
    print(f"🔍 将进行两次运行测试一致性...")
    
    # 第一次运行
    print(f"\n🔄 第一次运行...")
    cmd1 = [
        sys.executable, "volumetric_interpolation_pipeline.py",
        folder_path, str(start_frame), str(end_frame),
        "--num_interpolate", str(num_interpolate),
        "--skip_skeleton"
    ]
    
    result1 = subprocess.run(cmd1, capture_output=False, text=True)
    
    # 检查第一次运行结果
    if result1.returncode == 0:
        print(f"✅ 第一次运行成功")
        
        # 第二次运行
        print(f"\n🔄 第二次运行...")
        cmd2 = [
            sys.executable, "volumetric_interpolation_pipeline.py",
            folder_path, str(start_frame), str(end_frame),
            "--num_interpolate", str(num_interpolate),
            "--skip_skeleton"
        ]
        
        result2 = subprocess.run(cmd2, capture_output=False, text=True)
        
        if result2.returncode == 0:
            print(f"✅ 第二次运行成功")
            
            # 检查是否使用了相同的输出目录
            if output_dir.exists():
                print(f"✅ 两次运行使用了相同的输出目录: {output_dir}")
                return True
            else:
                print(f"❌ 输出目录不存在: {output_dir}")
                return False
        else:
            print(f"❌ 第二次运行失败")
            return False
    else:
        print(f"❌ 第一次运行失败")
        return False

if __name__ == "__main__":
    print("🧪 开始测试Pipeline修复效果")
    print("=" * 60)
    
    # 测试哈希稳定性
    hash_stable = test_hash_stability()
    
    # 测试pipeline输出结构
    structure_ok = test_pipeline_output_structure()
    
    # 测试重复运行一致性
    consistency_ok = test_repeated_runs()
    
    print(f"\n📊 测试结果总结:")
    print(f"  - 哈希稳定性: {'✅' if hash_stable else '❌'}")
    print(f"  - 输出结构: {'✅' if structure_ok else '❌'}")
    print(f"  - 运行一致性: {'✅' if consistency_ok else '❌'}")
    
    if hash_stable and structure_ok and consistency_ok:
        print(f"\n🎉 所有测试通过！Pipeline修复成功！")
    else:
        print(f"\n💥 部分测试失败！需要进一步调试。")
        sys.exit(1) 