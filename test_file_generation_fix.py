#!/usr/bin/env python3
"""
测试文件生成修复效果
"""

import subprocess
import sys
from pathlib import Path

def test_file_generation():
    """测试文件生成修复效果"""
    print("🧪 测试文件生成修复效果")
    print("=" * 50)
    
    # 测试参数
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    start_frame = 10
    end_frame = 15
    num_interpolate = 5  # 测试生成5个文件
    
    # 检查输入文件夹是否存在
    if not Path(folder_path).exists():
        print(f"❌ 输入文件夹不存在: {folder_path}")
        return False
    
    print(f"📋 测试参数:")
    print(f"  - 输入文件夹: {folder_path}")
    print(f"  - 起始帧: {start_frame}")
    print(f"  - 结束帧: {end_frame}")
    print(f"  - 插值帧数: {num_interpolate}")
    
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
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline执行成功！")
            
            # 查找最新的pipeline目录
            output_base = Path("output")
            pipeline_dirs = list(output_base.glob("pipeline_*"))
            
            if pipeline_dirs:
                latest_pipeline = max(pipeline_dirs, key=lambda x: x.stat().st_mtime)
                interpolation_dir = latest_pipeline / "interpolation_results"
                
                if interpolation_dir.exists():
                    # 检查生成的文件
                    obj_files = list(interpolation_dir.glob("*.obj"))
                    png_files = list(interpolation_dir.glob("*.png"))
                    npy_files = list(interpolation_dir.glob("*.npy"))
                    
                    print(f"\n📊 文件生成结果:")
                    print(f"  - OBJ文件: {len(obj_files)} 个 (预期: {num_interpolate})")
                    print(f"  - PNG文件: {len(png_files)} 个")
                    print(f"  - NPY文件: {len(npy_files)} 个")
                    
                    if obj_files:
                        print(f"\n📋 OBJ文件列表:")
                        for obj_file in sorted(obj_files):
                            size = obj_file.stat().st_size
                            print(f"  - {obj_file.name} (大小: {size} bytes)")
                    
                    # 验证修复效果
                    if len(obj_files) == num_interpolate:
                        print(f"\n✅ 修复成功！生成了正确数量的文件")
                        print(f"  - 预期: {num_interpolate} 个文件")
                        print(f"  - 实际: {len(obj_files)} 个文件")
                        return True
                    else:
                        print(f"\n❌ 修复失败！文件数量不正确")
                        print(f"  - 预期: {num_interpolate} 个文件")
                        print(f"  - 实际: {len(obj_files)} 个文件")
                        return False
                else:
                    print(f"❌ 插值结果目录不存在: {interpolation_dir}")
                    return False
            else:
                print(f"❌ 没有找到pipeline目录")
                return False
        else:
            print(f"\n❌ Pipeline执行失败！")
            print(f"返回码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ 执行pipeline时出错: {e}")
        return False

if __name__ == "__main__":
    success = test_file_generation()
    
    if success:
        print(f"\n🎉 文件生成修复测试成功！")
    else:
        print(f"\n💥 文件生成修复测试失败！")
        sys.exit(1) 