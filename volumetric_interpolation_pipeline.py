#!/usr/bin/env python3
"""
Volumetric Video Interpolation Pipeline

完整的体素视频插值流水线，包括：
1. 骨骼预测 (SkelSequencePrediction.py)
2. 插值生成 (Interpolate.py)
3. 蒙皮权重优化 (Skinning.py)

使用流程：
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [num_interpolate]
"""

import os
import sys
import argparse
from pathlib import Path
import time
import hashlib

def check_dependencies():
    """检查依赖项"""
    print("🔍 检查依赖项...")
    
    required_modules = [
        'torch', 'numpy', 'open3d', 'scipy', 'matplotlib', 
        'trimesh', 'pygltflib', 'imageio', 'cv2'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module} - 缺失")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ 缺少以下依赖项: {missing_modules}")
        print("请安装缺失的依赖项后重试")
        return False
    
    print("✅ 所有依赖项检查通过")
    return True

def setup_paths(folder_path):
    """设置输出路径"""
    folder_path = Path(folder_path)
    
    # 创建输出目录结构
    output_base = Path("output")
    output_base.mkdir(exist_ok=True)
    
    # 使用稳定的哈希算法为每个输入文件夹创建唯一的输出目录
    folder_str = str(folder_path.absolute())
    folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]  # 使用MD5哈希的后8位
    output_dir = output_base / f"pipeline_{folder_path.name}_{folder_hash}"
    output_dir.mkdir(exist_ok=True)
    
    # 子目录
    skeleton_dir = output_dir / "skeleton_prediction"
    skinning_dir = output_dir / "skinning_weights"
    interpolation_dir = output_dir / "interpolation_results"
    
    skeleton_dir.mkdir(exist_ok=True)
    skinning_dir.mkdir(exist_ok=True)
    interpolation_dir.mkdir(exist_ok=True)
    
    print(f"📁 输出目录设置:")
    print(f"  - 输入文件夹: {folder_path}")
    print(f"  - 文件夹哈希: {folder_hash}")
    print(f"  - 输出目录: {output_dir}")
    
    return {
        'base': output_dir,
        'skeleton': skeleton_dir,
        'skinning': skinning_dir,
        'interpolation': interpolation_dir
    }

def step1_skeleton_prediction(folder_path, output_paths):
    """步骤1: 骨骼预测"""
    print("\n" + "="*60)
    print("🎯 步骤1: 骨骼预测")
    print("="*60)
    
    step_start_time = time.time()
    
    # 检查是否已经存在骨骼数据
    skeleton_data_path = output_paths['skeleton']
    keypoints_file = skeleton_data_path / "keypoints.npy"
    transforms_file = skeleton_data_path / "transforms.npy"
    parents_file = skeleton_data_path / "parents.npy"
    
    if keypoints_file.exists() and transforms_file.exists() and parents_file.exists():
        print(f"✅ 发现已存在的骨骼数据: {skeleton_data_path}")
        print("  跳过骨骼预测步骤")
        return True
    
    print(f"🔧 开始骨骼预测...")
    print(f"  输入文件夹: {folder_path}")
    print(f"  输出目录: {skeleton_data_path}")
    
    try:
        # 导入并运行骨骼预测
        from SkelSequencePrediction import main as skel_prediction_main
        
        # 保存原始参数
        original_argv = sys.argv.copy()
        
        # 设置新的参数
        sys.argv = [
            'SkelSequencePrediction.py',
            '--mesh_folder', str(folder_path),
            '--output_dir', str(skeleton_data_path),
            '--max_frames', '200'  # 限制最大帧数
        ]
        
        # 运行骨骼预测
        prediction_start = time.time()
        skel_prediction_main()
        prediction_time = time.time() - prediction_start
        
        # 恢复原始参数
        sys.argv = original_argv
        
        step_time = time.time() - step_start_time
        print(f"✅ 骨骼预测完成！")
        print(f"  - 预测耗时: {prediction_time:.2f}秒")
        print(f"  - 步骤总耗时: {step_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 骨骼预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def step2_interpolation(folder_path, start_frame, end_frame, num_interpolate, output_paths):
    """步骤2: 插值生成"""
    print("\n" + "="*60)
    print("🎯 步骤2: 插值生成")
    print("="*60)
    
    step_start_time = time.time()
    
    print(f"🔧 开始插值生成...")
    print(f"  输入文件夹: {folder_path}")
    print(f"  起始帧: {start_frame}")
    print(f"  结束帧: {end_frame}")
    print(f"  插值帧数: {num_interpolate}")
    print(f"  输出目录: {output_paths['interpolation']}")
    print(f"  权重目录: {output_paths['skinning']}")
    
    try:
        # 导入插值器
        from Interpolate import VolumetricInterpolator
        
        # 初始化插值器
        init_start = time.time()
        interpolator = VolumetricInterpolator(
            skeleton_data_dir=str(output_paths['skeleton']),
            mesh_folder_path=str(folder_path),
            weights_path=None  # 让插值器自动处理权重
        )
        
        # 设置插值器的输出目录为base目录，这样权重文件会保存在正确位置
        interpolator.output_dir = str(output_paths['base'])
        
        init_time = time.time() - init_start
        print(f"  - 插值器初始化耗时: {init_time:.2f}秒")
        print(f"  - 插值器输出目录: {interpolator.output_dir}")
        
        # 生成插值帧
        generation_start = time.time()
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=start_frame,
            frame_end=end_frame,
            num_interpolate=num_interpolate,
            max_optimize_frames=5,
            optimize_weights=True,
            output_dir=str(output_paths['interpolation'])
        )
        generation_time = time.time() - generation_start
        
        if not interpolated_frames:
            print("❌ 没有生成插值帧")
            return False
        
        step_time = time.time() - step_start_time
        print(f"✅ 插值生成完成！")
        print(f"  - 生成帧数: {len(interpolated_frames)}")
        print(f"  - 插值生成耗时: {generation_time:.2f}秒")
        print(f"  - 步骤总耗时: {step_time:.2f}秒")
        print(f"  - 输出目录: {output_paths['interpolation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 插值生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_skinning_weights_path(start_frame, end_frame, step=1):
    """生成蒙皮权重文件路径"""
    return f"skinning_weights_ref{start_frame}_opt{start_frame}-{end_frame}_step{step}.npz"

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Volumetric Video Interpolation Pipeline")
    parser.add_argument("folder_path", help="输入网格文件夹路径")
    parser.add_argument("start_frame", type=int, help="起始帧索引")
    parser.add_argument("end_frame", type=int, help="结束帧索引")
    parser.add_argument("--num_interpolate", type=int, default=10, help="插值帧数 (默认: 10)")
    parser.add_argument("--skip_skeleton", action="store_true", help="跳过骨骼预测步骤")
    parser.add_argument("--visualization", action="store_true", help="启用可视化 (默认: 关闭)")
    
    args = parser.parse_args()
    
    pipeline_start_time = time.time()
    
    print("🎬 Volumetric Video Interpolation Pipeline")
    print("="*60)
    print(f"输入文件夹: {args.folder_path}")
    print(f"起始帧: {args.start_frame}")
    print(f"结束帧: {args.end_frame}")
    print(f"插值帧数: {args.num_interpolate}")
    print(f"可视化: {'启用' if args.visualization else '禁用'}")
    
    # 检查依赖项
    if not check_dependencies():
        return False
    
    # 检查输入路径
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"❌ 输入文件夹不存在: {folder_path}")
        return False
    
    # 设置输出路径
    setup_start = time.time()
    output_paths = setup_paths(folder_path)
    setup_time = time.time() - setup_start
    print(f"📁 输出目录: {output_paths['base']}")
    print(f"⏱️  路径设置耗时: {setup_time:.2f}秒")
    
    # 步骤1: 骨骼预测
    if not args.skip_skeleton:
        if not step1_skeleton_prediction(folder_path, output_paths):
            return False
    else:
        print("⏭️  跳过骨骼预测步骤")
    
    # 步骤2: 插值生成
    if not step2_interpolation(folder_path, args.start_frame, args.end_frame, args.num_interpolate, output_paths):
        return False
    
    # 完成
    pipeline_time = time.time() - pipeline_start_time
    print("\n" + "="*60)
    print("🎉 Pipeline 完成！")
    print("="*60)
    print(f"📁 结果保存在: {output_paths['base']}")
    print(f"  - 骨骼数据: {output_paths['skeleton']}")
    print(f"  - 蒙皮权重: {output_paths['skinning']}")
    print(f"  - 插值结果: {output_paths['interpolation']}")
    print(f"⏱️  Pipeline总耗时: {pipeline_time:.2f}秒")
    
    # 显示生成的文件
    interpolation_dir = output_paths['interpolation']
    obj_files = list(interpolation_dir.glob("*.obj"))
    png_files = list(interpolation_dir.glob("*.png"))
    
    print(f"\n📊 生成的文件:")
    print(f"  - OBJ文件: {len(obj_files)} 个")
    print(f"  - PNG文件: {len(png_files)} 个")
    
    if obj_files:
        print(f"  - 示例OBJ: {obj_files[0].name}")
    if png_files:
        print(f"  - 示例PNG: {png_files[0].name}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Pipeline 执行成功！")
        sys.exit(0)
    else:
        print("\n❌ Pipeline 执行失败！")
        sys.exit(1) 