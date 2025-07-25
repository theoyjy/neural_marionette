#!/usr/bin/env python3
"""
体素视频插值系统 - 命令行界面

使用方法:
    python interpolate_cli.py --start 10 --end 50 --num 20 --output output/interpolation
    python interpolate_cli.py --start 0 --end 100 --num 50 --visualize
    python interpolate_cli.py --config config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from Interpolate import VolumetricInterpolator

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="体素视频插值系统 - 基于骨骼的网格插值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本插值
  python interpolate_cli.py --start 10 --end 50 --num 20
  
  # 可视化插值结果
  python interpolate_cli.py --start 0 --end 100 --num 50 --visualize
  
  # 导出不同格式
  python interpolate_cli.py --start 5 --end 25 --num 10 --format ply
  
  # 使用配置文件
  python interpolate_cli.py --config my_config.json
        """
    )
    
    # 基本参数
    parser.add_argument('--start', type=int, required=False,
                       help='起始帧索引')
    parser.add_argument('--end', type=int, required=False,
                       help='结束帧索引')
    parser.add_argument('--num', type=int, required=False,
                       help='插值帧数')
    
    # 路径参数
    parser.add_argument('--skeleton-dir', type=str, default='output/skeleton_prediction',
                       help='骨骼数据目录路径 (默认: output/skeleton_prediction)')
    parser.add_argument('--mesh-dir', type=str, default='D:/Code/VVEditor/Rafa_Approves_hd_4k',
                       help='网格文件目录路径')
    parser.add_argument('--weights', type=str, default='output/skinning_weights_auto.npz',
                       help='预计算蒙皮权重路径')
    parser.add_argument('--output', type=str, default='output/interpolation_results',
                       help='输出目录路径')
    
    # 功能参数
    parser.add_argument('--format', type=str, default='obj', choices=['obj', 'ply', 'stl'],
                       help='输出格式 (默认: obj)')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化插值结果')
    parser.add_argument('--no-optimize', action='store_true',
                       help='跳过权重优化（使用预计算权重）')
    parser.add_argument('--save-animation', action='store_true',
                       help='保存动画帧')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='配置文件路径 (JSON格式)')
    
    # 高级参数
    parser.add_argument('--max-iter', type=int, default=300,
                       help='权重优化最大迭代次数 (默认: 300)')
    parser.add_argument('--regularization', type=float, default=0.01,
                       help='正则化系数 (默认: 0.01)')
    
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return None

def validate_paths(args):
    """验证路径是否存在"""
    errors = []
    
    # 检查骨骼数据目录
    if not os.path.exists(args.skeleton_dir):
        errors.append(f"骨骼数据目录不存在: {args.skeleton_dir}")
    
    # 检查网格目录
    if not os.path.exists(args.mesh_dir):
        errors.append(f"网格目录不存在: {args.mesh_dir}")
    
    # 检查权重文件（如果指定）
    if args.weights and not os.path.exists(args.weights):
        print(f"⚠️  权重文件不存在: {args.weights} (将自动优化权重)")
    
    if errors:
        print("❌ 路径验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True

def run_interpolation(args):
    """运行插值"""
    print("🎬 体素视频插值系统")
    print("=" * 50)
    
    # 验证路径
    if not validate_paths(args):
        return False
    
    try:
        # 初始化插值器
        print("🔧 初始化插值器...")
        interpolator = VolumetricInterpolator(
            skeleton_data_dir=args.skeleton_dir,
            mesh_folder_path=args.mesh_dir,
            weights_path=args.weights if os.path.exists(args.weights) else None
        )
        
        print(f"📋 插值参数:")
        print(f"  - 起始帧: {args.start}")
        print(f"  - 结束帧: {args.end}")
        print(f"  - 插值帧数: {args.num}")
        print(f"  - 输出格式: {args.format}")
        print(f"  - 输出目录: {args.output}")
        print(f"  - 可视化: {args.visualize}")
        print(f"  - 权重优化: {not args.no_optimize}")
        
        # 创建输出目录
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 执行插值
        success = True
        
        # 导出插值序列
        print(f"\n📦 导出插值序列...")
        interpolator.export_interpolation_sequence(
            frame_start=args.start,
            frame_end=args.end,
            num_interpolate=args.num,
            max_optimize_frames = 20,
            output_dir=args.output,
            format=args.format
        )
        
        # 可视化（如果需要）
        if args.visualize:
            print(f"\n🎨 可视化插值结果...")
            interpolator.visualize_interpolation(
                frame_start=args.start,
                frame_end=args.end,
                num_interpolate=args.num,
                output_dir=args.output,
                save_animation=args.save_animation
            )
        
        print(f"\n🎉 插值完成！")
        print(f"📁 结果保存在: {args.output}")
        
        return True
        
    except Exception as e:
        print(f"❌ 插值过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_config():
    """创建示例配置文件"""
    config = {
        "skeleton_dir": "output/skeleton_prediction",
        "mesh_dir": "D:/Code/VVEditor/Rafa_Approves_hd_4k",
        "weights": "output/skinning_weights_auto.npz",
        "output": "output/interpolation_results",
        "format": "obj",
        "visualize": True,
        "save_animation": True,
        "max_iter": 300,
        "regularization": 0.01,
        "interpolation_examples": [
            {
                "name": "short_sequence",
                "start": 10,
                "end": 50,
                "num": 20
            },
            {
                "name": "long_sequence", 
                "start": 0,
                "end": 100,
                "num": 50
            },
            {
                "name": "smooth_interpolation",
                "start": 5,
                "end": 25,
                "num": 40
            }
        ]
    }
    
    config_path = "interpolation_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📝 示例配置文件已创建: {config_path}")
    print("你可以编辑此文件来配置插值参数")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 如果指定了配置文件，加载配置
    if args.config:
        config = load_config(args.config)
        if config is None:
            return False
        
        # 将配置参数合并到args中
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # 检查必需参数
    if args.start is None or args.end is None or args.num is None:
        print("❌ 缺少必需参数: --start, --end, --num")
        print("使用 --help 查看帮助信息")
        print("\n或者使用配置文件:")
        create_sample_config()
        return False
    
    # 验证参数
    if args.start < 0 or args.end < 0:
        print("❌ 帧索引不能为负数")
        return False
    
    if args.start >= args.end:
        print("❌ 起始帧必须小于结束帧")
        return False
    
    if args.num <= 0:
        print("❌ 插值帧数必须大于0")
        return False
    
    # 运行插值
    return run_interpolation(args)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 