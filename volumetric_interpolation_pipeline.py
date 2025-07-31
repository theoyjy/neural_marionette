#!/usr/bin/env python3
"""
Volumetric Video Interpolation Pipeline

完整的体素视频插值流水线，包括：
1. 骨骼预测 (SkelSequencePrediction.py)
2. 插值生成 (Interpolate.py) - 支持多种插值方法
3. 蒙皮权重优化 (Skinning.py)
4. 纹理处理 (texture_utils.py) - 新增纹理支持

支持的插值方法：
- baseline: 基础插值方法
- dual_reference: 双参考帧插值
- adaptive_similarity: 相似帧自适应插值

使用流程：
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [num_interpolate] [--method] [--texture]
"""

import os
import sys
import argparse
from pathlib import Path
import time
import hashlib
import json

# 导入纹理处理模块
try:
    from texture_utils import VertexColorProcessor, integrate_vertex_color_processing
    VERTEX_COLOR_AVAILABLE = True
except ImportError:
    print("警告: 顶点颜色处理模块不可用，将跳过顶点颜色处理")
    VERTEX_COLOR_AVAILABLE = False


def check_dependencies():
    print("Check Dependencies...")
    
    required_modules = [
        'torch', 'numpy', 'open3d', 'scipy', 'matplotlib', 
        'trimesh', 'pygltflib', 'imageio', 'cv2'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"{module}")
        except ImportError:
            print(f"{module} - Missing")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing Dependencies: {missing_modules}")
        print("Please install the missing dependencies and try again")
        return False
    
    print("All Dependencies Checked")
    return True

def setup_paths(folder_path, method="baseline", start_frame=0, end_frame=0, num_interpolate=10):
    """Set Output Paths"""
    folder_path = Path(folder_path)
    
    # create output directory structure
    output_base = Path("output")
    output_base.mkdir(exist_ok=True)
    
    # 使用稳定的哈希算法为每个输入文件夹创建唯一的输出目录
    folder_str = str(folder_path.absolute())
    folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]  # 使用MD5哈希的后8位
    output_dir = output_base / f"pipeline_{folder_path.name}_{folder_hash}"
    output_dir.mkdir(exist_ok=True)
    
    # 子目录 - 统一的结构
    skeleton_dir = output_dir / "skeleton_prediction"
    skinning_dir = output_dir / "skinning_weights"
    
    skeleton_dir.mkdir(exist_ok=True)
    skinning_dir.mkdir(exist_ok=True)
    
    # 插值结果目录 - 按方法区分
    interpolation_root_dir = output_dir / f"interpolation_{method}"
    interpolation_dir = interpolation_root_dir / f"{start_frame}_{end_frame}_{num_interpolate}"
    interpolation_root_dir.mkdir(exist_ok=True)
    interpolation_dir.mkdir(exist_ok=True)
    
    print(f"Output Directory:")
    print(f"Input Folder: {folder_path}")
    print(f"Folder Hash: {folder_hash}")
    print(f"Interpolation Method: {method}")
    print(f"Output Directory: {output_dir}")
    print(f"Interpolation Results: {interpolation_dir}")
    
    return {
        'base': output_dir,
        'skeleton': skeleton_dir,
        'skinning': skinning_dir,
        'interpolation': interpolation_dir
    }

def step1_skeleton_prediction(folder_path, output_paths):
    """Step 1: Skeleton Prediction"""
    print("\n" + "="*60)
    print("Step 1: Skeleton Prediction")
    print("="*60)
    
    step_start_time = time.time()
    
    print(f"Start Skeleton Prediction...")
    print(f"Input Folder: {folder_path}")
    print(f"Output Directory: {output_paths['skeleton']}")
    
    # 检查是否已经存在骨骼预测结果
    skeleton_dir = output_paths['skeleton']
    keypoints_file = os.path.join(skeleton_dir, 'keypoints.npy')
    transforms_file = os.path.join(skeleton_dir, 'transforms.npy')
    parents_file = os.path.join(skeleton_dir, 'parents.npy')
    
    if os.path.exists(keypoints_file) and os.path.exists(transforms_file) and os.path.exists(parents_file):
        print(f"✅ 发现已保存的骨骼预测结果，跳过预测步骤")
        print(f"  - Keypoints: {keypoints_file}")
        print(f"  - Transforms: {transforms_file}")
        print(f"  - Parents: {parents_file}")
        return True
    
    try:
        from SkelSequencePrediction import SequenceSkeletonPredictor
        
        # 配置预训练模型路径
        exp_dir = 'pretrained/aist'
        checkpoint_path = os.path.join(exp_dir, 'aist_pretrained.pth')
        opt_path = os.path.join(exp_dir, 'opt.pickle')
        
        predictor = SequenceSkeletonPredictor(
            checkpoint_path=checkpoint_path,
            opt_path=opt_path
        )
        
        # 加载网格序列
        print("加载网格序列...")
        voxel_sequence, mesh_sequence, points_sequence = predictor.load_mesh_sequence(
            str(folder_path), file_pattern="*.obj", max_frames=160
        )
        
        # 预测骨骼
        prediction_start = time.time()
        results = predictor.predict_skeleton_sequence(voxel_sequence)
        prediction_time = time.time() - prediction_start
        
        # 保存结果
        print("保存骨骼预测结果...")
        predictor.save_skeleton_results(results, str(output_paths['skeleton']), points_sequence)
        
        success = results is not None
        
        if success:
            step_time = time.time() - step_start_time
            print(f"Skeleton Prediction Completed!")
            print(f"  - Prediction Time: {prediction_time:.2f} seconds")
            print(f"  - Step Total Time: {step_time:.2f} seconds")
            print(f"  - Output Directory: {output_paths['skeleton']}")
            return True
        else:
            print("Skeleton Prediction Failed")
            return False
            
    except Exception as e:
        print(f"骨骼预测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def step2_interpolation(folder_path, start_frame, end_frame, num_interpolate, output_paths, method="baseline", 
                       save_standard_obj=False, save_npy_files=False):
    """
    Step 2: 插值生成
    
    Args:
        folder_path: 输入文件夹路径
        start_frame: 起始帧索引（排序后文件列表的索引，从0开始）
        end_frame: 结束帧索引（排序后文件列表的索引，从0开始）
        num_interpolate: 插值帧数
        output_paths: 输出路径字典
        method: 插值方法
        save_standard_obj: 是否保存标准obj文件（避免重复）
        save_npy_files: 是否保存npy文件（通常不需要）
    """
    print("\n" + "="*60)
    print(f"Step 2: Interpolation Generation ({method})")
    print("Vertex Color Processing: Enabled")
    print("="*60)
    
    step_start_time = time.time()
    
    print(f"Start Interpolation Generation...")
    print(f"Input Folder: {folder_path}")
    print(f"Start Frame Index: {start_frame}")
    print(f"End Frame Index: {end_frame}")
    print(f"Number of Interpolated Frames: {num_interpolate}")
    print(f"Interpolation Method: {method}")
    print(f"Output Directory: {output_paths['interpolation']}")
    print(f"Weights Directory: {output_paths['skinning']}")
    
    try:
        # 初始化顶点颜色处理器
        vertex_color_processor = None
        if VERTEX_COLOR_AVAILABLE:
            print(f"初始化顶点颜色处理器...")
            vertex_color_processor = VertexColorProcessor(str(folder_path))
        
        # select interpolator based on method
        if method == "baseline":
            from Interpolate import VolumetricInterpolator
            interpolator = VolumetricInterpolator(
                skeleton_data_dir=str(output_paths['skeleton']),
                mesh_folder_path=str(folder_path),
                weights_path=output_paths['skinning']
            )
        elif method == "dual_reference":
            from Interpolate import DualReferenceInterpolator
            interpolator = DualReferenceInterpolator(
                skeleton_data_dir=str(output_paths['skeleton']),
                mesh_folder_path=str(folder_path),
                weights_path=output_paths['skinning']
            )
        elif method == "adaptive_similarity":
            from Interpolate import AdaptiveSimilarityInterpolator
            interpolator = AdaptiveSimilarityInterpolator(
                skeleton_data_dir=str(output_paths['skeleton']),
                mesh_folder_path=str(folder_path),
                weights_path=output_paths['skinning']
            )
        elif method == "enhanced_adaptive":
            from EnhancedAdaptiveInterpolator import EnhancedAdaptiveInterpolator
            interpolator = EnhancedAdaptiveInterpolator(
                skeleton_data_dir=str(output_paths['skeleton']),
                mesh_folder_path=str(folder_path),
                weights_path=output_paths['skinning']
            )
        elif method == "neural_marionette":
            from Interpolate import NeuralMarionetteInterpolator
            interpolator = NeuralMarionetteInterpolator(
                skeleton_data_dir=str(output_paths['skeleton']),
                mesh_folder_path=str(folder_path),
                weights_path=output_paths['skinning']
            )
        else:
            raise ValueError(f"Unsupported interpolation method: {method}")
        
        # set output directory for interpolator
        interpolator.output_dir = str(output_paths['base'])
        
        print(f"  - Interpolator Type: {type(interpolator).__name__}")
        print(f"  - Interpolator Output Directory: {interpolator.output_dir}")
        
        # 集成顶点颜色处理（如果启用）
        if vertex_color_processor is not None:
            print(f"集成顶点颜色处理到插值器...")
            integrate_vertex_color_processing(interpolator, str(folder_path))
        
        # generate interpolated frames
        generation_start = time.time()
        
        # 调用插值方法 - start_frame和end_frame已经是排序后文件列表的索引
        interpolated_frames = interpolator.generate_interpolated_frames(
            frame_start=start_frame,
            frame_end=end_frame,
            num_interpolate=num_interpolate,
            max_optimize_frames=5,
            optimize_weights=True,
            output_dir=str(output_paths['interpolation']),
            save_standard_obj=save_standard_obj,
            save_npy_files=save_npy_files
        )
        
        generation_time = time.time() - generation_start
        
        if not interpolated_frames:
            print("No interpolated frames generated")
            return False
        
        step_time = time.time() - step_start_time
        print(f"Interpolation Generation Completed!")
        print(f"  - Number of Generated Frames: {len(interpolated_frames)}")
        print(f"  - Interpolation Generation Time: {generation_time:.2f} seconds")
        print(f"  - Step Total Time: {step_time:.2f} seconds")
        print(f"  - Output Directory: {output_paths['interpolation']}")
        
        # 统计纹理处理结果
        if vertex_color_processor is not None:
            vertex_color_success_count = sum(1 for frame in interpolated_frames if frame.get('success', False))
            print(f"  - Vertex Color Processing: {vertex_color_success_count}/{len(interpolated_frames)} frames processed successfully")
        
        return True
        
    except Exception as e:
        print(f"插值生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_skinning_weights_path(start_frame, end_frame, step=1):
    """生成蒙皮权重文件路径"""
    return f"skinning_weights_ref{start_frame}_opt{start_frame}-{end_frame}_step{step}.npz"

def main():
    parser = argparse.ArgumentParser(description="Volumetric Video Interpolation Pipeline")
    parser.add_argument("folder_path", help="Mesh Sequence Folder Path")
    parser.add_argument("start_frame", type=int, help="Start Frame Index (index in sorted file list, starting from 0)")
    parser.add_argument("end_frame", type=int, help="End Frame Index (index in sorted file list, starting from 0)")
    parser.add_argument("--num_interpolate", type=int, default=10, help="Number of Interpolated Frames (Default: 10)")
    parser.add_argument("--method", choices=["baseline", "dual_reference", "adaptive_similarity", "neural_marionette"], 
                       default="baseline", help="Interpolation Method (Default: baseline)")
    parser.add_argument("--skip-skinning", action="store_true", help="Skip skinning weights optimization")
    parser.add_argument("--result_path", help="Results Info Saved Once Interpolation Finished")

    args = parser.parse_args()
    
    print("="*60)
    print("Volumetric Video Interpolation Pipeline")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 检查顶点颜色处理可用性
    if not VERTEX_COLOR_AVAILABLE:
        print("错误: 顶点颜色处理不可用，请检查依赖项")
        return
    
    # 设置输出路径
    output_paths = setup_paths(
        args.folder_path, 
        args.method, 
        args.start_frame, 
        args.end_frame, 
        args.num_interpolate
    )
    
    total_start_time = time.time()
    
    # Step 1: Skeleton Prediction
    if not step1_skeleton_prediction(args.folder_path, output_paths):
        print("Skeleton Prediction failed, exiting...")
        return
    
    # Step 2: Interpolation Generation
    print(f"\n开始插值生成...")
    print(f"输入文件夹: {args.folder_path}")
    print(f"起始帧索引: {args.start_frame} (排序后文件列表的索引，从0开始)")
    print(f"结束帧索引: {args.end_frame} (排序后文件列表的索引，从0开始)")
    print(f"插值帧数: {args.num_interpolate}")
    print(f"插值方法: {args.method}")
    
    if not step2_interpolation(
        args.folder_path, 
        args.start_frame, 
        args.end_frame, 
        args.num_interpolate, 
        output_paths, 
        args.method
    ):
        print("插值生成失败，退出...")
        return
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("Pipeline Completed Successfully!")
    print("="*60)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Output Directory: {output_paths['base']}")
    print(f"Interpolation Results: {output_paths['interpolation']}")
    
    if args.result_path:
        interpolation_dir = os.path.abspath(output_paths['interpolation'])
        results = {
            "input_folder": args.folder_path,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "num_interpolate": args.num_interpolate,
            "method": args.method,
            "results_path": args.result_path,
            "status": "success",
            "interpolated_folder": str(interpolation_dir),
            "other_output_paths": {
                "base": str(output_paths['base']),
                "skeleton": str(output_paths['skeleton']),
                "skinning": str(output_paths['skinning']),
            }
        }
        with open(args.result_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        print(f"Results info saved to: {args.result_path}: {results}")

if __name__ == "__main__":
    main() 