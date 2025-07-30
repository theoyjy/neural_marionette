#!/usr/bin/env python3
"""
Volumetric Video Interpolation Pipeline

完整的体素视频插值流水线，包括：
1. 骨骼预测 (SkelSequencePrediction.py)
2. 插值生成 (Interpolate.py) - 支持多种插值方法
3. 蒙皮权重优化 (Skinning.py)

支持的插值方法：
- baseline: 基础插值方法
- dual_reference: 双参考帧插值
- adaptive_similarity: 相似帧自适应插值

使用流程：
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [num_interpolate] [--method]
"""

import os
import sys
import argparse
from pathlib import Path
import time
import hashlib
import json


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
    
    # 检查是否已经存在骨骼数据
    skeleton_data_path = output_paths['skeleton']
    keypoints_file = skeleton_data_path / "keypoints.npy"
    transforms_file = skeleton_data_path / "transforms.npy"
    parents_file = skeleton_data_path / "parents.npy"
    
    if keypoints_file.exists() and transforms_file.exists() and parents_file.exists():
        print(f"Found existing skeleton data: {skeleton_data_path}")
        print("  Skip Skeleton Prediction Step")
        return True
    
    print(f"Start Skeleton Prediction...")
    print(f"  Input Folder: {folder_path}")
    print(f"  Output Directory: {skeleton_data_path}")
    
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
        print(f"Skeleton Prediction Completed!")
        print(f"Prediction Time: {prediction_time:.2f} seconds")
        print(f"Step Total Time: {step_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Skeleton Prediction Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def step2_interpolation(folder_path, start_frame, end_frame, num_interpolate, output_paths, method="baseline"):
    """Step 2: Interpolation Generation"""
    print("\n" + "="*60)
    print(f"Step 2: Interpolation Generation ({method})")
    print("="*60)
    
    step_start_time = time.time()
    
    print(f"Start Interpolation Generation...")
    print(f"Input Folder: {folder_path}")
    print(f"Start Frame: {start_frame}")
    print(f"End Frame: {end_frame}")
    print(f"Number of Interpolated Frames: {num_interpolate}")
    print(f"Interpolation Method: {method}")
    print(f"Output Directory: {output_paths['interpolation']}")
    print(f"Weights Directory: {output_paths['skinning']}")
    
    try:
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
        
        # generate interpolated frames
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
            print("No interpolated frames generated")
            return False
        
        step_time = time.time() - step_start_time
        print(f"Interpolation Generation Completed!")
        print(f"  - Number of Generated Frames: {len(interpolated_frames)}")
        print(f"  - Interpolation Generation Time: {generation_time:.2f} seconds")
        print(f"  - Step Total Time: {step_time:.2f} seconds")
        print(f"  - Output Directory: {output_paths['interpolation']}")
        
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
    parser.add_argument("start_frame", type=int, help="Start Frame Index")
    parser.add_argument("end_frame", type=int, help="End Frame Index")
    parser.add_argument("--num_interpolate", type=int, default=10, help="Number of Interpolated Frames (Default: 10)")
    parser.add_argument("--method", choices=["baseline", "dual_reference", "adaptive_similarity", "neural_marionette"], 
                       default="baseline", help="Interpolation Method (Default: baseline)")
    parser.add_argument("--skip_skeleton", action="store_true", help="Skip Skeleton Prediction Step")
    parser.add_argument("--visualization", action="store_true", help="Enable Visualization (Default: disabled)")
    parser.add_argument("--result_path", help="Results Info Saved Once Interpolation Finished")
    
    args = parser.parse_args()
    
    pipeline_start_time = time.time()
    
    print("Volumetric Video Interpolation Pipeline")
    print("="*60)
    print(f"Input Folder: {args.folder_path}")
    print(f"Start Frame: {args.start_frame}")
    print(f"End Frame: {args.end_frame}")
    print(f"Number of Interpolated Frames: {args.num_interpolate}")
    print(f"Interpolation Method: {args.method}")
    print(f"Visualization: {'Enabled' if args.visualization else 'Disabled'}")
    
    # check dependencies
    if not check_dependencies():
        return False
    
    # check input path
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Input Folder does not exist: {folder_path}")
        return False
    
    # set output path
    setup_start = time.time()
    output_paths = setup_paths(folder_path, args.method, args.start_frame, args.end_frame, args.num_interpolate)
    setup_time = time.time() - setup_start
    print(f"Output Directory: {output_paths['base']}")
    print(f"Path Setup Time: {setup_time:.2f} seconds")
    
    # step 1: skeleton prediction
    if not args.skip_skeleton:
        if not step1_skeleton_prediction(folder_path, output_paths):
            return False
    else:
        print("Skip Skeleton Prediction Step")
    
    # step 2: interpolation generation
    if not step2_interpolation(folder_path, args.start_frame, args.end_frame, args.num_interpolate, output_paths, args.method):
        return False
    
    # done
    pipeline_time = time.time() - pipeline_start_time
    print("\n" + "="*60)
    print("Pipeline Completed!")
    print("="*60)
    print(f"Results saved in: {output_paths['base']}")
    print(f"Skeleton Data: {output_paths['skeleton']}")
    print(f"Skinning Weights: {output_paths['skinning']}")
    print(f"Interpolation Results: {output_paths['interpolation']}")
    print(f"Pipeline Total Time: {pipeline_time:.2f} seconds")
    
    # show generated files
    interpolation_dir = output_paths['interpolation']
    interpolation_dir = os.path.abspath(interpolation_dir)
    
    # save results info as json
    if args.result_path:
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

    if obj_files:
        print(f"  - Example OBJ: {obj_files[0].name}")
    if png_files:
        print(f"  - Example PNG: {png_files[0].name}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nPipeline Execution Successful!")
        sys.exit(0)
    else:
        print("\nPipeline Execution Failed!")
        sys.exit(1) 