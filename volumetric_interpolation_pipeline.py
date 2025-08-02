#!/usr/bin/env python3
"""
Volumetric Video Interpolation Pipeline

Complete volumetric video interpolation pipeline including:
1. Skeleton Prediction (SkelSequencePrediction.py)
2. Interpolation Generation (Interpolate.py) - Supports multiple interpolation methods
3. Skinning Weight Optimization (Skinning.py)
4. Texture Processing (texture_utils.py) - Newly added texture support

Supported interpolation methods:
- baseline: Basic interpolation method
- dual_reference: Dual reference frame interpolation
- adaptive_similarity: Adaptive similarity frame interpolation

Usage:
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [num_interpolate] [--method] [--texture]
"""

import os
import sys
import argparse
from pathlib import Path
import time
import hashlib
import json

# Import texture processing module
try:
    from texture_utils import VertexColorProcessor, integrate_vertex_color_processing
    VERTEX_COLOR_AVAILABLE = True
except ImportError:
    print("Warning: Vertex color processing module not available, will skip vertex color processing")
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

def setup_paths(folder_path, method="baseline", start_frame=0, end_frame=0, num_interpolate=10, 
                evaluation_mode=False, evaluation_output_dir=None):
    """Set Output Paths"""
    folder_path = Path(folder_path)
    
    if evaluation_mode and evaluation_output_dir:
        # 评估模式：使用指定的输出目录
        output_dir = Path(evaluation_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 子目录 - 统一的结构
        sequence_name = output_dir.parent.stem
        components = sequence_name.split('_')
        sub_seq_name = '_'.join(components[:-1])

        new_out_dir = output_dir.parent.parent / f"{sub_seq_name}"
        new_out_dir.mkdir(parents=True, exist_ok=True)

        skeleton_dir = new_out_dir / "skeleton_prediction"
        skinning_dir = new_out_dir / "skinning_weights"
        
        skeleton_dir.mkdir(exist_ok=True)
        skinning_dir.mkdir(exist_ok=True)
        
        # 插值结果目录 - 按方法区分
        interpolation_dir = output_dir / f"{method}"
        interpolation_dir.mkdir(exist_ok=True)
        
        print(f"Evaluation Mode Output Directory:")
        print(f"Input Folder: {folder_path}")
        print(f"Evaluation Output: {output_dir}")
        print(f"Interpolation Method: {method}")
        print(f"Interpolation Results: {interpolation_dir}")
        
    else:
        # 标准模式：使用哈希目录
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
        
        print(f"Standard Mode Output Directory:")
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
        print(f"SUCCESS Found existing skeleton prediction results, skipping prediction step")
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
        
        # Load mesh sequence
        print("Loading mesh sequence...")
        voxel_sequence, mesh_sequence, points_sequence = predictor.load_mesh_sequence(
            str(folder_path), file_pattern="*.obj", max_frames=160
        )
        
        # Predict skeleton
        prediction_start = time.time()
        results = predictor.predict_skeleton_sequence(voxel_sequence)
        prediction_time = time.time() - prediction_start
        
        # Save results
        print("Saving skeleton prediction results...")
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
        print(f"Skeleton prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def step2_interpolation(folder_path, start_frame, end_frame, num_interpolate, output_paths, evaluation_mode, method="baseline", 
                       save_standard_obj=False, save_npy_files=False):
    """
    Step 2: Interpolation Generation
    
    Args:
        folder_path: Input folder path
        start_frame: Start frame index (index in sorted file list, starting from 0)
        end_frame: End frame index (index in sorted file list, starting from 0)
        num_interpolate: Number of interpolated frames
        output_paths: Output paths dictionary
        method: Interpolation method
        save_standard_obj: Whether to save standard obj files (avoid duplication)
        save_npy_files: Whether to save npy files (usually not needed)
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
        # Initialize vertex color processor
        vertex_color_processor = None
        if VERTEX_COLOR_AVAILABLE:
            print(f"Initializing vertex color processor...")
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
        
        # Integrate vertex color processing (if enabled)
        print(f"Evaluation Mode: {evaluation_mode}")
        if vertex_color_processor is not None and not evaluation_mode:
            print(f"Integrating vertex color processing into interpolator...")
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
        print(f"Interpolation generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_skinning_weights_path(start_frame, end_frame, step=1):
    """Generate skinning weights file path"""
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
    parser.add_argument("--evaluation-mode", action="store_true", help="Enable evaluation mode with unified output directory")
    parser.add_argument("--evaluation-output-dir", help="Output directory for evaluation mode (required when --evaluation-mode is used)")

    args = parser.parse_args()
    
    print("="*60)
    print("Volumetric Video Interpolation Pipeline")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check vertex color processing availability
    if not VERTEX_COLOR_AVAILABLE:
        print("Error: Vertex color processing not available, please check dependencies")
        return
    
    # Validate evaluation mode parameters
    if args.evaluation_mode and not args.evaluation_output_dir:
        print("Error: Evaluation mode requires --evaluation-output-dir parameter")
        return
    
    # 设置输出路径
    output_paths = setup_paths(
        args.folder_path, 
        args.method, 
        args.start_frame, 
        args.end_frame, 
        args.num_interpolate,
        evaluation_mode=args.evaluation_mode,
        evaluation_output_dir=args.evaluation_output_dir
    )
    
    total_start_time = time.time()
    
    # Step 1: Skeleton Prediction
    if not step1_skeleton_prediction(args.folder_path, output_paths):
        print("Skeleton Prediction failed, exiting...")
        return
    
    # Step 2: Interpolation Generation
    print(f"\nStarting interpolation generation...")
    print(f"Input folder: {args.folder_path}")
    print(f"Start frame index: {args.start_frame} (index in sorted file list, starting from 0)")
    print(f"End frame index: {args.end_frame} (index in sorted file list, starting from 0)")
    print(f"Interpolation frames: {args.num_interpolate}")
    print(f"Interpolation method: {args.method}")
    
    if not step2_interpolation(
        args.folder_path, 
        args.start_frame, 
        args.end_frame, 
        args.num_interpolate, 
        output_paths, 
        args.evaluation_mode,
        args.method,
        args.evaluation_mode
    ):
        print("Interpolation generation failed, exiting...")
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