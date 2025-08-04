#!/usr/bin/env python3
"""
The main show: The Volumetric Video Interpolation Pipeline.

This thing handles the whole process from start to finish:
1.  Predicting the skeleton (SkelSequencePrediction.py)
2.  Generating the in-between frames (Interpolate.py) - we've got a few ways to do this.
3.  Optimizing the skinning weights (Skinning.py)
4.  Processing textures (texture_utils.py) - Yep, it handles textures now!

Available interpolation methods:
- baseline: The simple, classic approach.
- dual_reference: Uses two reference frames for a bit more smarts.
- adaptive_similarity: An even fancier method using adaptive similarity.

How to run it:
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [num_interpolate] [--method] [--texture]
"""

import os
import sys
import argparse
from pathlib import Path
import time
import hashlib
import json

# Import our texture processing stuff.
try:
    from texture_utils import VertexColorProcessor, integrate_vertex_color_processing
    VERTEX_COLOR_AVAILABLE = True
except ImportError:
    print("Warning: Couldn't find the vertex color processing module. We'll have to skip that part.")
    VERTEX_COLOR_AVAILABLE = False


def check_dependencies():
    """Let's see if we have all the Python packages we need."""
    print("Checking our dependencies...")
    
    required_modules = [
        'torch', 'numpy', 'open3d', 'scipy', 'matplotlib', 
        'trimesh', 'pygltflib', 'imageio', 'cv2'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  - {module}... looks good.")
        except ImportError:
            print(f"  - {module}... MISSING!")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nLooks like we're missing: {missing_modules}")
        print("You'll need to install these before we can go on.")
        return False
    
    print("All dependencies are in place. Rock on.")
    return True

def setup_paths(folder_path, method="baseline", start_frame=0, end_frame=0, num_interpolate=10, 
                evaluation_mode=False, evaluation_output_dir=None):
    """Sets up all the directories where we'll save our output."""
    folder_path = Path(folder_path)
    
    if evaluation_mode and evaluation_output_dir:
        # For evaluation mode, we're told exactly where to put things.
        output_dir = Path(evaluation_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories - keep 'em consistent.
        sequence_name = output_dir.parent.stem
        components = sequence_name.split('_')
        sub_seq_name = '_'.join(components[:-1])

        new_out_dir = output_dir.parent.parent / f"{sub_seq_name}"
        new_out_dir.mkdir(parents=True, exist_ok=True)

        skeleton_dir = new_out_dir / "skeleton_prediction"
        skinning_dir = new_out_dir / "skinning_weights"
        
        skeleton_dir.mkdir(exist_ok=True)
        skinning_dir.mkdir(exist_ok=True)
        
        # Interpolation results get their own folder based on the method used.
        interpolation_dir = output_dir / f"{method}"
        interpolation_dir.mkdir(exist_ok=True)
        
        print(f"Running in Evaluation Mode. Here's the plan:")
        print(f"  - Input Folder: {folder_path}")
        print(f"  - Evaluation Output: {output_dir}")
        print(f"  - Interpolation Method: {method}")
        print(f"  - Interpolation Results will be in: {interpolation_dir}")
        
    else:
        # In standard mode, we create a unique hashed directory to keep things clean.
        output_base = Path("output")
        output_base.mkdir(exist_ok=True)
        
        # Use a hash of the folder path to make a unique directory name.
        folder_str = str(folder_path.absolute())
        # Just grab the last 8 chars of the MD5 hash. Keeps it short.
        folder_hash = hashlib.md5(folder_str.encode('utf-8')).hexdigest()[-8:]  
        output_dir = output_base / f"pipeline_{folder_path.name}_{folder_hash}"
        output_dir.mkdir(exist_ok=True)
        
        # Subdirectories - a nice, consistent structure.
        skeleton_dir = output_dir / "skeleton_prediction"
        skinning_dir = output_dir / "skinning_weights"
        
        skeleton_dir.mkdir(exist_ok=True)
        skinning_dir.mkdir(exist_ok=True)
        
        # Interpolation results go into method-specific folders.
        interpolation_root_dir = output_dir / f"interpolation_{method}"
        interpolation_dir = interpolation_root_dir / f"{start_frame}_{end_frame}_{num_interpolate}"
        interpolation_root_dir.mkdir(exist_ok=True)
        interpolation_dir.mkdir(exist_ok=True)
        
        print(f"Running in Standard Mode. Here's the plan:")
        print(f"  - Input Folder: {folder_path}")
        print(f"  - Folder Hash: {folder_hash}")
        print(f"  - Interpolation Method: {method}")
        print(f"  - Main Output Directory: {output_dir}")
        print(f"  - Interpolation Results will be in: {interpolation_dir}")
    
    return {
        'base': output_dir,
        'skeleton': skeleton_dir,
        'skinning': skinning_dir,
        'interpolation': interpolation_dir
    }

def step1_skeleton_prediction(folder_path, output_paths):
    """Step 1: Time to figure out the skeleton from the mesh sequence."""
    print("\n" + "="*60)
    print("Step 1: Skeleton Prediction")
    print("="*60)
    
    step_start_time = time.time()
    
    print(f"Starting skeleton prediction...")
    print(f"  - Input Folder: {folder_path}")
    print(f"  - Output will be saved to: {output_paths['skeleton']}")
    
    # Let's see if we've already done this work.
    skeleton_dir = output_paths['skeleton']
    keypoints_file = os.path.join(skeleton_dir, 'keypoints.npy')
    transforms_file = os.path.join(skeleton_dir, 'transforms.npy')
    parents_file = os.path.join(skeleton_dir, 'parents.npy')
    
    if os.path.exists(keypoints_file) and os.path.exists(transforms_file) and os.path.exists(parents_file):
        print(f"Awesome! Found existing skeleton results, so we can skip this step.")
        print(f"  - Keypoints: {keypoints_file}")
        print(f"  - Transforms: {transforms_file}")
        print(f"  - Parents: {parents_file}")
        return True
    
    try:
        from SkelSequencePrediction import SequenceSkeletonPredictor
        
        # Point to the pre-trained model files.
        exp_dir = 'pretrained/aist'
        checkpoint_path = os.path.join(exp_dir, 'aist_pretrained.pth')
        opt_path = os.path.join(exp_dir, 'opt.pickle')
        
        predictor = SequenceSkeletonPredictor(
            checkpoint_path=checkpoint_path,
            opt_path=opt_path
        )
        
        # Load up the mesh sequence.
        print("Loading mesh sequence...")
        voxel_sequence, mesh_sequence, points_sequence = predictor.load_mesh_sequence(
            str(folder_path), file_pattern="*.obj", max_frames=None
        )
        
        # And now, predict the skeleton.
        prediction_start = time.time()
        results = predictor.predict_skeleton_sequence(voxel_sequence)
        prediction_time = time.time() - prediction_start
        
        # Save our hard-earned results.
        print("Saving skeleton prediction results...")
        predictor.save_skeleton_results(results, str(output_paths['skeleton']), points_sequence)
        
        success = results is not None
        
        if success:
            step_time = time.time() - step_start_time
            print(f"Skeleton Prediction Completed!")
            print(f"  - Prediction took: {prediction_time:.2f} seconds")
            print(f"  - Total for this step: {step_time:.2f} seconds")
            print(f"  - Results are in: {output_paths['skeleton']}")
            return True
        else:
            print("Skeleton Prediction Failed. Bummer.")
            return False
            
    except Exception as e:
        print(f"Ouch, skeleton prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def step2_interpolation(folder_path, start_frame, end_frame, num_interpolate, output_paths, evaluation_mode, method="baseline", 
                       save_standard_obj=False, save_npy_files=False):
    """
    Step 2: Let's create those in-between frames!
    
    Args:
        folder_path: Where the input meshes are.
        start_frame: The index of the first frame.
        end_frame: The index of the last frame.
        num_interpolate: How many frames to create in between.
        output_paths: The dictionary of paths we set up earlier.
        method: Which interpolation method to use.
        save_standard_obj: Should we save standard obj files? (can create duplicates).
        save_npy_files: Should we save the npy files? (usually not necessary).
    """
    print("\n" + "="*60)
    print(f"Step 2: Generating Interpolated Frames ({method})")
    print("Vertex Color Processing: Enabled")
    print("="*60)
    
    step_start_time = time.time()
    
    print(f"Starting interpolation generation...")
    print(f"  - Input Folder: {folder_path}")
    print(f"  - Start Frame: {start_frame}")
    print(f"  - End Frame: {end_frame}")
    print(f"  - Frames to create: {num_interpolate}")
    print(f"  - Method: {method}")
    print(f"  - Output will be saved to: {output_paths['interpolation']}")
    print(f"  - Weights will be read from: {output_paths['skinning']}")
    
    try:
        # Get the vertex color processor ready.
        vertex_color_processor = None
        if VERTEX_COLOR_AVAILABLE:
            print(f"Initializing vertex color processor...")
            vertex_color_processor = VertexColorProcessor(str(folder_path))
        
        # Pick the right interpolator for the job.
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
        else:
            raise ValueError(f"Don't know this interpolation method: {method}")
        
        # Tell the interpolator where to save everything.
        interpolator.output_dir = str(output_paths['base'])
        
        print(f"  - Using Interpolator: {type(interpolator).__name__}")
        print(f"  - Interpolator's base output directory: {interpolator.output_dir}")
        
        # If vertex coloring is on, let's hook it into the interpolator.
        print(f"Evaluation Mode: {evaluation_mode}")
        if vertex_color_processor is not None and not evaluation_mode:
            print(f"Integrating vertex color processing into the interpolator...")
            integrate_vertex_color_processing(interpolator, str(folder_path), method=method)
        
        # Let's make some new frames!
        generation_start = time.time()

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
            print("Hmm, no interpolated frames were generated.")
            return False
        
        step_time = time.time() - step_start_time
        print(f"Interpolation Generation Completed!")
        print(f"  - Generated {len(interpolated_frames)} new frames.")
        print(f"  - Generation took: {generation_time:.2f} seconds")
        print(f"  - Total for this step: {step_time:.2f} seconds")
        print(f"  - Results are in: {output_paths['interpolation']}")
        
        # Let's see how the vertex color processing went.
        if vertex_color_processor is not None:
            vertex_color_success_count = sum(1 for frame in interpolated_frames if frame.get('success', False))
            print(f"  - Vertex Color Processing: {vertex_color_success_count}/{len(interpolated_frames)} frames were processed successfully.")
        
        return True
        
    except Exception as e:
        print(f"Yikes, interpolation generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_skinning_weights_path(start_frame, end_frame, step=1):
    """A little helper to create a file path for the skinning weights."""
    return f"skinning_weights_ref{start_frame}_opt{start_frame}-{end_frame}_step{step}.npz"

def main():
    parser = argparse.ArgumentParser(description="A pipeline for volumetric video interpolation.")
    parser.add_argument("folder_path", help="Path to the folder with the mesh sequence.")
    parser.add_argument("start_frame", type=int, help="Index of the start frame (starts at 0).")
    parser.add_argument("end_frame", type=int, help="Index of the end frame (starts at 0).")
    parser.add_argument("--num_interpolate", type=int, default=10, help="How many frames to create in between (default: 10).")
    parser.add_argument("--method", choices=["baseline", "dual_reference"], 
                       default="baseline", help="Which interpolation method to use (default: baseline).") 
    parser.add_argument("--skip-skinning", action="store_true", help="Skip the skinning weights optimization step.")
    parser.add_argument("--result_path", help="File path to save info about the results when we're done.")
    parser.add_argument("--evaluation-mode", action="store_true", help="Run in evaluation mode with a consistent output directory.")
    parser.add_argument("--evaluation-output-dir", help="The output directory for evaluation mode (you need this if you use --evaluation-mode).")

    args = parser.parse_args()
    
    print("="*60)
    print("Kicking off the Volumetric Video Interpolation Pipeline")
    print("="*60)
    
    # First, let's make sure we have everything we need.
    if not check_dependencies():
        return
    
    # Make sure vertex color processing is available.
    if not VERTEX_COLOR_AVAILABLE:
        print("Error: Vertex color processing isn't available. Check your dependencies.")
        return

    # VERTEX_COLOR_AVAILABLE = False
    
    # If in evaluation mode, we need the output directory.
    if args.evaluation_mode and not args.evaluation_output_dir:
        print("Error: When in evaluation mode, you have to tell me where to put the output with --evaluation-output-dir.")
        return
    
    # Set up all our output paths.
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
    
    # Step 1: Predict the skeleton.
    if not step1_skeleton_prediction(args.folder_path, output_paths):
        print("Skeleton Prediction failed, so we have to stop here.")
        return
    
    # Step 2: Generate the interpolated frames.
    print(f"\nAlright, let's get to interpolating...")
    print(f"  - Input folder: {args.folder_path}")
    print(f"  - Start frame index: {args.start_frame}")
    print(f"  - End frame index: {args.end_frame}")
    print(f"  - Frames to interpolate: {args.num_interpolate}")
    print(f"  - Interpolation method: {args.method}")
    
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
        print("Interpolation generation failed, so that's all for now.")
        return
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*60)
    print("Pipeline finished successfully! High fives all around.")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Base Output Directory: {output_paths['base']}")
    print(f"Interpolated frames are in: {output_paths['interpolation']}")
    
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
            
        print(f"Saved the results info to: {args.result_path}")

if __name__ == "__main__":
    main()
