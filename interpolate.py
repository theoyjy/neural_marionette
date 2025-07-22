#!/usr/bin/env python3
"""
Simple command-line interface for mesh interpolation using Neural Marionette data.

Example usage:
    python interpolate.py --help
    python interpolate.py --list
    python interpolate.py 1 50 --steps 20 --method direct
    python interpolate.py Frame_00001_textured_hd_t_s_c Frame_00050_textured_hd_t_s_c --visualize
"""

import os
import sys
import argparse
from mesh_interpolation import load_mesh_data, interpolate_meshes, visualize_interpolation

def main():
    parser = argparse.ArgumentParser(
        description="Interpolate between two meshes using Neural Marionette skeleton data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List available frames:
    python interpolate.py --list

  Interpolate between frame 1 and frame 10 (using frame numbers):
    python interpolate.py 1 10 --steps 20

  Interpolate between specific frame names:
    python interpolate.py Frame_00001_textured_hd_t_s_c Frame_00050_textured_hd_t_s_c

  Use skeletal interpolation with visualization:
    python interpolate.py 1 50 --method skeleton --visualize

  Save to custom directory:
    python interpolate.py 1 100 --output ./my_interpolation --steps 30
        """
    )
    
    parser.add_argument('frame_a', nargs='?', help='First frame (number 1-157 or full name)')
    parser.add_argument('frame_b', nargs='?', help='Second frame (number 1-157 or full name)')
    
    parser.add_argument('--data_folder', default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons",
                       help='Path to generated_skeletons folder')
    parser.add_argument('--steps', type=int, default=10, help='Number of interpolation steps (default: 10)')
    parser.add_argument('--method', choices=['direct', 'skeleton'], default='direct', 
                       help='Interpolation method: direct (vertex) or skeleton (bone-based)')
    parser.add_argument('--output', help='Output directory for interpolated meshes')
    parser.add_argument('--visualize', action='store_true', help='Show animation after generation')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between frames in visualization (seconds)')
    parser.add_argument('--list', action='store_true', help='List all available frames and exit')
    
    args = parser.parse_args()
    
    # Check if data folder exists
    if not os.path.exists(args.data_folder):
        print(f"ERROR: Data folder not found: {args.data_folder}")
        print("Please check the path to your generated_skeletons folder.")
        sys.exit(1)
    
    # Load data to get available frames
    try:
        print("Loading mesh data...")
        dembone_results, mesh_data = load_mesh_data(args.data_folder)
        frame_names = sorted(list(mesh_data.keys()))
        print(f"SUCCESS: Loaded {len(frame_names)} frames successfully")
    except Exception as e:
        print(f"ERROR: Error loading data: {e}")
        sys.exit(1)
    
    # List frames if requested
    if args.list:
        print(f"\nAvailable frames ({len(frame_names)} total):")
        for i, name in enumerate(frame_names, 1):
            print(f"  {i:3d}: {name}")
        return
    
    # Validate arguments
    if not args.frame_a or not args.frame_b:
        print("ERROR: Please specify both frame_a and frame_b")
        print("Use --list to see available frames")
        print("Use --help for usage examples")
        sys.exit(1)
    
    # Parse frame names
    def parse_frame_name(frame_input):
        """Convert frame input to actual frame name."""
        if frame_input.isdigit():
            # Frame number (1-based)
            frame_num = int(frame_input)
            if 1 <= frame_num <= len(frame_names):
                return frame_names[frame_num - 1]
            else:
                print(f"ERROR: Frame number {frame_num} out of range (1-{len(frame_names)})")
                sys.exit(1)
        else:
            # Frame name
            if frame_input in frame_names:
                return frame_input
            else:
                print(f"ERROR: Frame '{frame_input}' not found")
                print("Use --list to see available frames")
                sys.exit(1)
    
    frame_a = parse_frame_name(args.frame_a)
    frame_b = parse_frame_name(args.frame_b)
    
    print(f"\nTARGET: Interpolating between:")
    print(f"   Frame A: {frame_a}")
    print(f"   Frame B: {frame_b}")
    print(f"   Method: {args.method}")
    print(f"   Steps: {args.steps}")
    
    # Generate interpolation
    try:
        print(f"\nPROCESSING: Generating interpolation...")
        meshes, output_dir = interpolate_meshes(
            args.data_folder,
            frame_a,
            frame_b,
            num_steps=args.steps,
            method=args.method,
            output_dir=args.output
        )
        
        print(f"\nSUCCESS: Interpolation complete!")
        print(f"OUTPUT: Generated {len(meshes)} meshes in: {output_dir}")
        
        # Show visualization if requested
        if args.visualize:
            print(f"\nVISUALIZATION: Starting visualization (delay: {args.delay}s per frame)")
            print("Press 'q' to quit the visualization window")
            visualize_interpolation(meshes, delay=args.delay)
        
        print(f"\nDONE: You can find the interpolated meshes in:")
        print(f"   {output_dir}")
        
    except Exception as e:
        print(f"ERROR: Error during interpolation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
