#!/usr/bin/env python3
"""
Example usage of GenerateSkel.py

This script demonstrates how to use the rewritten GenerateSkel.py to process
a folder of .obj files and generate skeletons using NeuralMarionette and DemBones.
"""

import os
from GenerateSkel import process_obj_files

def main():
    # Example 1: Process all .obj files in a specific folder
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"  # Change this to your folder path
    
    if os.path.exists(folder_path):
        print(f"Processing .obj files in: {folder_path}")
        
        # Process with default settings (with visualization)
        process_obj_files(
            folder_path=folder_path,
            output_dir=None,  # Will create 'generated_skeletons' subfolder
            is_bind=False,
            visualize=True
        )
        
        print("Processing complete!")
        print(f"Check the 'generated_skeletons' folder in {folder_path} for results")
        
    else:
        print(f"Folder {folder_path} does not exist. Please update the path.")
        print("\nTo use this script:")
        print("1. Update 'folder_path' to point to your .obj files directory")
        print("2. Run: python example_usage.py")
        print("\nOr use the command line interface:")
        print("python GenerateSkel.py path/to/your/obj/folder")

if __name__ == "__main__":
    main()
