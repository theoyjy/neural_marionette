#!/usr/bin/env python3

import sys
import os
import traceback

# Add the current directory to the path so we can import GenerateSkel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GenerateSkel import process_obj_files

def test_single_mesh():
    """Test processing with just one mesh to see the complete output"""
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    # Create a temporary folder with just one mesh file
    import shutil
    temp_folder = "D:/Code/neural_marionette/temp_single_mesh"
    
    try:
        # Clean up if exists
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
            
        os.makedirs(temp_folder)
        
        # Copy just the first mesh file
        source_file = os.path.join(folder_path, "Frame_00001_textured_hd_t_s_c.obj")
        dest_file = os.path.join(temp_folder, "Frame_00001_textured_hd_t_s_c.obj")
        shutil.copy2(source_file, dest_file)
        
        print(f"Processing single mesh: {dest_file}")
        
        # Process this single mesh
        process_obj_files(
            folder_path=temp_folder,
            output_dir=None,
            is_bind=False,
            visualize=False
        )
        
        print("\n=== Results ===")
        output_dir = os.path.join(temp_folder, "generated_skeletons")
        if os.path.exists(output_dir):
            for file in os.listdir(output_dir):
                print(f"  {file}")
        else:
            print("No output directory created")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        
    finally:
        # Clean up
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

if __name__ == "__main__":
    test_single_mesh()
