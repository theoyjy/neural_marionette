#!/usr/bin/env python3

import os
import pickle

def inspect_cached_data():
    """Inspect the structure of cached data"""
    
    folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    output_dir = os.path.join(folder_path, "generated_skeletons")
    
    # Check the first data file
    data_files = [f for f in os.listdir(output_dir) if f.endswith('_data.pkl')]
    if not data_files:
        print("No data files found!")
        return
    
    sample_file = os.path.join(output_dir, data_files[0])
    print(f"Inspecting: {sample_file}")
    
    with open(sample_file, 'rb') as f:
        data = pickle.load(f)
    
    print("Data structure:")
    if isinstance(data, dict):
        for key, value in data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {type(value)} with shape {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"  {key}: {type(value)} with length {len(value)}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    else:
        print(f"Data is of type: {type(data)}")
        if hasattr(data, '__dict__'):
            print("Attributes:")
            for attr in dir(data):
                if not attr.startswith('_'):
                    value = getattr(data, attr)
                    print(f"  {attr}: {type(value)}")

if __name__ == "__main__":
    inspect_cached_data()
