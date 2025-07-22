# GenerateSkel.py - Batch Skeleton Generation

This script processes all `.obj` files in a given folder and generates skeletons using NeuralMarionette and DemBones for skinning weight computation.

## Features

1. **Batch Processing**: Processes all `.obj` files in a specified folder
2. **Voxel Processing**: Converts meshes to voxels for NeuralMarionette input
3. **Skeleton Prediction**: Uses NeuralMarionette to predict joints and parent hierarchy
4. **Data Caching**: Saves intermediate results to avoid recomputation
5. **DemBones Integration**: Generates skinning weights and rest poses for all meshes
6. **Visualization**: Optional real-time visualization of skeletons and results
7. **Multiple Output Formats**: Saves data in various formats for different use cases

## Usage

### Command Line Interface

```bash
# Basic usage - process all .obj files in a folder
python GenerateSkel.py path/to/your/obj/folder

# With custom output directory
python GenerateSkel.py path/to/your/obj/folder --output_dir path/to/output

# Disable visualization (for batch processing)
python GenerateSkel.py path/to/your/obj/folder --no_visualize

# Use bind transform for mesh coordinates
python GenerateSkel.py path/to/your/obj/folder --is_bind
```

### Python API

```python
from GenerateSkel import process_obj_files

# Process with default settings
process_obj_files(
    folder_path="path/to/your/obj/folder",
    output_dir=None,  # Will create 'generated_skeletons' subfolder
    is_bind=False,
    visualize=True
)
```

## Requirements

Make sure you have the following dependencies installed:
- `torch`
- `open3d`
- `scipy`
- `matplotlib`
- `numpy`
- `py_dem_bones`

You can install them using:
```bash
pip install torch open3d scipy matplotlib numpy
# Note: py_dem_bones may need special installation
```

## Output Files

The script generates several types of output files:

### Per-Mesh Files
- `{mesh_name}_data.pkl`: Raw processing data (joints, rotations, normalization params)
- `{mesh_name}_skeleton.pkl`: Skeleton-specific results (joints, parents, rotations)
- `{mesh_name}_rest_pose.obj`: Rest pose mesh in world coordinates

### Combined Results
- `dembone_results.pkl`: Complete DemBones results including:
  - Rest pose vertices
  - Skinning weights matrix
  - Bone transformations
  - Parent hierarchy
  - Template mesh data

## Workflow

1. **Mesh Loading**: Each `.obj` file is loaded and converted to voxels
2. **Skeleton Detection**: NeuralMarionette predicts joint positions and hierarchy
3. **Data Caching**: Results are saved to avoid reprocessing (check for existing `*_data.pkl` files)
4. **Vertex Unification**: All meshes are mapped to a common vertex template using the mesh with the most vertices
5. **DemBones Processing**: Skinning weights and rest poses are computed across all meshes
6. **Visualization**: Optional display of skeletons, rest poses, and skinning weights
7. **Output Generation**: Multiple output formats for different downstream applications

## Key Functions

- `process_obj_files()`: Main function to process all meshes in a folder
- `process_single_mesh()`: Process individual mesh and extract skeleton
- `solve_with_dem_bones()`: Run DemBones algorithm on multiple meshes
- `load_voxel_from_mesh()`: Convert mesh to voxel representation
- `sanitize_parents()`: Fix parent hierarchy issues
- `draw_skeleton()`: Visualization utilities
- `draw_skinning_colors()`: Skinning weight visualization

## Example Directory Structure

```
your_project/
├── input_meshes/
│   ├── character1.obj
│   ├── character2.obj
│   └── character3.obj
├── generated_skeletons/
│   ├── character1_data.pkl
│   ├── character1_skeleton.pkl
│   ├── character1_rest_pose.obj
│   ├── character2_data.pkl
│   ├── character2_skeleton.pkl
│   ├── character2_rest_pose.obj
│   ├── character3_data.pkl
│   ├── character3_skeleton.pkl
│   ├── character3_rest_pose.obj
│   └── dembone_results.pkl
```

## Notes

- The script uses the first frame as the rest pose for DemBones
- Meshes with different vertex counts are remapped using nearest neighbor interpolation
- Parent hierarchy cycles are automatically detected and fixed
- Intermediate results are cached to speed up subsequent runs
- The template mesh (with most vertices) is used as the reference for vertex mapping

## Troubleshooting

1. **CUDA Errors**: Make sure you have a CUDA-compatible GPU and PyTorch with CUDA support
2. **Memory Issues**: Reduce the number of meshes processed simultaneously or use smaller meshes
3. **Import Errors**: Ensure all dependencies are properly installed
4. **Visualization Issues**: Use `--no_visualize` flag if running in headless environment
5. **DemBones Errors**: Check that `py_dem_bones` is properly compiled and installed
