#!/usr/bin/env python3
"""
Vertex color processing tool

Redesigned: Process vertex colors directly during interpolation, not after interpolation
"""

import os
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2

class VertexColorProcessor:
    """Vertex color processor - Load vertex colors from texture files to reference mesh"""
    
    def __init__(self, mesh_folder_path: str):
        """
        Initialize vertex color processor
        
        Args:
            mesh_folder_path: Mesh file directory path
        """
        self.mesh_folder_path = Path(mesh_folder_path)
        
        # Load sorted mesh file list
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        if len(self.mesh_files) == 0:
            raise ValueError(f"No obj files found in {self.mesh_folder_path}")
        
        # Load sorted texture file list
        self.texture_files = sorted(list(self.mesh_folder_path.glob("*.jpg")) + list(self.mesh_folder_path.glob("*.png")))
        
        self.mesh_cache = {}  # Cache loaded mesh
        self.texture_cache = {}  # Cache loaded texture
        
        print(f"Initialize vertex color processor: {mesh_folder_path}")
        print(f"  - Mesh file count: {len(self.mesh_files)}")
        print(f"  - Texture file count: {len(self.texture_files)}")
    
    def load_texture_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Load texture file for specified frame
        
        Args:
            frame_id: Frame index (index in sorted file list, starting from 0)
            
        Returns:
            Texture image array, if not found return None
        """
        if frame_id in self.texture_cache:
            return self.texture_cache[frame_id]
        
        # 检查帧索引范围
        if frame_id >= len(self.texture_files):
            print(f"Frame index out of range: {frame_id} >= {len(self.texture_files)}")
            return None
        
        # Directly use sorted texture file list index
        texture_file = self.texture_files[frame_id]
        
        print(f"Load texture: {texture_file.name} (index {frame_id})")
        
        try:
            # 使用OpenCV加载贴图
            texture = cv2.imread(str(texture_file))
            if texture is None:
                print(f"Cannot load texture file: {texture_file}")
                return None
            
            # 转换为RGB格式
            texture_rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            print(f"Successfully loaded texture: {texture_rgb.shape}")
            
            # 缓存结果
            self.texture_cache[frame_id] = texture_rgb
            return texture_rgb
            
        except Exception as e:
            print(f"Load texture failed: {e}")
            return None
    
    def generate_vertex_colors_from_texture(self,   mesh: o3d.geometry.TriangleMesh, texture: np.ndarray) -> np.ndarray:
        """
        Generate vertex colors from texture (using UV coordinates in mesh)
        
        Args:
            mesh: Mesh object
            texture: Texture image
            
        Returns:
            Vertex color array
        """
        try:
            vertices = np.asarray(mesh.vertices)
            
            # Check if mesh has UV coordinates
            if not hasattr(mesh, 'triangle_uvs') or len(mesh.triangle_uvs) == 0:
                print(f"Mesh has no UV coordinates, using default colors")
                return self._generate_default_vertex_colors(vertices)
            
            print(f"Using UV coordinates in mesh: {len(mesh.triangle_uvs)}")
            
            # Get UV coordinates
            triangle_uvs = np.asarray(mesh.triangle_uvs)  # [num_triangles * 3, 2]
            triangles = np.asarray(mesh.triangles)  # [num_triangles, 3]
            
            # Calculate average UV coordinates for each vertex
            vertex_uvs = np.zeros((len(vertices), 2))
            vertex_counts = np.zeros(len(vertices))
            
            # Traverse each triangle
            for i in range(len(triangles)):
                triangle = triangles[i]
                # Each triangle has 3 UV coordinates
                for j in range(3):
                    vertex_idx = triangle[j]
                    uv_idx = i * 3 + j
                    vertex_uvs[vertex_idx] += triangle_uvs[uv_idx]
                    vertex_counts[vertex_idx] += 1
            
            # Calculate average UV coordinates
            valid_vertices = vertex_counts > 0
            vertex_uvs[valid_vertices] /= vertex_counts[valid_vertices, np.newaxis]
            
            # Map UV coordinates to texture pixels
            texture_height, texture_width = texture.shape[:2]
            u_coords = vertex_uvs[:, 0]  # U coordinates
            v_coords = vertex_uvs[:, 1]  # V coordinates
            
            # Convert UV coordinates to pixel coordinates
            u_pixels = np.clip(u_coords * (texture_width - 1), 0, texture_width - 1).astype(int)
            v_pixels = np.clip((1 - v_coords) * (texture_height - 1), 0, texture_height - 1).astype(int)
            
            # Sample colors from texture
            vertex_colors = texture[v_pixels, u_pixels] / 255.0  # Normalize to [0,1]
            
            print(f"Generate vertex colors from texture: {len(vertex_colors)}")
            print(f"  - UV coordinates range: U[{u_coords.min():.3f}, {u_coords.max():.3f}], V[{v_coords.min():.3f}, {v_coords.max():.3f}]")
            return vertex_colors
            
        except Exception as e:
            print(f"Generate vertex colors from texture failed: {e}")
            import traceback
            traceback.print_exc()
            # If failed, generate default colors
            return self._generate_default_vertex_colors(vertices)
    
    def load_reference_mesh_with_colors(self, frame_id: int) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Load reference frame mesh and load vertex colors from texture
        
        Args:
            frame_id: Frame index (index in sorted file list, starting from 0)
            
        Returns:
            Mesh with vertex colors
        """
        if frame_id in self.mesh_cache:
            return self.mesh_cache[frame_id]
        
        # Check frame index range
        if frame_id >= len(self.mesh_files):
            print(f"Frame index out of range: {frame_id} >= {len(self.mesh_files)}")
            return None
        
        # Directly use sorted mesh file list index
        mesh_file = self.mesh_files[frame_id]
        print(f"Load reference mesh: {mesh_file.name} (index {frame_id})")
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        
        if len(mesh.vertices) == 0:
            print(f"Mesh file is empty: {mesh_file}")
            return None
        
        # Check if there are vertex colors
        if len(mesh.vertex_colors) == 0:
            print(f"Mesh has no vertex colors, trying to load from texture...")
            
            # Try to load texture
            texture = self.load_texture_for_frame(frame_id)
            if texture is not None:
                # Generate vertex colors from texture
                vertex_colors = self.generate_vertex_colors_from_texture(mesh, texture)
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                print(f"Successfully loaded vertex colors from texture")
            else:
                print(f"Cannot load texture, generating default colors")
                vertices = np.asarray(mesh.vertices)
                default_colors = self._generate_default_vertex_colors(vertices)
                mesh.vertex_colors = o3d.utility.Vector3dVector(default_colors)
        else:
            print(f"Found vertex colors: {len(mesh.vertex_colors)}")
        
        # Ensure there are vertex normals
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            print(f"Compute vertex normals")
        
        # Cache result
        self.mesh_cache[frame_id] = mesh
        return mesh
    
    def save_reference_mesh_with_colors(self, mesh: o3d.geometry.TriangleMesh, frame_id: int, output_dir: Path):
        """
        Save reference mesh with vertex colors (for debug, generate PLY and OBJ format)
        
        Args:
            mesh: Mesh with vertex colors
            frame_id: Frame index (index in sorted file list, starting from 0)
            output_dir: Output directory
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure there are vertex normals
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # Save PLY format (compatible with Assimp)
            # ply_filename = f"reference_frame_{frame_id:05d}_with_colors.ply"
            # ply_filepath = output_dir / ply_filename
            # ply_success = o3d.io.write_triangle_mesh(str(ply_filepath), mesh, write_ascii=True)
            
            # Save OBJ format (compatible with traditional tools)
            obj_filename = f"reference_frame_{frame_id:05d}_with_colors.obj"
            obj_filepath = output_dir / obj_filename
            obj_success = o3d.io.write_triangle_mesh(str(obj_filepath), mesh)
            
            if obj_success:
                print(f"Save reference mesh with vertex colors: {ply_filename}, {obj_filename}")
                return str(obj_filepath)  # Return PLY file path as main output
            else:
                print(f"Save reference mesh failed: PLY={ply_success}, OBJ={obj_success}")
                return None
                
        except Exception as e:
            print(f"Save reference mesh exception: {e}")
            return None
    
    def _generate_default_vertex_colors(self, vertices: np.ndarray) -> np.ndarray:
        """Generate default vertex colors (based on vertex positions)"""
        # Normalize vertex coordinates to [0,1] range
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords + 1e-8)
        
        # Use normalized coordinates as colors (X->R, Y->G, Z->B)
        colors = np.clip(normalized_vertices, 0, 1)
        return colors
    
    def save_mesh_with_vertex_colors(self, mesh: o3d.geometry.TriangleMesh, 
                                   output_path: Path, frame_idx: int) -> Optional[str]:
        """
        Save mesh with vertex colors (generate PLY and OBJ format)
        
        Args:
            mesh: Mesh to save
            output_path: Output directory
            frame_idx: Frame index (index in sorted file list, starting from 0)
            
        Returns:
            Saved file path, if failed return None
        """
        try:
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Ensure there are vertex normals
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # Save PLY format (compatible with Assimp)
            # ply_filename = f"frame_{frame_idx:05d}_with_colors.ply"
            # ply_filepath = output_path / ply_filename
            # ply_success = o3d.io.write_triangle_mesh(str(ply_filepath), mesh, write_ascii=True)
            
            # Save OBJ format (compatible with traditional tools)
            obj_filename = f"frame_{frame_idx:05d}_with_colors.obj"
            obj_filepath = output_path / obj_filename
            obj_success = o3d.io.write_triangle_mesh(str(obj_filepath), mesh)
            
            if obj_success:
                print(f"Save mesh with vertex colors: {obj_filename}")
                return str(obj_filename)  # Return PLY file path as main output
            else:
                print(f"Save mesh failed: OBJ={obj_success}")
                return None
                
        except Exception as e:
            print(f"Save mesh exception: {e}")
            return None

def integrate_vertex_color_processing(interpolator, mesh_folder_path: str, evaluation_mode: bool = False, method: str = "baseline"):
    """
    Integrate vertex color processing into interpolator
    
    New design: Process vertex colors directly during interpolation
    Args:
        evaluation_mode: If True, skip color processing but keep enhanced file saving
        method: The interpolation method being used.
    """
    mode_str = "evaluation mode (skip color processing)" if evaluation_mode else "full color processing"
    print(f"Integrate vertex color processing into interpolator ({mode_str})...")
    
    # Create vertex color processor (only if not in evaluation mode)
    vertex_color_processor = None if evaluation_mode else VertexColorProcessor(mesh_folder_path)
    
    # Save original interpolation method
    original_method = interpolator.generate_interpolated_frames
    
    def enhanced_generate_interpolated_frames(frame_start, frame_end, num_interpolate, 
                                            max_optimize_frames=5, optimize_weights=True, 
                                            output_dir=None, debug_frames=None, smooth_mesh=False, 
                                            subdivide_iter=3, use_vertex_colors=True,
                                            save_npy_files=False, save_standard_obj=True):
        """Enhanced interpolation frame generation method - process vertex colors directly during interpolation"""
        
        if evaluation_mode:
            print(f"Start interpolation generation in evaluation mode (skip color processing)...")
            # In evaluation mode, skip color processing but use enhanced file saving
            start_mesh = None
            end_mesh = None
        else:
            print(f"Start interpolation generation, enable vertex color processing...")
            
            # Load reference frame mesh and vertex colors before interpolation
            print(f"Load reference frame mesh and vertex colors before interpolation...")
            start_mesh = vertex_color_processor.load_reference_mesh_with_colors(frame_start)
            end_mesh = vertex_color_processor.load_reference_mesh_with_colors(frame_end)
        
        if (start_mesh is None or end_mesh is None) and not evaluation_mode:
            print(f"Cannot load reference frame mesh, using original interpolation method")
            return original_method(
                frame_start, frame_end, num_interpolate, max_optimize_frames, 
                optimize_weights, output_dir, debug_frames, smooth_mesh, subdivide_iter,
                save_npy_files=save_npy_files,
                save_standard_obj=save_standard_obj
            )
        
        # Save reference mesh with vertex colors (for debug) - only if not in evaluation mode
        if output_dir and not evaluation_mode:
            debug_dir = Path(output_dir) / "debug_reference_meshes"
            vertex_color_processor.save_reference_mesh_with_colors(start_mesh, frame_start, debug_dir)
            vertex_color_processor.save_reference_mesh_with_colors(end_mesh, frame_end, debug_dir)
            print(f"Save reference mesh to: {debug_dir}")
        
        # Call original interpolation method, but do not save files (we will save manually later)
        interpolated_frames = original_method(
            frame_start, frame_end, num_interpolate, max_optimize_frames, 
            optimize_weights, output_dir, debug_frames, smooth_mesh, subdivide_iter,
            use_vertex_colors=False,  # We will handle vertex colors manually later
            save_npy_files=save_npy_files,
            save_standard_obj=False  # Do not save standard obj file, we will save the colored version
        )
        
        # If vertex color processing is enabled, add vertex colors to each interpolated frame
        if use_vertex_colors and interpolated_frames:
            print(f"Add vertex colors to each interpolated frame...")
            
            for i, frame_data in enumerate(interpolated_frames):
                if 'mesh' in frame_data:
                    mesh = frame_data['mesh']
                    frame_idx = frame_data.get('frame_idx', i)
                    t = frame_data.get('interpolation_t', i / len(interpolated_frames))
                    
                    # Different color handling strategy based on method
                    if method == 'baseline':
                        # Baseline method: use start frame colors
                        interpolated_colors = np.asarray(start_mesh.vertex_colors)
                    elif method == 'dual_reference':
                        # Dual reference method: use colors from the actual reference frame being used
                        # Dual reference uses piecewise logic: t < 0.5 use start, t >= 0.5 use end
                        if t < 0.5:
                            interpolated_colors = np.asarray(start_mesh.vertex_colors)
                            print(f"  - Dual reference: t={t:.3f} < 0.5, using start frame colors")
                        else:
                            interpolated_colors = np.asarray(end_mesh.vertex_colors)
                            print(f"  - Dual reference: t={t:.3f} >= 0.5, using end frame colors")
                    else:
                        # Other methods: interpolate between start and end frame colors
                        interpolated_colors = _interpolate_vertex_colors_from_references(
                            start_mesh, end_mesh, t
                        )
                    
                    if interpolated_colors is not None:
                        # Apply interpolated vertex colors
                        mesh.vertex_colors = o3d.utility.Vector3dVector(interpolated_colors)
                        print(f"Apply interpolated vertex colors to frame {frame_idx}")
                    
                    # Ensure there are vertex normals
                    if not mesh.has_vertex_normals():
                        mesh.compute_vertex_normals()
                    
                    # 保存带颜色的mesh
                    if output_dir:
                        mesh_file = vertex_color_processor.save_mesh_with_vertex_colors(
                            mesh, Path(output_dir), frame_idx
                        )
                        if mesh_file:
                            frame_data['mesh_file'] = mesh_file
        
        return interpolated_frames
    
    def _interpolate_vertex_colors_from_references(start_mesh, end_mesh, t):
        """Interpolate vertex colors from reference frames"""
        try:
            start_colors = np.asarray(start_mesh.vertex_colors)
            end_colors = np.asarray(end_mesh.vertex_colors)
            
            # Linear interpolation of vertex colors
            interpolated_colors = start_colors * (1 - t) + end_colors * t
            return interpolated_colors
            
        except Exception as e:
            print(f"Vertex color interpolation failed: {e}")
            return None
    
    # Replace the interpolation method
    interpolator.generate_interpolated_frames = enhanced_generate_interpolated_frames
    
    print("Vertex color processing integration completed")

# For backward compatibility, keep TextureProcessor class name but use VertexColorProcessor internally
class TextureProcessor(VertexColorProcessor):
    """Backward compatible TextureProcessor class"""
    pass 