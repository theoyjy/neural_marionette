#!/usr/bin/env python3
"""
顶点颜色处理工具

重新设计：在插值过程中直接处理顶点颜色，而不是事后插值
"""

import os
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Dict, Any, List
import cv2

class VertexColorProcessor:
    """顶点颜色处理器 - 从贴图文件加载顶点颜色到reference mesh"""
    
    def __init__(self, mesh_folder_path: str):
        """
        初始化顶点颜色处理器
        
        Args:
            mesh_folder_path: 网格文件目录路径
        """
        self.mesh_folder_path = Path(mesh_folder_path)
        
        # 加载排序后的网格文件列表
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        if len(self.mesh_files) == 0:
            raise ValueError(f"在 {self.mesh_folder_path} 中未找到obj文件")
        
        # 加载排序后的贴图文件列表
        self.texture_files = sorted(list(self.mesh_folder_path.glob("*.jpg")) + list(self.mesh_folder_path.glob("*.png")))
        
        self.mesh_cache = {}  # 缓存已加载的mesh
        self.texture_cache = {}  # 缓存已加载的贴图
        
        print(f"🎨 初始化顶点颜色处理器: {mesh_folder_path}")
        print(f"  - 网格文件数量: {len(self.mesh_files)}")
        print(f"  - 贴图文件数量: {len(self.texture_files)}")
    
    def load_texture_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        为指定帧加载贴图文件
        
        Args:
            frame_id: 帧索引（排序后文件列表的索引，从0开始）
            
        Returns:
            贴图图像数组，如果找不到则返回None
        """
        if frame_id in self.texture_cache:
            return self.texture_cache[frame_id]
        
        # 检查帧索引范围
        if frame_id >= len(self.texture_files):
            print(f"⚠️  帧索引超出范围: {frame_id} >= {len(self.texture_files)}")
            return None
        
        # 直接使用排序后的贴图文件列表索引
        texture_file = self.texture_files[frame_id]
        
        print(f"📁 加载贴图: {texture_file.name} (索引 {frame_id})")
        
        try:
            # 使用OpenCV加载贴图
            texture = cv2.imread(str(texture_file))
            if texture is None:
                print(f"❌ 无法加载贴图文件: {texture_file}")
                return None
            
            # 转换为RGB格式
            texture_rgb = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            print(f"✅ 成功加载贴图: {texture_rgb.shape}")
            
            # 缓存结果
            self.texture_cache[frame_id] = texture_rgb
            return texture_rgb
            
        except Exception as e:
            print(f"❌ 加载贴图失败: {e}")
            return None
    
    def generate_vertex_colors_from_texture(self, mesh: o3d.geometry.TriangleMesh, texture: np.ndarray) -> np.ndarray:
        """
        从贴图生成顶点颜色（使用mesh中已有的UV坐标）
        
        Args:
            mesh: 网格对象
            texture: 贴图图像
            
        Returns:
            顶点颜色数组
        """
        try:
            vertices = np.asarray(mesh.vertices)
            
            # 检查mesh是否有UV坐标
            if not hasattr(mesh, 'triangle_uvs') or len(mesh.triangle_uvs) == 0:
                print(f"⚠️  mesh没有UV坐标，使用默认颜色")
                return self._generate_default_vertex_colors(vertices)
            
            print(f"✅ 使用mesh中已有的UV坐标: {len(mesh.triangle_uvs)} 个")
            
            # 获取UV坐标
            triangle_uvs = np.asarray(mesh.triangle_uvs)  # [num_triangles * 3, 2]
            triangles = np.asarray(mesh.triangles)  # [num_triangles, 3]
            
            # 为每个顶点计算平均UV坐标
            vertex_uvs = np.zeros((len(vertices), 2))
            vertex_counts = np.zeros(len(vertices))
            
            # 遍历每个三角形
            for i in range(len(triangles)):
                triangle = triangles[i]
                # 每个三角形有3个UV坐标
                for j in range(3):
                    vertex_idx = triangle[j]
                    uv_idx = i * 3 + j
                    vertex_uvs[vertex_idx] += triangle_uvs[uv_idx]
                    vertex_counts[vertex_idx] += 1
            
            # 计算平均UV坐标
            valid_vertices = vertex_counts > 0
            vertex_uvs[valid_vertices] /= vertex_counts[valid_vertices, np.newaxis]
            
            # 将UV坐标映射到贴图像素
            texture_height, texture_width = texture.shape[:2]
            u_coords = vertex_uvs[:, 0]  # U坐标
            v_coords = vertex_uvs[:, 1]  # V坐标
            
            # 将UV坐标转换为像素坐标
            u_pixels = np.clip(u_coords * (texture_width - 1), 0, texture_width - 1).astype(int)
            v_pixels = np.clip((1 - v_coords) * (texture_height - 1), 0, texture_height - 1).astype(int)
            
            # 从贴图中采样颜色
            vertex_colors = texture[v_pixels, u_pixels] / 255.0  # 归一化到[0,1]
            
            print(f"✅ 从贴图生成顶点颜色: {len(vertex_colors)} 个")
            print(f"  - UV坐标范围: U[{u_coords.min():.3f}, {u_coords.max():.3f}], V[{v_coords.min():.3f}, {v_coords.max():.3f}]")
            return vertex_colors
            
        except Exception as e:
            print(f"❌ 从贴图生成顶点颜色失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果失败，生成默认颜色
            return self._generate_default_vertex_colors(vertices)
    
    def load_reference_mesh_with_colors(self, frame_id: int) -> Optional[o3d.geometry.TriangleMesh]:
        """
        加载参考帧的mesh并从贴图加载顶点颜色
        
        Args:
            frame_id: 帧索引（排序后文件列表的索引，从0开始）
            
        Returns:
            带顶点颜色的mesh
        """
        if frame_id in self.mesh_cache:
            return self.mesh_cache[frame_id]
        
        # 检查帧索引范围
        if frame_id >= len(self.mesh_files):
            print(f"⚠️  帧索引超出范围: {frame_id} >= {len(self.mesh_files)}")
            return None
        
        # 直接使用排序后的网格文件列表索引
        mesh_file = self.mesh_files[frame_id]
        print(f"📁 加载参考mesh: {mesh_file.name} (索引 {frame_id})")
        
        # 加载mesh
        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        
        if len(mesh.vertices) == 0:
            print(f"❌ mesh文件为空: {mesh_file}")
            return None
        
        # 检查是否有顶点颜色
        if len(mesh.vertex_colors) == 0:
            print(f"⚠️  mesh没有顶点颜色，尝试从贴图加载...")
            
            # 尝试加载贴图
            texture = self.load_texture_for_frame(frame_id)
            if texture is not None:
                # 从贴图生成顶点颜色
                vertex_colors = self.generate_vertex_colors_from_texture(mesh, texture)
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                print(f"✅ 从贴图成功加载顶点颜色")
            else:
                print(f"⚠️  无法加载贴图，生成默认颜色")
                vertices = np.asarray(mesh.vertices)
                default_colors = self._generate_default_vertex_colors(vertices)
                mesh.vertex_colors = o3d.utility.Vector3dVector(default_colors)
        else:
            print(f"✅ 找到顶点颜色: {len(mesh.vertex_colors)} 个")
        
        # 确保有法线
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
            print(f"✅ 计算顶点法线")
        
        # 缓存结果
        self.mesh_cache[frame_id] = mesh
        return mesh
    
    def save_reference_mesh_with_colors(self, mesh: o3d.geometry.TriangleMesh, frame_id: int, output_dir: Path):
        """
        保存带顶点颜色的reference mesh（用于debug，同时生成PLY和OBJ格式）
        
        Args:
            mesh: 带顶点颜色的mesh
            frame_id: 帧索引（排序后文件列表的索引，从0开始）
            output_dir: 输出目录
        """
        try:
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 确保有法线
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # 保存PLY格式（兼容Assimp）
            ply_filename = f"reference_frame_{frame_id:05d}_with_colors.ply"
            ply_filepath = output_dir / ply_filename
            ply_success = o3d.io.write_triangle_mesh(str(ply_filepath), mesh, write_ascii=True)
            
            # 保存OBJ格式（兼容传统工具）
            obj_filename = f"reference_frame_{frame_id:05d}_with_colors.obj"
            obj_filepath = output_dir / obj_filename
            obj_success = o3d.io.write_triangle_mesh(str(obj_filepath), mesh)
            
            if ply_success and obj_success:
                print(f"✅ 保存带顶点颜色的reference mesh: {ply_filename}, {obj_filename}")
                return str(ply_filepath)  # 返回PLY文件路径作为主要输出
            else:
                print(f"❌ 保存reference mesh失败: PLY={ply_success}, OBJ={obj_success}")
                return None
                
        except Exception as e:
            print(f"❌ 保存reference mesh异常: {e}")
            return None
    
    def _generate_default_vertex_colors(self, vertices: np.ndarray) -> np.ndarray:
        """生成默认顶点颜色（基于顶点位置）"""
        # 归一化顶点坐标到[0,1]范围
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        normalized_vertices = (vertices - min_coords) / (max_coords - min_coords + 1e-8)
        
        # 使用归一化坐标作为颜色 (X->R, Y->G, Z->B)
        colors = np.clip(normalized_vertices, 0, 1)
        return colors
    
    def save_mesh_with_vertex_colors(self, mesh: o3d.geometry.TriangleMesh, 
                                   output_path: Path, frame_idx: int) -> Optional[str]:
        """
        保存带顶点颜色的mesh（同时生成PLY和OBJ格式）
        
        Args:
            mesh: 要保存的mesh
            output_path: 输出目录
            frame_idx: 帧索引（排序后文件列表的索引，从0开始）
            
        Returns:
            保存的文件路径，如果失败返回None
        """
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 确保有法线
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # 保存PLY格式（兼容Assimp）
            ply_filename = f"frame_{frame_idx:05d}_with_colors.ply"
            ply_filepath = output_path / ply_filename
            ply_success = o3d.io.write_triangle_mesh(str(ply_filepath), mesh, write_ascii=True)
            
            # 保存OBJ格式（兼容传统工具）
            obj_filename = f"frame_{frame_idx:05d}_with_colors.obj"
            obj_filepath = output_path / obj_filename
            obj_success = o3d.io.write_triangle_mesh(str(obj_filepath), mesh)
            
            if ply_success and obj_success:
                print(f"✅ 保存带顶点颜色的mesh: {ply_filename}, {obj_filename}")
                return str(ply_filepath)  # 返回PLY文件路径作为主要输出
            else:
                print(f"❌ 保存mesh失败: PLY={ply_success}, OBJ={obj_success}")
                return None
                
        except Exception as e:
            print(f"❌ 保存mesh异常: {e}")
            return None

def integrate_vertex_color_processing(interpolator, mesh_folder_path: str):
    """
    将顶点颜色处理集成到插值器中
    
    新的设计：在插值过程中直接处理顶点颜色
    """
    print("🎨 集成顶点颜色处理到插值器...")
    
    # 创建顶点颜色处理器
    vertex_color_processor = VertexColorProcessor(mesh_folder_path)
    
    # 保存原始的插值方法
    original_method = interpolator.generate_interpolated_frames
    
    def enhanced_generate_interpolated_frames(frame_start, frame_end, num_interpolate, 
                                            max_optimize_frames=5, optimize_weights=True, 
                                            output_dir=None, debug_frames=None, smooth_mesh=False, 
                                            subdivide_iter=3, use_vertex_colors=True,
                                            save_npy_files=False, save_standard_obj=True):
        """增强的插值帧生成方法 - 在插值过程中直接处理顶点颜色"""
        
        print(f"🎨 开始插值生成，启用顶点颜色处理...")
        
        # 在插值之前加载参考帧的mesh和顶点颜色
        print(f"🎨 在插值之前加载reference mesh和顶点颜色...")
        start_mesh = vertex_color_processor.load_reference_mesh_with_colors(frame_start)
        end_mesh = vertex_color_processor.load_reference_mesh_with_colors(frame_end)
        
        if start_mesh is None or end_mesh is None:
            print(f"❌ 无法加载参考帧mesh，使用原始插值方法")
            return original_method(
                frame_start, frame_end, num_interpolate, max_optimize_frames, 
                optimize_weights, output_dir, debug_frames, smooth_mesh, subdivide_iter,
                save_npy_files=save_npy_files,
                save_standard_obj=save_standard_obj
            )
        
        # 保存带顶点颜色的reference mesh（用于debug）
        if output_dir:
            debug_dir = Path(output_dir) / "debug_reference_meshes"
            vertex_color_processor.save_reference_mesh_with_colors(start_mesh, frame_start, debug_dir)
            vertex_color_processor.save_reference_mesh_with_colors(end_mesh, frame_end, debug_dir)
            print(f"✅ 保存reference mesh到: {debug_dir}")
        
        # 调用原始的插值方法，但不保存文件（我们会在后面手动保存）
        interpolated_frames = original_method(
            frame_start, frame_end, num_interpolate, max_optimize_frames, 
            optimize_weights, output_dir, debug_frames, smooth_mesh, subdivide_iter,
            use_vertex_colors=False,  # 我们会在后面手动处理顶点颜色
            save_npy_files=save_npy_files,
            save_standard_obj=False  # 不保存标准obj文件，我们会保存带颜色的版本
        )
        
        # 如果启用了顶点颜色处理，为每个插值帧添加顶点颜色
        if use_vertex_colors and interpolated_frames:
            print(f"🎨 为插值帧添加顶点颜色...")
            
            for i, frame_data in enumerate(interpolated_frames):
                if 'mesh' in frame_data:
                    mesh = frame_data['mesh']
                    frame_idx = frame_data.get('frame_idx', i)
                    t = frame_data.get('interpolation_t', i / len(interpolated_frames))
                    
                    # 从参考帧插值顶点颜色
                    interpolated_colors = _interpolate_vertex_colors_from_references(
                        start_mesh, end_mesh, t
                    )
                    
                    if interpolated_colors is not None:
                        # 应用插值的顶点颜色
                        mesh.vertex_colors = o3d.utility.Vector3dVector(interpolated_colors)
                        print(f"✅ 为帧 {frame_idx} 应用插值顶点颜色")
                    
                    # 确保有法线
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
        """从参考帧插值顶点颜色"""
        try:
            start_colors = np.asarray(start_mesh.vertex_colors)
            end_colors = np.asarray(end_mesh.vertex_colors)
            
            # 线性插值顶点颜色
            interpolated_colors = start_colors * (1 - t) + end_colors * t
            return interpolated_colors
            
        except Exception as e:
            print(f"❌ 顶点颜色插值失败: {e}")
            return None
    
    # 替换插值器的方法
    interpolator.generate_interpolated_frames = enhanced_generate_interpolated_frames
    
    print("✅ 顶点颜色处理集成完成")

# 为了向后兼容，保留TextureProcessor类名但实际使用VertexColorProcessor
class TextureProcessor(VertexColorProcessor):
    """向后兼容的TextureProcessor类"""
    pass 