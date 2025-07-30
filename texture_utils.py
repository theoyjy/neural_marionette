#!/usr/bin/env python3
"""
纹理处理工具

处理插值mesh的纹理生成、UV坐标映射和材质文件创建
"""

import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import re
from typing import Optional, Tuple, Dict, Any
import time

class TextureProcessor:
    """纹理处理器"""
    
    def __init__(self, mesh_folder_path: str, texture_folder_path: Optional[str] = None):
        """
        初始化纹理处理器
        
        Args:
            mesh_folder_path: 网格文件目录路径
            texture_folder_path: 纹理文件目录路径（可选）
        """
        self.mesh_folder_path = Path(mesh_folder_path)
        self.texture_folder_path = Path(texture_folder_path) if texture_folder_path else self.mesh_folder_path
        
        # 加载mesh文件列表
        self.mesh_files = sorted(list(self.mesh_folder_path.glob("*.obj")))
        print(f"📁 找到 {len(self.mesh_files)} 个mesh文件")
        
        # 加载纹理文件列表
        self.texture_files = sorted(list(self.texture_folder_path.glob("*.jpg")) + 
                                   list(self.texture_folder_path.glob("*.png")))
        print(f"📁 找到 {len(self.texture_files)} 个纹理文件")
        
        # 缓存
        self.texture_cache = {}
        self.interpolation_cache = {}
    
    def load_original_mesh_texture_info(self, frame_id):
        """从原始mesh中加载纹理信息"""
        try:
            # 查找对应的原始mesh文件
            original_mesh_file = None
            for mesh_file in self.mesh_files:
                if f"Frame_{frame_id:05d}" in mesh_file.name or f"frame_{frame_id:05d}" in mesh_file.name:
                    original_mesh_file = mesh_file
                    break
            
            if original_mesh_file is None:
                print(f"⚠️  未找到帧 {frame_id} 的原始mesh文件")
                return None, None, None
            
            print(f"📁 加载原始mesh: {original_mesh_file}")
            
            # 加载原始mesh
            original_mesh = o3d.io.read_triangle_mesh(str(original_mesh_file))
            
            # 检查是否有纹理坐标
            if hasattr(original_mesh, 'triangle_uvs') and len(original_mesh.triangle_uvs) > 0:
                print(f"✅ 找到原始UV坐标: {len(original_mesh.triangle_uvs)} 个")
                uvs = np.asarray(original_mesh.triangle_uvs)
            else:
                print(f"⚠️  原始mesh没有UV坐标")
                return None, None, None
            
            # 查找对应的纹理文件
            texture_file = None
            mesh_name = original_mesh_file.stem
            possible_texture_files = [
                original_mesh_file.parent / f"{mesh_name}.jpg",
                original_mesh_file.parent / f"{mesh_name}.png",
                original_mesh_file.parent / f"Frame_{frame_id:05d}_textured_hd_t_s_c.jpg",
                original_mesh_file.parent / f"frame_{frame_id:05d}_textured_hd_t_s_c.jpg"
            ]
            
            for tex_file in possible_texture_files:
                if tex_file.exists():
                    texture_file = tex_file
                    break
            
            if texture_file is None:
                print(f"⚠️  未找到帧 {frame_id} 的纹理文件")
                return None, None, None
            
            print(f"📁 加载纹理文件: {texture_file}")
            
            # 加载纹理
            texture = cv2.imread(str(texture_file))
            if texture is None:
                print(f"❌ 无法加载纹理文件: {texture_file}")
                return None, None, None
            
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
            print(f"✅ 纹理加载成功: {texture.shape}")
            
            return original_mesh, uvs, texture
            
        except Exception as e:
            print(f"❌ 加载原始mesh纹理信息失败: {e}")
            return None, None, None
    
    def save_mesh_with_colors(self, mesh: o3d.geometry.TriangleMesh, 
                             output_path: Path, frame_idx: int):
        """保存带顶点颜色的mesh（保留插值后mesh的原始纹理坐标和顶点颜色）"""
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存mesh文件（保留原有的纹理坐标和顶点颜色）
            mesh_filename = f"interpolated_frame_{frame_idx:04d}_colored.obj"
            mesh_file = output_path / mesh_filename
            
            # 检查mesh是否有纹理坐标和顶点颜色
            has_uvs = hasattr(mesh, 'triangle_uvs') and len(mesh.triangle_uvs) > 0
            has_vertex_colors = hasattr(mesh, 'vertex_colors') and len(mesh.vertex_colors) > 0
            
            if has_uvs:
                print(f"✅ 使用插值mesh的原始纹理坐标: {len(mesh.triangle_uvs)} 个")
            else:
                print(f"⚠️  插值mesh没有纹理坐标")
            
            if has_vertex_colors:
                print(f"✅ 使用插值mesh的原始顶点颜色: {len(mesh.vertex_colors)} 个")
            else:
                print(f"⚠️  插值mesh没有顶点颜色")
            
            # 确保有法线
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            # 直接保存mesh，保留其原有的纹理坐标和顶点颜色
            o3d.io.write_triangle_mesh(str(mesh_file), mesh)
            print(f"✅ 保存mesh文件: {mesh_file}")
            
            return str(mesh_file)
            
        except Exception as e:
            print(f"❌ 保存mesh文件失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_standard_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                          output_path: Path, frame_idx: int):
        """保存标准mesh文件（不包含顶点颜色）"""
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存标准mesh文件
            mesh_filename = f"interpolated_frame_{frame_idx:04d}.obj"
            mesh_file = output_path / mesh_filename
            
            # 创建标准mesh副本
            standard_mesh = o3d.geometry.TriangleMesh()
            standard_mesh.vertices = mesh.vertices
            standard_mesh.triangles = mesh.triangles
            
            # 确保有法线
            if hasattr(mesh, 'vertex_normals') and len(mesh.vertex_normals) > 0:
                standard_mesh.vertex_normals = mesh.vertex_normals
            else:
                standard_mesh.compute_vertex_normals()
            
            # 保存标准mesh文件
            o3d.io.write_triangle_mesh(str(mesh_file), standard_mesh)
            print(f"✅ 保存标准mesh文件: {mesh_file}")
            
            return str(mesh_file)
            
        except Exception as e:
            print(f"❌ 保存标准mesh文件失败: {e}")
            return None
    
    def create_material_file(self, output_path: Path, frame_idx: int, texture_filename: str):
        """创建材质文件"""
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 创建材质文件
            mtl_filename = f"interpolated_frame_{frame_idx:04d}.mtl"
            mtl_file = output_path / mtl_filename
            
            mtl_content = f"""# Material file for interpolated frame {frame_idx}
newmtl material_interpolated_frame_{frame_idx:04d}
Ka 0.200000 0.200000 0.200000
Kd 1.000000 1.000000 1.000000
Ks 1.000000 1.000000 1.000000
Tr 1.000000
illum 2
Ns 0.000000
map_Kd {texture_filename}
"""
            
            with open(mtl_file, 'w') as f:
                f.write(mtl_content)
            
            print(f"✅ 创建材质文件: {mtl_file}")
            return str(mtl_file)
            
        except Exception as e:
            print(f"❌ 创建材质文件失败: {e}")
            return None
    
    def save_texture(self, texture: np.ndarray, output_path: Path, frame_idx: int):
        """保存纹理文件"""
        try:
            # 确保输出目录存在
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 保存纹理文件
            texture_filename = f"texture_{frame_idx:04d}.jpg"
            texture_file = output_path / texture_filename
            
            # 转换回BGR格式用于保存
            texture_bgr = cv2.cvtColor(texture, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(texture_file), texture_bgr)
            
            print(f"✅ 保存纹理文件: {texture_file}")
            return str(texture_file)
            
        except Exception as e:
            print(f"❌ 保存纹理文件失败: {e}")
            return None
    
    def _extract_frame_id(self, filename: str) -> Optional[int]:
        """从文件名中提取帧ID"""
        # 支持多种文件名格式
        patterns = [
            r'Frame_(\d+)',  # Frame_00001
            r'frame_(\d+)',  # frame_00001
            r'(\d+)',        # 00001
            r'texture_(\d+)', # texture_00001
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return int(match.group(1))
        
        return None
    
    def interpolate_texture(self, frame_start: int, frame_end: int, t: float) -> Optional[np.ndarray]:
        """在两个帧之间插值纹理"""
        try:
            # 根据插值参数t计算对应的帧ID
            target_frame = int(frame_start + t * (frame_end - frame_start))
            print(f"🎯 目标帧ID: {target_frame} (t={t:.3f}, 范围: {frame_start}-{frame_end})")
            
            # 查找目标帧的纹理
            target_texture = self._find_closest_texture(target_frame)
            
            if target_texture is None:
                print(f"⚠️  无法找到帧 {target_frame} 的纹理，尝试插值")
                
                # 如果找不到目标帧，尝试插值
                start_texture = self._find_closest_texture(frame_start)
                end_texture = self._find_closest_texture(frame_end)
                
                if start_texture is None and end_texture is None:
                    print(f"⚠️  无法找到帧 {frame_start} 和 {frame_end} 的纹理")
                    return None
                
                if start_texture is None:
                    print(f"⚠️  使用帧 {frame_end} 的纹理作为插值结果")
                    return end_texture
                
                if end_texture is None:
                    print(f"⚠️  使用帧 {frame_start} 的纹理作为插值结果")
                    return start_texture
                
                # 确保纹理尺寸一致
                if start_texture.shape != end_texture.shape:
                    print(f"⚠️  纹理尺寸不一致: {start_texture.shape} vs {end_texture.shape}")
                    # 调整尺寸
                    end_texture = cv2.resize(end_texture, (start_texture.shape[1], start_texture.shape[0]))
                
                # 线性插值纹理
                interpolated_texture = cv2.addWeighted(start_texture, 1 - t, end_texture, t, 0)
                print(f"✅ 纹理插值完成 (t={t:.3f})")
                
                return interpolated_texture
            else:
                print(f"✅ 直接使用帧 {target_frame} 的纹理")
                return target_texture
            
        except Exception as e:
            print(f"❌ 纹理插值失败: {e}")
            return None
    
    def _find_closest_texture(self, frame_id: int) -> Optional[np.ndarray]:
        """查找最接近的纹理文件"""
        try:
            print(f"🔍 查找帧 {frame_id} 的纹理...")
            
            # 首先尝试精确匹配
            for texture_file in self.texture_files:
                extracted_id = self._extract_frame_id(texture_file.name)
                if extracted_id == frame_id:
                    texture = cv2.imread(str(texture_file))
                    if texture is not None:
                        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                        print(f"✅ 找到精确匹配的纹理: {texture_file.name}")
                        return texture
            
            # 如果找不到精确匹配，找最接近的
            closest_file = None
            min_distance = float('inf')
            
            for texture_file in self.texture_files:
                extracted_id = self._extract_frame_id(texture_file.name)
                if extracted_id is not None:
                    distance = abs(extracted_id - frame_id)
                    if distance < min_distance:
                        min_distance = distance
                        closest_file = texture_file
            
            if closest_file is not None:
                texture = cv2.imread(str(closest_file))
                if texture is not None:
                    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
                    print(f"✅ 使用最接近的纹理: {closest_file.name} (距离: {min_distance})")
                    return texture
            
            print(f"❌ 未找到帧 {frame_id} 的纹理")
            return None
            
        except Exception as e:
            print(f"❌ 查找纹理失败: {e}")
            return None
    
    def process_interpolated_frame(self, mesh: o3d.geometry.TriangleMesh, 
                                 frame_idx: int, frame_start: int, frame_end: int, 
                                 t: float, output_dir: Path, save_standard_obj: bool = False,
                                 save_npy_files: bool = False) -> Dict[str, Any]:
        """处理单个插值帧"""
        result = {
            'mesh_file': None,
            'texture_file': None,
            'material_file': None,
            'standard_mesh_file': None
        }
        
        try:
            print(f"🎨 处理插值帧 {frame_idx} (t={t:.3f})")
            
            # 保存带颜色的mesh文件
            mesh_file = self.save_mesh_with_colors(mesh, output_dir, frame_idx)
            if mesh_file:
                result['mesh_file'] = mesh_file
            
            # 插值纹理
            interpolated_texture = self.interpolate_texture(frame_start, frame_end, t)
            if interpolated_texture is not None:
                # 保存纹理文件
                texture_file = self.save_texture(interpolated_texture, output_dir, frame_idx)
                if texture_file:
                    result['texture_file'] = texture_file
                    
                    # 创建材质文件
                    texture_filename = Path(texture_file).name
                    material_file = self.create_material_file(output_dir, frame_idx, texture_filename)
                    if material_file:
                        result['material_file'] = material_file
            
            # 保存标准obj文件（如果需要）
            if save_standard_obj:
                standard_mesh_file = self.save_standard_mesh(mesh, output_dir, frame_idx)
                if standard_mesh_file:
                    result['standard_mesh_file'] = standard_mesh_file
            
            print(f"✅ 帧 {frame_idx} 处理完成")
            return result
            
        except Exception as e:
            print(f"❌ 处理帧 {frame_idx} 失败: {e}")
            return result

def integrate_texture_processing(interpolator, mesh_folder_path: str, 
                               texture_folder_path: Optional[str] = None,
                               use_texture: bool = True, use_vertex_colors: bool = False):
    """
    将纹理处理集成到插值器中
    
    Args:
        interpolator: 插值器对象
        mesh_folder_path: 网格文件目录路径
        texture_folder_path: 纹理文件目录路径（可选）
        use_texture: 是否启用纹理处理
        use_vertex_colors: 是否启用顶点颜色
    """
    if not use_texture and not use_vertex_colors:
        print("⚠️  纹理处理和顶点颜色都未启用")
        return interpolator
    
    print("🎨 集成纹理处理到插值器...")
    
    # 创建纹理处理器
    texture_processor = TextureProcessor(mesh_folder_path, texture_folder_path)
    
    # 保存原始的插值方法
    original_method = interpolator.generate_interpolated_frames
    
    def enhanced_generate_interpolated_frames(frame_start, frame_end, num_interpolate, 
                                            max_optimize_frames=5, optimize_weights=True, 
                                            output_dir=None, debug_frames=None, smooth_mesh=False, 
                                            subdivide_iter=3, use_texture=False, use_vertex_colors=False,
                                            save_npy_files=False, save_standard_obj=True):
        """增强的插值帧生成方法"""
        
        # 检查是否已经有缓存的插值结果
        cache_key = f"{frame_start}_{frame_end}_{num_interpolate}"
        if hasattr(texture_processor, 'interpolation_cache') and cache_key in texture_processor.interpolation_cache:
            print(f"🔄 使用缓存的插值结果: {cache_key}")
            interpolated_frames = texture_processor.interpolation_cache[cache_key]
        else:
            # 调用原始的插值方法
            print(f"生成新的插值帧...")
            interpolated_frames = original_method(
                frame_start, frame_end, num_interpolate, max_optimize_frames, 
                optimize_weights, output_dir, debug_frames, smooth_mesh, subdivide_iter,
                save_npy_files=save_npy_files,  # 传递配置选项
                save_standard_obj=save_standard_obj  # 传递配置选项
            )
            
            # 缓存结果
            if not hasattr(texture_processor, 'interpolation_cache'):
                texture_processor.interpolation_cache = {}
            texture_processor.interpolation_cache[cache_key] = interpolated_frames
        
        # 如果启用了纹理处理，处理每个插值帧
        if use_texture and output_dir:
            print("开始纹理处理...")
            output_path = Path(output_dir)
            
            processed_frames = 0
            for i, frame_data in enumerate(interpolated_frames):
                if 'mesh' in frame_data and 'frame_idx' in frame_data:
                    mesh = frame_data['mesh']
                    frame_idx = frame_data['frame_idx']
                    
                    # 计算插值参数
                    t = i / (len(interpolated_frames) - 1) if len(interpolated_frames) > 1 else 0
                    
                    # 处理纹理
                    result = texture_processor.process_interpolated_frame(
                        mesh, frame_idx, frame_start, frame_end, t, output_path,
                        save_standard_obj=save_standard_obj, save_npy_files=save_npy_files
                    )
                    
                    # 更新frame_data
                    frame_data.update(result)
                    processed_frames += 1
            
            print(f"纹理处理完成: {processed_frames}/{len(interpolated_frames)} 帧成功")
        
        return interpolated_frames
    
    # 替换插值器的方法
    interpolator.generate_interpolated_frames = enhanced_generate_interpolated_frames
    
    print("✅ 纹理处理集成完成")
    return interpolator 