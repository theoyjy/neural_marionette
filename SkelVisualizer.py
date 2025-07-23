import numpy as np
import os
from pathlib import Path
import trimesh
import pygltflib
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Accessor, BufferView, Buffer
from pygltflib import Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler
import struct
import json
import open3d as o3d
import imageio
import cv2

class SkeletonGLBVisualizer:
    def __init__(self):
        """初始化GLB可视化器"""
        pass
    
    def create_joint_geometry(self, radius=0.02):
        """创建关节的球体几何体"""
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
        return sphere.vertices, sphere.faces
    
    def create_bone_geometry(self, start_pos, end_pos, radius=0.01):
        """创建骨骼的圆柱体几何体"""
        # 计算骨骼方向和长度
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        if length < 1e-6:  # 避免长度为0的骨骼
            return np.array([]), np.array([])
        
        # 创建圆柱体
        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
        
        # 计算旋转矩阵将圆柱体对齐到骨骼方向
        up = np.array([0, 0, 1])
        direction_norm = direction / length
        
        # 如果方向向量与up向量平行，使用不同的参考向量
        if abs(np.dot(direction_norm, up)) > 0.99:
            up = np.array([1, 0, 0])
        
        # 计算旋转矩阵
        right = np.cross(direction_norm, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction_norm)
        
        rotation_matrix = np.column_stack([right, up, direction_norm])
        
        # 应用变换
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = start_pos + direction * 0.5  # 圆柱体中心位置
        
        cylinder.apply_transform(transform)
        
        return cylinder.vertices, cylinder.faces
    
    def load_skeleton_data(self, data_dir):
        """加载保存的骨骼数据"""
        keypoints = np.load(os.path.join(data_dir, 'keypoints.npy'))  # (T, K, 4)
        transforms = np.load(os.path.join(data_dir, 'transforms.npy'))  # (T, K, 4, 4)
        parents = np.load(os.path.join(data_dir, 'parents.npy'))  # (K,)
        
        return {
            'keypoints': keypoints,
            'transforms': transforms,
            'parents': parents,
            'num_frames': keypoints.shape[0],
            'num_joints': keypoints.shape[1]
        }
    
    def create_skeleton_mesh(self, keypoints_frame, parents, joint_radius=0.02, bone_radius=0.008):
        """为单帧创建骨骼网格"""
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        # 生成关节颜色
        np.random.seed(42)
        joint_colors = np.random.rand(len(keypoints_frame), 3)
        all_colors = []
        
        # 创建关节球体
        for i, joints in enumerate(keypoints_frame):
            joint_pos = joints[:3]
            alpha = joints[-1]  # 置信度
            if alpha < 0.2:  # 跳过不可见的关节
                continue
                
            vertices, faces = self.create_joint_geometry(joint_radius)
            vertices = vertices + joint_pos[:3]  # 移动到关节位置
            
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            
            # 添加颜色
            joint_color = joint_colors[i]
            colors = np.tile(joint_color, (len(vertices), 1))
            all_colors.append(colors)
            
            vertex_offset += len(vertices)
        
        # 创建骨骼圆柱体
        for child_idx, parent_idx in enumerate(parents):
            if parent_idx == child_idx:  # 根关节
                continue
            
            child_alpha = keypoints_frame[child_idx, -1]
            parent_alpha = keypoints_frame[parent_idx, -1]
            
            if child_alpha < 0.2 or parent_alpha < 0.2:  # 跳过不可见的关节
                continue
            
            child_pos = keypoints_frame[child_idx, :3]
            parent_pos = keypoints_frame[parent_idx, :3]
            
            vertices, faces = self.create_bone_geometry(parent_pos, child_pos, bone_radius)
            
            if len(vertices) > 0:
                all_vertices.append(vertices)
                all_faces.append(faces + vertex_offset)
                
                # 骨骼使用绿色
                bone_color = np.array([0.2, 0.8, 0.2])
                colors = np.tile(bone_color, (len(vertices), 1))
                all_colors.append(colors)
                
                vertex_offset += len(vertices)
        
        if len(all_vertices) == 0:
            return None, None, None
        
        # 合并所有几何体
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)
        colors = np.vstack(all_colors)
        
        return vertices, faces, colors
    
    def create_animated_glb(self, skeleton_data, output_path, fps=30):
        """创建包含动画的GLB文件"""
        keypoints = skeleton_data['keypoints']
        parents = skeleton_data['parents']
        num_frames = skeleton_data['num_frames']
        
        print(f"创建动画GLB: {num_frames} 帧, {skeleton_data['num_joints']} 个关节")
        
        # 创建第一帧的网格作为基础
        vertices, faces, colors = self.create_skeleton_mesh(keypoints[0], parents)
        
        if vertices is None:
            raise ValueError("无法创建骨骼网格，请检查数据")
        
        # 创建trimesh对象
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        
        # 导出为GLB（使用trimesh的简单方法）
        # 注意：这会创建一个静态网格，不包含动画
        mesh.export(output_path)
        
        print(f"GLB文件已保存到: {output_path}")
        
        return output_path
    
    def create_frame_sequence_gif(self, skeleton_data, output_path, fps=10, image_size=(800, 600)):
        """创建骨骼动画GIF文件"""
        keypoints = skeleton_data['keypoints']
        parents = skeleton_data['parents']
        num_frames = skeleton_data['num_frames']
        print(f"创建骨骼动画GIF: {num_frames} 帧")
        
        VIS_THRESHOLD = 0.2
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=image_size[0], height=image_size[1], visible=False)
        
        # 创建保存图片的临时目录
        frame_dir = os.path.dirname(output_path)
        temp_frame_dir = os.path.join(frame_dir, 'frame_skeleton')
        os.makedirs(temp_frame_dir, exist_ok=True)
        
        frame_images = []
        
        # 设置视角 (可选，如果有source.json文件的话)
        ctr = vis.get_view_control()
        source_json_path = "data/demo/source/source.json"
        if os.path.exists(source_json_path):
            try:
                parameters = o3d.io.read_pinhole_camera_parameters(source_json_path)
                ctr.convert_from_pinhole_camera_parameters(parameters)
            except:
                # 如果读取失败，使用默认视角
                pass
        
        # 为每一帧生成骨骼可视化
        for t in range(num_frames):
            # 清除之前的几何体
            vis.clear_geometries()
            
            # 获取当前帧的关键点
            keypoint = keypoints[t]  # (K, 4)
            kypts = keypoint[..., :3]  # 位置坐标
            alphas = keypoint[..., -1].clip(0, 1)  # 置信度
            
            # 绘制关节球体和骨骼圆锥
            for k in range(keypoints.shape[1]):
                if alphas[k] < VIS_THRESHOLD:
                    continue
                    
                # 绘制关节球体
                vis.add_geometry(self.drawSphere(kypts[k], [0.7, 0.1, 0]))
                
                # 绘制到父关节的骨骼
                parent = parents[k]
                if alphas[parent] < VIS_THRESHOLD or k == parent:
                    continue
                    
                vis.add_geometry(self.drawCone1(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
                vis.add_geometry(self.drawCone2(kypts[parent], kypts[k], color=[0, 0.6, 0.1]))
            
            # 捕获当前帧图像
            cone_img = np.asarray(vis.capture_screen_float_buffer(True))
            frame_img = (cone_img * 255).astype(np.uint8)
            
            # 保存当前帧图片
            frame_path = os.path.join(temp_frame_dir, f'frame_{t:04d}.png')
            cv2.imwrite(frame_path, frame_img[..., ::-1])  # BGR格式保存
            frame_images.append(frame_img)
            
            print(f"处理帧 {t+1}/{num_frames}")
        
        # 关闭可视化窗口
        vis.destroy_window()
        
        # 创建GIF动画
        if len(frame_images) > 0:
            # 转换为RGB格式用于imageio
            rgb_images = [img[..., ::-1] for img in frame_images]  # 转换为RGB
            imageio.mimsave(output_path, rgb_images, fps=fps)
            print(f"GIF动画已保存到: {output_path}")
        else:
            print("警告：没有生成任何帧图像")
        
        return output_path
    
    def drawSphere(self, center, color=[0.6, 0.9, 0.6], radius=0.03):
        """创建球体几何"""
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.translate(center)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        return sphere
    
    def drawCone1(self, bottom_center, top_position, color=[0.6, 0.9, 0.6]):
        """创建圆锥几何1"""
        cone = o3d.geometry.TriangleMesh.create_cone(radius=0.03, height=np.linalg.norm(top_position - bottom_center)*0.8+1e-6, resolution=80)
        line1 = np.array([0.0, 0.0, 1.0])
        line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center)+1e-6)
        v = np.cross(line1, line2)
        c = np.dot(line1, line2) + 1e-8
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
        if np.abs(c + 1.0) < 1e-4:
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        margin = np.linalg.norm(top_position - bottom_center) * 0.2
        T = bottom_center + margin * line2
        cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
        cone.paint_uniform_color(color)
        cone.compute_vertex_normals()
        return cone
    
    def drawCone2(self, bottom_center, top_position, color=[0.6, 0.9, 0.6]):
        """创建圆锥几何2"""
        cone = o3d.geometry.TriangleMesh.create_cone(0.03, height=np.linalg.norm(top_position - bottom_center) * 0.2 + 1e-6, resolution=80)
        cone = cone.rotate(cone.get_rotation_matrix_from_xyz((np.pi, 0, 0)))
        line1 = np.array([0.0, 0.0, 1.0])
        line2 = (top_position - bottom_center) / (np.linalg.norm(top_position - bottom_center) + 1e-6)
        v = np.cross(line1, line2)
        c = np.dot(line1, line2) + 1e-8
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + k + np.matmul(k, k) * (1 / (1 + c))
        if np.abs(c + 1.0) < 1e-4:
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        margin = np.linalg.norm(top_position - bottom_center) * 0.195
        T = bottom_center + margin * line2
        cone.transform(np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
        cone.paint_uniform_color(color)
        cone.compute_vertex_normals()
        return cone
    
    def create_simple_animation_data(self, skeleton_data, output_path):
        """创建简单的动画数据文件（JSON格式）"""
        keypoints = skeleton_data['keypoints']
        parents = skeleton_data['parents']
        
        animation_data = {
            'fps': 30,
            'num_frames': skeleton_data['num_frames'],
            'num_joints': skeleton_data['num_joints'],
            'parents': parents.tolist(),
            'frames': []
        }
        
        for frame_idx in range(keypoints.shape[0]):
            frame_data = {
                'frame': frame_idx,
                'joints': []
            }
            
            for joint_idx in range(keypoints.shape[1]):
                joint_data = {
                    'index': joint_idx,
                    'position': keypoints[frame_idx, joint_idx, :3].tolist(),
                    'visibility': float(keypoints[frame_idx, joint_idx, 3])
                }
                frame_data['joints'].append(joint_data)
            
            animation_data['frames'].append(frame_data)
        
        # 保存JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(animation_data, f, indent=2, ensure_ascii=False)
        
        print(f"动画数据已保存到: {output_path}")
        return output_path

def visualize_skeleton(data_dir, output_dir, create_sequence=True, create_animation=True):
    """
    主函数：将保存的骨骼数据可视化为GLB文件
    
    Args:
        data_dir: 包含骨骼数据的目录（keypoints.npy, transforms.npy, parents.npy）
        output_dir: 输出目录
        create_sequence: 是否创建动画GIF文件
        create_animation: 是否创建动画数据文件
    """
    # 创建可视化器
    visualizer = SkeletonGLBVisualizer()
    
    # 加载数据
    print("加载骨骼数据...")
    skeleton_data = visualizer.load_skeleton_data(data_dir)
    print(f"数据加载完成: {skeleton_data['num_frames']} 帧, {skeleton_data['num_joints']} 个关节")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建第一帧的GLB作为预览
    first_frame_path = os.path.join(output_dir, 'skeleton_preview.glb')
    visualizer.create_animated_glb(skeleton_data, first_frame_path)
    
    results = {
        'preview': first_frame_path
    }
    
    # 创建动画GIF文件
    if create_sequence:
        gif_path = os.path.join(output_dir, 'skeleton_animation.gif')
        gif_file = visualizer.create_frame_sequence_gif(skeleton_data, gif_path, fps=10)
        results['gif'] = gif_file
    
    # 创建动画数据文件
    if create_animation:
        animation_path = os.path.join(output_dir, 'skeleton_animation.json')
        visualizer.create_simple_animation_data(skeleton_data, animation_path)
        results['animation_data'] = animation_path
    
    print("GLB可视化完成！")
    print(f"预览文件: {first_frame_path}")
    if create_sequence:
        print(f"GIF动画: {gif_path}")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 骨骼数据目录（包含keypoints.npy, transforms.npy, parents.npy）
    data_dir = 'output/skeleton_prediction'
    
    # 输出目录
    output_dir = 'output/skeleton_visualization'
    
    # 创建GLB可视化
    results = visualize_skeleton(
        data_dir=data_dir,
        output_dir=output_dir,
        create_sequence=True,    # 创建动画GIF文件
        create_animation=True    # 创建动画数据JSON
    )
    
    print("创建的文件:")
    for key, value in results.items():
        print(f"{key}: {value}")


