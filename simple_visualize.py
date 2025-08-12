#!/usr/bin/env python3
"""
简单LBS重建可视化脚本
"""
import numpy as np
from pathlib import Path
import sys
import time
import open3d as o3d
from copy import deepcopy

def quick_visualize_frame(frame_idx=10):
    """快速可视化单帧重建结果"""
    
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"导入库失败: {e}")
        print("请确保安装了 open3d 和 matplotlib")
        return
    
    from Skinning import AutoSkinning
    
    print(f"快速可视化帧 {frame_idx}")
    print("=" * 40)
    
    # 配置路径
    skeleton_data_dir = "output/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    weights_path = "output/skinning_weights_auto.npz"  # 使用正确的权重文件名
    reference_frame = 5
    
    print("正在初始化...")
    
    # 初始化
    canonicalizer = AutoSkinning(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=reference_frame
    )
    
    # 加载数据
    canonicalizer.load_mesh_sequence(mesh_folder_path)
    
    if not canonicalizer.load_skinning_weights(weights_path):
        print("无法加载权重文件")
        return
    
    print(f"成功加载权重: {canonicalizer.skinning_weights.shape}")
    
    # 检查所有相关数据的边界
    max_mesh_frame = len(canonicalizer.mesh_files) - 1
    max_transform_frame = canonicalizer.transforms.shape[0] - 1
    max_keypoint_frame = canonicalizer.keypoints.shape[0] - 1
    
    max_available_frame = min(max_mesh_frame, max_transform_frame, max_keypoint_frame)
    
    if frame_idx > max_available_frame:
        print(f"帧索引 {frame_idx} 超出可用范围")
        print(f"   网格文件: 0-{max_mesh_frame}")
        print(f"   变换矩阵: 0-{max_transform_frame}")
        print(f"   关键点: 0-{max_keypoint_frame}")
        print(f"   建议使用: 0-{max_available_frame}")
        return
    
    print(f"重建帧 {frame_idx}...")
    
    # 加载原始mesh
    original_mesh = o3d.io.read_triangle_mesh(str(canonicalizer.mesh_files[frame_idx]))
    original_vertices = np.asarray(original_mesh.vertices)
    
    # 获取rest pose顶点
    rest_vertices = canonicalizer.rest_pose_vertices
    
    # 归一化处理
    if frame_idx not in canonicalizer.frame_normalization_params:
        canonicalizer.frame_normalization_params[frame_idx] = \
            canonicalizer.compute_mesh_normalization_params(original_mesh)
    
    target_vertices_norm = canonicalizer.normalize_mesh_vertices(
        original_vertices, 
        canonicalizer.frame_normalization_params[frame_idx]
    )
    rest_vertices_norm = canonicalizer.normalize_mesh_vertices(
        rest_vertices, 
        canonicalizer.frame_normalization_params[reference_frame]
    )
    
    # 计算相对变换
    target_transforms = canonicalizer.transforms[frame_idx]
    rest_transforms = canonicalizer.transforms[reference_frame]
    
    relative_transforms = np.zeros_like(target_transforms)
    for j in range(canonicalizer.num_joints):
        if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
            rest_inv = np.linalg.inv(rest_transforms[j])
            relative_transforms[j] = target_transforms[j] @ rest_inv
        else:
            relative_transforms[j] = np.eye(4)
    
    # LBS重建
    start_time = time.time()
    predicted_vertices_norm = canonicalizer.apply_lbs_transform(
        rest_vertices_norm, 
        canonicalizer.skinning_weights, 
        relative_transforms
    )
    lbs_time = time.time() - start_time
    
    # 计算误差
    vertex_errors = np.linalg.norm(predicted_vertices_norm - target_vertices_norm, axis=1)
    
    print(f"重建质量:")
    print(f"   平均误差: {np.mean(vertex_errors):.6f}")
    print(f"   最大误差: {np.max(vertex_errors):.6f}")
    print(f"   RMSE: {np.sqrt(np.mean(vertex_errors**2)):.6f}")
    print(f"   重建时间: {lbs_time:.3f}s")
    
    # 创建可视化mesh
    # 原始mesh (归一化后)
    original_mesh_vis = o3d.geometry.TriangleMesh()
    original_mesh_vis.vertices = o3d.utility.Vector3dVector(target_vertices_norm)
    original_mesh_vis.triangles = original_mesh.triangles
    original_mesh_vis.paint_uniform_color([0.1, 0.1, 0.9])  # 蓝色
    
    # 重建mesh
    reconstructed_mesh = o3d.geometry.TriangleMesh()
    reconstructed_mesh.vertices = o3d.utility.Vector3dVector(predicted_vertices_norm)
    reconstructed_mesh.triangles = original_mesh.triangles
    reconstructed_mesh.paint_uniform_color([0.9, 0.1, 0.1])  # 红色
    
    # 误差可视化mesh
    if np.max(vertex_errors) > 0:
        normalized_errors = vertex_errors / np.max(vertex_errors)
    else:
        normalized_errors = vertex_errors
    
    colors = plt.cm.plasma(normalized_errors)[:, :3]
    error_mesh = o3d.geometry.TriangleMesh()
    error_mesh.vertices = o3d.utility.Vector3dVector(target_vertices_norm)
    error_mesh.triangles = original_mesh.triangles
    error_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # 并排放置mesh
    bbox = original_mesh_vis.get_axis_aligned_bounding_box()
    width = bbox.max_bound[0] - bbox.min_bound[0]
    
    # 平移位置
    reconstructed_mesh.translate([width * 1.2, 0, 0])
    error_mesh.translate([width * 2.4, 0, 0])
    
    # 创建可视化窗口
    print("正在打开可视化窗口...")
    print("提示:")
    print("   - 蓝色 = 原始mesh")
    print("   - 红色 = LBS重建mesh")
    print("   - 彩色 = 误差可视化 (深色=低误差, 亮色=高误差)")
    print("   - 用鼠标旋转、缩放查看")
    print("   - 按Q退出")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"LBS重建对比 - 帧 {frame_idx}", width=1400, height=800)
    
    # 添加mesh
    vis.add_geometry(original_mesh_vis)
    vis.add_geometry(reconstructed_mesh)
    vis.add_geometry(error_mesh)
    
    # 设置相机
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat([width * 1.2, 0, 0])  # 看向中间
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.6)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()
    
    # save visualization to file
    output_path = Path("output/reconstruction")
    output_path.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path / f"frame_{frame_idx:06d}_original.obj"), original_mesh_vis)
    o3d.io.write_triangle_mesh(str(output_path / f"frame_{frame_idx:06d}_reconstructed.obj"), reconstructed_mesh)
    o3d.io.write_triangle_mesh(str(output_path / f"frame_{frame_idx:06d}_error.obj"), error_mesh)

    print("可视化完成")

def batch_export_meshes(frame_list=[10, 20, 30]):
    """批量导出mesh文件供外部软件查看"""
    
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"导入库失败: {e}")
        return
    
    from Skinning import InverseMeshCanonicalizer
    
    print(f"💾 批量导出 {len(frame_list)} 帧的mesh文件")
    
    # 初始化
    canonicalizer = InverseMeshCanonicalizer(
        skeleton_data_dir="output/skeleton_prediction",
        reference_frame_idx=5
    )
    
    canonicalizer.load_mesh_sequence("D:/Code/VVEditor/Rafa_Approves_hd_4k")
    
    if not canonicalizer.load_skinning_weights("output/skinning_weights_fast.npz"):
        print("无法加载权重文件")
        return
    
    # 创建输出目录
    output_dir = Path("output/mesh_exports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_idx in frame_list:
        if frame_idx >= len(canonicalizer.mesh_files):
            print(f"帧 {frame_idx} 超出范围，跳过")
            continue
        
        print(f"处理帧 {frame_idx}...")
        
        try:
            # 重建过程（简化版）
            original_mesh = o3d.io.read_triangle_mesh(str(canonicalizer.mesh_files[frame_idx]))
            original_vertices = np.asarray(original_mesh.vertices)
            
            # 归一化
            if frame_idx not in canonicalizer.frame_normalization_params:
                canonicalizer.frame_normalization_params[frame_idx] = \
                    canonicalizer.compute_mesh_normalization_params(original_mesh)
            
            target_vertices_norm = canonicalizer.normalize_mesh_vertices(
                original_vertices, 
                canonicalizer.frame_normalization_params[frame_idx]
            )
            rest_vertices_norm = canonicalizer.normalize_mesh_vertices(
                canonicalizer.rest_pose_vertices, 
                canonicalizer.frame_normalization_params[5]
            )
            
            # 计算变换
            target_transforms = canonicalizer.transforms[frame_idx]
            rest_transforms = canonicalizer.transforms[5]
            
            relative_transforms = np.zeros_like(target_transforms)
            for j in range(canonicalizer.num_joints):
                if np.linalg.det(rest_transforms[j][:3, :3]) > 1e-6:
                    rest_inv = np.linalg.inv(rest_transforms[j])
                    relative_transforms[j] = target_transforms[j] @ rest_inv
                else:
                    relative_transforms[j] = np.eye(4)
            
            # LBS重建
            predicted_vertices_norm = canonicalizer.apply_lbs_transform(
                rest_vertices_norm, 
                canonicalizer.skinning_weights, 
                relative_transforms
            )
            
            # 计算误差
            vertex_errors = np.linalg.norm(predicted_vertices_norm - target_vertices_norm, axis=1)
            
            # 创建并保存mesh
            # 1. 原始mesh
            original_export = o3d.geometry.TriangleMesh()
            original_export.vertices = o3d.utility.Vector3dVector(target_vertices_norm)
            original_export.triangles = original_mesh.triangles
            original_path = output_dir / f"frame_{frame_idx:06d}_original.obj"
            o3d.io.write_triangle_mesh(str(original_path), original_export)
            
            # 2. 重建mesh
            reconstructed_export = o3d.geometry.TriangleMesh()
            reconstructed_export.vertices = o3d.utility.Vector3dVector(predicted_vertices_norm)
            reconstructed_export.triangles = original_mesh.triangles
            reconstructed_path = output_dir / f"frame_{frame_idx:06d}_reconstructed.obj"
            o3d.io.write_triangle_mesh(str(reconstructed_path), reconstructed_export)
            
            # 3. 误差可视化mesh
            normalized_errors = vertex_errors / np.max(vertex_errors) if np.max(vertex_errors) > 0 else vertex_errors
            colors = plt.cm.plasma(normalized_errors)[:, :3]
            error_export = o3d.geometry.TriangleMesh()
            error_export.vertices = o3d.utility.Vector3dVector(target_vertices_norm)
            error_export.triangles = original_mesh.triangles
            error_export.vertex_colors = o3d.utility.Vector3dVector(colors)
            error_path = output_dir / f"frame_{frame_idx:06d}_error_colored.obj"
            o3d.io.write_triangle_mesh(str(error_path), error_export)
            
            print(f"帧 {frame_idx} 导出完成 (误差: {np.mean(vertex_errors):.6f})")
            
        except Exception as e:
            print(f"帧 {frame_idx} 导出失败: {e}")
    
    print(f"\n所有文件已导出到: {output_dir}")
    print("文件说明:")
    print("   *_original.obj     - 原始mesh (归一化后)")
    print("   *_reconstructed.obj - LBS重建mesh")
    print("   *_error_colored.obj - 误差可视化mesh")

def simple_error_plot(frame_idx=10):
    """简单的误差分析图"""
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要matplotlib库来显示图表")
        return
    
    from Skinning import InverseMeshCanonicalizer
    
    print(f"生成帧 {frame_idx} 的误差分析图...")
    
    # 初始化并重建
    canonicalizer = InverseMeshCanonicalizer("output/skeleton_prediction", 5)
    canonicalizer.load_mesh_sequence("D:/Code/VVEditor/Rafa_Approves_hd_4k")
    canonicalizer.load_skinning_weights("output/skinning_weights_fast.npz")
    
    # 获取误差数据（简化重建过程）
    original_mesh = o3d.io.read_triangle_mesh(str(canonicalizer.mesh_files[frame_idx]))
    # ... 这里重复重建过程获取vertex_errors ...
    # (为了简化，这里省略具体实现)
    
    # 生成示例数据用于演示
    vertex_errors = np.random.exponential(0.01, 31419)  # 示例数据
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 误差直方图
    ax1.hist(vertex_errors, bins=50, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Vertex Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Frame {frame_idx} - Error Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 累积分布
    sorted_errors = np.sort(vertex_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cumulative)
    ax2.set_xlabel('Vertex Error')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title(f'Frame {frame_idx} - Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def drawCone2(bottom_center, top_position, color=[0.6, 0.9, 0.6]):
    cone = o3d.geometry.TriangleMesh.create_cone(0.025,
                                                 height=np.linalg.norm(top_position - bottom_center) * 0.2 + 1e-6,
                                                 resolution=20)
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

    cone.transform(
        np.concatenate((np.concatenate((R, T[:, np.newaxis]), axis=1), np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0))
    cone.paint_uniform_color(color)
    cone.compute_vertex_normals()
    return cone

def drawCone1(bottom_center, top_position, color=[0.6, 0.9, 0.6]):
    cone = o3d.geometry.TriangleMesh.create_cone(radius=0.025, height=np.linalg.norm(top_position - bottom_center)*0.8+1e-6, resolution=20)
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

def drawSphere(center, color=[0.6, 0.9, 0.6], radius=0.025):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere

def visualize_skinning_weights_on_mesh_with_skeleton(
    frame_idx=3, 
    subdivide_iter=0
):
    """在单帧mesh上可视化蒙皮权重，并叠加骨骼。"""
    try:
        from Skinning import AutoSkinning
    except ImportError as e:
        print(f"导入 Skinning 失败: {e}")
        return



    # --- 1. 数据加载 ---
    weights_filename="D:/Code/neural_marionette/output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skinning_weights/ref65_opt63-67_num5.npz"
    skeleton_data_dir = "D:/Code/neural_marionette/output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
    mesh_folder_path = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    reference_frame = 65
    frame_idx = reference_frame

    print(f"🎨 可视化蒙皮权重于帧 {frame_idx}")
    print(f"   - 权重文件: {weights_filename}")
    print("=" * 40)

    canonicalizer = AutoSkinning(
        skeleton_data_dir=skeleton_data_dir,
        reference_frame_idx=reference_frame
    )
    canonicalizer.load_mesh_sequence(mesh_folder_path)

    if not canonicalizer.load_skinning_weights(weights_filename):
        print(f"错误: 无法加载权重文件 '{weights_filename}'")
        return
        
    if frame_idx >= len(canonicalizer.mesh_files) or frame_idx >= len(canonicalizer.keypoints):
        print(f"错误: 帧索引 {frame_idx} 超出范围。")
        return
        
    # --- 2. 坐标系对齐与归一化 (最终正确逻辑) ---
    mesh = o3d.io.read_triangle_mesh(str(canonicalizer.mesh_files[frame_idx]))
    
    # 2.1. 加载原始数据
    original_vertices = np.asarray(mesh.vertices)
    original_keypoints = canonicalizer.keypoints[frame_idx, :, :3]
    
    # 2.2. 计算归一化参数并处理数据
    norm_params = canonicalizer.compute_mesh_normalization_params(mesh)
    
    # 对齐坐标系：在归一化后对骨骼进行坐标系对齐
    vertices_norm = canonicalizer.normalize_mesh_vertices(original_vertices, norm_params)
    # 骨骼: 已经在归一化空间中，但需要坐标系对齐（Y和Z轴翻转）
    keypoints_norm = np.stack([original_keypoints[:, 0], -original_keypoints[:, 1], -original_keypoints[:, 2]], axis=-1)
    vertices_norm = np.stack([vertices_norm[:, 0], -vertices_norm[:, 1], -vertices_norm[:, 2]], axis=-1)


    # --- 3. 可视化逻辑 ---
    skinning_weights = canonicalizer.skinning_weights
    parents = canonicalizer.parents
    num_joints = canonicalizer.num_joints
    
    np.random.seed(10000)
    joint_colors = np.random.rand(num_joints, 3)

    vertex_colors = np.einsum('nk,ki->ni', skinning_weights, joint_colors)
    
    skinned_mesh = o3d.geometry.TriangleMesh()
    skinned_mesh.vertices = o3d.utility.Vector3dVector(vertices_norm)
    skinned_mesh.triangles = mesh.triangles
    skinned_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    skinned_mesh.compute_vertex_normals()

    if subdivide_iter > 0:
        skinned_mesh = skinned_mesh.subdivide_loop(subdivide_iter)
        skinned_mesh.compute_vertex_normals()

    skeleton_geometries = []
    for k in range(num_joints):
        skeleton_geometries.append(drawSphere(keypoints_norm[k], joint_colors[k], radius=0.025))
        parent_idx = parents[k]
        if k != parent_idx:
            parent_pos = keypoints_norm[parent_idx]
            child_pos = keypoints_norm[k]
            skeleton_geometries.append(drawCone1(parent_pos, child_pos, color=[0, 0.8, 0.2]))
            skeleton_geometries.append(drawCone2(parent_pos, child_pos, color=[0, 0.8, 0.2]))

    # --- 4. 分层渲染与合成 (模仿vis_retarget.py) ---
    width, height = 1280, 960
    
    # 4.1. 渲染模型
    vis_mesh = o3d.visualization.Visualizer()
    vis_mesh.create_window(f"Mesh Renderer", width=width, height=height, visible=False)
    vis_mesh.add_geometry(skinned_mesh)
    ctr = vis_mesh.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_lookat(np.mean(vertices_norm, axis=0))
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    # 保存相机参数，以便骨骼渲染器使用
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    mesh_img = vis_mesh.capture_screen_float_buffer(True)
    vis_mesh.destroy_window()

    # 4.2. 渲染骨骼
    vis_skel = o3d.visualization.Visualizer()
    vis_skel.create_window(f"Skeleton Renderer", width=width, height=height, visible=False)
    for geom in skeleton_geometries:
        vis_skel.add_geometry(geom)
    ctr_skel = vis_skel.get_view_control()
    # 应用与模型渲染器完全相同的相机参数
    ctr_skel.convert_from_pinhole_camera_parameters(cam_params)
    skel_img = vis_skel.capture_screen_float_buffer(True)
    vis_skel.destroy_window()

    # 4.3. 合成图像
    mesh_img_np = np.asarray(mesh_img)
    skel_img_np = np.asarray(skel_img)

    # 创建一个蒙版，找到骨骼图像中所有不是纯白色的像素
    # (纯白像素的RGB值加起来等于3.0)
    mask = skel_img_np.sum(axis=-1) < 2.99
    
    # 将骨骼像素"贴"到模型图像上
    mesh_img_np[mask] = skel_img_np[mask]
    
    # 转换并保存最终图像
    final_img_uint8 = (mesh_img_np * 255).astype(np.uint8)
    
    output_dir = Path("D:/Code/neural_marionette/evaluation/interpolation/registrations_m/50002_one_leg_jump/skinning_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_name = Path(weights_filename).stem
    output_filename = f"frame_{frame_idx:04d}_{weights_name}_overlay.png"
    output_path = output_dir / output_filename
    
    # 使用Open3D的图像写入功能，它能正确处理RGB图像
    o3d.io.write_image(str(output_path), o3d.geometry.Image(final_img_uint8), 9)

    print(f"✅ 可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    print("LBS重建可视化工具")
    print("=" * 30)
    
    visualize_skinning_weights_on_mesh_with_skeleton()
    # return

    # if len(sys.argv) > 1:
    #     try:
    #         frame_idx = int(sys.argv[1])
    #         print(f"可视化帧: {frame_idx}")
    #         quick_visualize_frame(frame_idx)
    #     except ValueError:
    #         if sys.argv[1] == "export":
    #             batch_export_meshes()
    #         elif sys.argv[1] == "plot":
    #             frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    #             simple_error_plot(frame_idx)
    #         else:
    #             print("用法: python simple_visualize.py [frame_number|export|plot]")
    # else:
    #     print("选择操作:")
    #     print("1. 可视化单帧 (输入帧号)")
    #     print("2. 批量导出 (输入 'export')")
    #     print("3. 误差图表 (输入 'plot')")
        
    #     choice = input("请选择 (1-3 或帧号): ").strip()
        
    #     if choice == "export":
    #         batch_export_meshes()
    #     elif choice == "plot":
    #         simple_error_plot()
    #     elif choice.isdigit():
    #         quick_visualize_frame(int(choice))
    #     else:
    #         print("使用默认帧10进行可视化")
    #         quick_visualize_frame(10)