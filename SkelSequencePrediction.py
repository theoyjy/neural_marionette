import torch
import numpy as np
import os
import pickle
import open3d as o3d
from model.neural_marionette import NeuralMarionette
from utils.dataset_utils import crop_sequence, voxelize, episodic_normalization
import glob
from pathlib import Path
from SkelVisualizer import visualize_skeleton
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class SequenceSkeletonPredictor:
    def __init__(self, checkpoint_path, opt_path):
        """
        初始化Neural Marionette模型
        
        Args:
            checkpoint_path: 预训练模型路径
            opt_path: 配置文件路径
        """
        start_time = time.time()
        
        # 加载配置
        with open(opt_path, 'rb') as f:
            self.opt = pickle.load(f)
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path)
        self.network = NeuralMarionette(self.opt).cuda()
        self.network.load_state_dict(checkpoint)
        self.network.eval()
        self.network.anneal(1)  # 启用affinity提取
        
        model_load_time = time.time() - start_time
        print(f"✅ 模型加载成功，关键点数量: {self.opt.nkeypoints}")
        print(f"⏱️  模型加载耗时: {model_load_time:.2f}秒")
    
    def process_single_mesh(self, mesh_file, frame_idx, total_frames):
        """
        处理单个网格文件（用于多线程）
        
        Args:
            mesh_file: 网格文件路径
            frame_idx: 帧索引
            total_frames: 总帧数
            
        Returns:
            dict: 包含处理结果的字典
        """
        try:
            print(f"🔄 处理第 {frame_idx+1}/{total_frames} 个文件: {os.path.basename(mesh_file)}")
            
            # 加载网格
            if mesh_file.endswith('.obj') or mesh_file.endswith('.ply'):
                mesh = o3d.io.read_triangle_mesh(mesh_file)
            else:
                # 尝试作为点云加载
                pcd = o3d.io.read_point_cloud(mesh_file)
                points = np.asarray(pcd.points)
            
            if 'mesh' in locals() and len(mesh.vertices) > 0:
                points = np.asarray(mesh.vertices)
                mesh_data = mesh
            elif 'pcd' in locals() and len(pcd.points) > 0:
                points = np.asarray(pcd.points)
                mesh_data = pcd
            else:
                raise ValueError(f"无法加载文件: {mesh_file}")
            
            # 归一化点云（模仿原代码的处理方式）
            points_norm = episodic_normalization(points[None], scale=1.0, x_trans=0.0, z_trans=0.0)[0]
            
            # 体素化
            try:
                voxel = voxelize(points_norm, (self.opt.grid_size,) * 3, is_binarized=True)
            except Exception as e:
                print(f"❌ 体素化失败: {e}")
                raise
            
            return {
                'frame_idx': frame_idx,
                'mesh': mesh_data,
                'points_norm': points_norm,
                'voxel': voxel,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ 处理文件失败 {mesh_file}: {e}")
            return {
                'frame_idx': frame_idx,
                'mesh': None,
                'points_norm': None,
                'voxel': None,
                'success': False,
                'error': str(e)
            }
    
    def load_mesh_sequence(self, mesh_folder, file_pattern="*.obj", max_frames=None):
        """
        加载网格序列并转换为体素（多线程版本）
        
        Args:
            mesh_folder: 包含网格文件的文件夹路径
            file_pattern: 文件匹配模式，如 "*.obj", "frame_*.ply"
            max_frames: 最大帧数限制
        
        Returns:
            voxel_sequence: (T, grid_size, grid_size, grid_size)
            mesh_sequence: 原始网格数据列表
        """
        start_time = time.time()
        
        mesh_files = sorted(glob.glob(os.path.join(mesh_folder, file_pattern)))
        
        if max_frames:
            mesh_files = mesh_files[:max_frames]
        
        if len(mesh_files) == 0:
            raise ValueError(f"在 {mesh_folder} 中未找到匹配 {file_pattern} 的文件")
        
        print(f"📁 找到 {len(mesh_files)} 个网格文件")
        print(f"🔄 开始多线程处理...")
        
        # 多线程处理
        max_workers = min(8, len(mesh_files))  # 限制最大线程数
        print(f"  - 使用 {max_workers} 个线程")
        
        voxel_sequence = []
        mesh_sequence = []
        points_sequence = []
        
        # 用于存储结果的列表（按帧索引排序）
        results = [None] * len(mesh_files)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(self.process_single_mesh, mesh_file, i, len(mesh_files)): i 
                for i, mesh_file in enumerate(mesh_files)
            }
            
            # 收集结果
            for future in as_completed(future_to_idx):
                result = future.result()
                if result['success']:
                    results[result['frame_idx']] = result
                    print(f"✅ 完成第 {result['frame_idx']+1}/{len(mesh_files)} 个文件")
                else:
                    print(f"❌ 第 {result['frame_idx']+1} 个文件处理失败: {result.get('error', '未知错误')}")
        
        # 按顺序整理结果
        for i, result in enumerate(results):
            if result is None or not result['success']:
                raise ValueError(f"第 {i+1} 个文件处理失败")
            
            mesh_sequence.append(result['mesh'])
            points_sequence.append(result['points_norm'])
            voxel_sequence.append(result['voxel'])
        
        # 转换为torch tensor
        voxel_sequence = torch.from_numpy(np.stack(voxel_sequence, axis=0)).float().cuda()
        
        processing_time = time.time() - start_time
        print(f"✅ 多线程处理完成！")
        print(f"  - 体素序列形状: {voxel_sequence.shape}")
        print(f"  - 处理耗时: {processing_time:.2f}秒")
        print(f"  - 平均每帧: {processing_time/len(mesh_files):.3f}秒")
        
        return voxel_sequence, mesh_sequence, points_sequence

    def predict_skeleton_sequence(self, voxel_sequence):
        """
        预测整个序列的骨骼
        
        Args:
            voxel_sequence: (T, grid_size, grid_size, grid_size)
        
        Returns:
            keypoints: (1, T, K, 4) - joints 坐标和置信度
            transforms: (T, K, 4, 4) - 变换矩阵,每个关节的局部坐标系
            affinity: 骨骼连接关系
            parents: 父子关系
        """
        start_time = time.time()
        print(f"🧠 开始神经网络预测...")
        print(f"  - 输入形状: {voxel_sequence.shape}")
        
        with torch.no_grad():
            # 一次性处理整个序列
            detector_log = self.network.kypt_detector(voxel_sequence[None])  # 添加batch维度
            keypoints = detector_log['keypoints']
            affinity = detector_log['affinity']
            
            # 保持一致的可见性（类似原代码）
            keypoints[:, 1:, :, -1] = keypoints[:, :1, :, -1].expand(-1, voxel_sequence.size(0) - 1, -1)
            
            # 编码动力学
            dyna_log = self.network.dyna_module.encode(keypoints, affinity)
            R = dyna_log['R'][0]  # (T, K, 3, 3)
            
            # 获取结构信息
            A = self.network.dyna_module.A
            priority = self.network.dyna_module.priority
            parents = self.network.dyna_module.parents
            
            # 构建4x4变换矩阵
            pos = keypoints[0, :, :, :3][..., None]  # (T, K, 3, 1)
            T4x4 = torch.cat([R, pos], dim=-1)  # (T, K, 3, 4)
            homo = torch.tensor([0.0, 0.0, 0.0, 1.0]).to(R.device)[None, None, None].expand(
                R.size(0), R.size(1), -1, -1)
            T4x4 = torch.cat([T4x4, homo], dim=-2)  # (T, K, 4, 4)
            
            prediction_time = time.time() - start_time
            print(f"✅ 神经网络预测完成！")
            print(f"  - 预测耗时: {prediction_time:.2f}秒")
            print(f"  - 关键点形状: {keypoints.shape}")
            print(f"  - 变换矩阵形状: {T4x4.shape}")
            
            return {
                'keypoints': keypoints,
                'transforms': T4x4,
                'affinity': affinity,
                'parents': parents.cpu().numpy(),
                'priority_values': priority.values.cpu().numpy(),  # Extract values
                'priority_indices': priority.indices.cpu().numpy(),  # Extract indices
                'A': A,
                'rotations': R
            }
    
    def save_skeleton_results(self, results, output_dir, points_sequence=None):
        """
        保存骨骼预测结果
        
        Args:
            results: predict_skeleton_sequence的输出
            output_dir: 输出目录
            points_sequence: 原始点云序列（用于可视化）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存数值结果
        np.save(os.path.join(output_dir, 'keypoints.npy'), results['keypoints'][0].cpu().numpy())
        np.save(os.path.join(output_dir, 'transforms.npy'), results['transforms'].cpu().numpy())
        np.save(os.path.join(output_dir, 'parents.npy'), results['parents'])
        np.save(os.path.join(output_dir, 'affinity.npy'), results['affinity'].cpu().numpy())
        np.save(os.path.join(output_dir, 'priority_values.npy'), results['priority_values'])
        np.save(os.path.join(output_dir, 'priority_indices.npy'), results['priority_indices'])
        np.save(os.path.join(output_dir, 'A.npy'), results['A'].cpu().numpy())
        np.save(os.path.join(output_dir, 'rotations.npy'), results['rotations'].cpu().numpy())

        # save normalized points
        if points_sequence is not None:
            # np.save(os.path.join(output_dir, 'points_sequence.npy'), np.stack(points_sequence, axis=0))
            self.visualize_skeleton_sequence(results, output_dir, points_sequence)
    
    def visualize_skeleton_sequence(self, results, output_dir, points_sequence, 
                                  vis_threshold=0.2, save_frames=True):
        """
        可视化骨骼序列
        """
        keypoints = results['keypoints'][0].cpu().numpy()  # (T, K, 4)
        parents = results['parents']
        
        # 生成关节颜色
        np.random.seed(42)
        joint_colors = np.random.rand(keypoints.shape[1], 3)
        
        if save_frames:
            frames_dir = os.path.join(output_dir, 'skeleton_frames')
            os.makedirs(frames_dir, exist_ok=True)
        
        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600, visible=not save_frames)
        
        for t in range(keypoints.shape[0]):
            vis.clear_geometries()
            print(f'处理帧 {t+1}/{keypoints.shape[0]}')

            # 添加原始点云
            if points_sequence:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_sequence[t])
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                vis.add_geometry(pcd)
                # print(f'min={np.min(points_sequence[t], axis=0)}, max={np.max(points_sequence[t], axis=0)}')
            
            # 添加关节和骨骼
            kypts = keypoints[t, :, :3]
            alphas = keypoints[t, :, -1]
            print(f'joints num = {kypts.shape[0]} min={np.min(kypts, axis=0)}, max={np.max(kypts, axis=0)}')
            print(f'parents: {parents}')
            draw_count = 0
            for k in range(keypoints.shape[1]):
                if alphas[k] < vis_threshold:
                    continue
                draw_count += 1
                # 添加关节球
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
                sphere.translate(kypts[k])
                sphere.paint_uniform_color(joint_colors[k])
                vis.add_geometry(sphere)
                
                # 添加骨骼连接
                parent = parents[k]
                if parent != k and alphas[parent] >= vis_threshold:
                    # 创建骨骼线
                    line_points = [kypts[parent], kypts[k]]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(line_points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.paint_uniform_color([0, 0.8, 0])
                    vis.add_geometry(line_set)

            print(f'绘制关节数量: {draw_count}')
            if save_frames:
                # 保存帧图像
                img = vis.capture_screen_float_buffer(True)
                img = (np.asarray(img) * 255).astype(np.uint8)
                o3d.io.write_image(os.path.join(frames_dir, f'frame_{t:04d}.png'), 
                                 o3d.geometry.Image(img))
            else:
                # 交互式显示
                vis.poll_events()
                vis.update_renderer()
        
        vis.destroy_window()
        print(f"可视化完成，共处理 {keypoints.shape[0]} 帧")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="骨骼序列预测")
    parser.add_argument("--mesh_folder", type=str, default="D:/Code/VVEditor/Rafa_Approves_hd_4k", 
                       help="输入网格文件夹路径")
    parser.add_argument("--output_dir", type=str, default="output/skeleton_prediction", 
                       help="输出目录")
    parser.add_argument("--max_frames", type=int, default=160, 
                       help="最大处理帧数")
    parser.add_argument("--visualization", action="store_true", 
                       help="启用可视化")
    
    args = parser.parse_args()
    
    # 配置路径
    exp_dir = 'pretrained/aist'
    checkpoint_path = os.path.join(exp_dir, 'aist_pretrained.pth')
    opt_path = os.path.join(exp_dir, 'opt.pickle')
    
    # 输入序列文件夹
    mesh_folder = args.mesh_folder
    skel_data_dir = args.output_dir
    visualize_dir = os.path.join(args.output_dir, 'visualization')

    # 创建预测器
    predictor = SequenceSkeletonPredictor(checkpoint_path, opt_path)
    
    # 加载网格序列
    voxel_sequence, mesh_sequence, points_sequence = predictor.load_mesh_sequence(
        mesh_folder, file_pattern="*.obj", max_frames=args.max_frames
    )
    
    # 预测骨骼
    print("开始预测骨骼...")
    results = predictor.predict_skeleton_sequence(voxel_sequence)
    print("骨骼预测完成!")
    
    # 保存结果
    predictor.save_skeleton_results(results, skel_data_dir, points_sequence)

    if args.visualization:
        from SkelVisualizer import visualize_skeleton
        visualize_skeleton(skel_data_dir, visualize_dir)

    print("处理完成!")

if __name__ == "__main__":
    main()