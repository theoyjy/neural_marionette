#!/usr/bin/env python3
"""
最终修正版骨骼插值工具
解决所有姿态不匹配问题，提供最高质量的插值

改进点:
1. 使用正确的rest_pose (实际模板帧)
2. 改进的骨骼变换验证和修正
3. 自适应插值质量控制
4. 便捷的命令行接口
"""

import os
import argparse
import pickle
import numpy as np
from corrected_skeletal_interpolator import CorrectedSkeletalInterpolator

class FinalInterpolationTool:
    def __init__(self, unified_data_path, optimized_weights_path):
        """初始化最终插值工具"""
        self.interpolator = CorrectedSkeletalInterpolator(unified_data_path, optimized_weights_path)
        self.max_frames = self.interpolator.unified_meshes.shape[0]
        
        print(f"🎬 修正插值工具已准备就绪!")
        print(f"   可用帧范围: 0 - {self.max_frames - 1}")
        print(f"   模板帧索引: {self.interpolator.template_frame_idx}")
        print(f"   总帧数: {self.max_frames}")
        
    def analyze_frame_quality(self, frame_indices=None):
        """分析帧的变换质量"""
        if frame_indices is None:
            # 分析所有关键帧
            frame_indices = list(range(0, self.max_frames, max(1, self.max_frames // 20)))
            
        print(f"\n🔍 分析 {len(frame_indices)} 个帧的变换质量:")
        
        quality_data = []
        for frame in frame_indices:
            if frame < self.max_frames:
                avg_error, max_error = self.interpolator.validate_transformation(frame)
                quality_data.append({
                    'frame': frame,
                    'avg_error': avg_error,
                    'max_error': max_error,
                    'quality': 'excellent' if avg_error < 0.02 else 'good' if avg_error < 0.05 else 'fair' if avg_error < 0.1 else 'poor'
                })
                print(f"   帧 {frame:3d}: 平均误差 {avg_error:.6f}, 最大误差 {max_error:.6f} ({quality_data[-1]['quality']})")
                
        # 统计质量分布
        quality_counts = {}
        for data in quality_data:
            quality = data['quality']
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
        print(f"\n📊 质量分布:")
        for quality, count in quality_counts.items():
            print(f"   {quality}: {count} 帧 ({count/len(quality_data)*100:.1f}%)")
            
        return quality_data
        
    def recommend_interpolation_pairs(self, min_quality='good', min_distance=10):
        """推荐适合插值的帧对"""
        print(f"\n🎯 推荐插值帧对 (最低质量: {min_quality}, 最小距离: {min_distance})")
        
        # 分析所有帧质量
        all_quality = []
        for frame in range(self.max_frames):
            avg_error, _ = self.interpolator.validate_transformation(frame)
            quality = 'excellent' if avg_error < 0.02 else 'good' if avg_error < 0.05 else 'fair' if avg_error < 0.1 else 'poor'
            all_quality.append({'frame': frame, 'avg_error': avg_error, 'quality': quality})
            
        # 筛选高质量帧
        quality_priority = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        min_priority = quality_priority[min_quality]
        
        good_frames = [data for data in all_quality if quality_priority[data['quality']] >= min_priority]
        good_frames.sort(key=lambda x: x['avg_error'])  # 按误差排序
        
        print(f"   找到 {len(good_frames)} 个 >= {min_quality} 质量的帧")
        
        # 推荐帧对
        recommendations = []
        for i, start_data in enumerate(good_frames):
            for j, end_data in enumerate(good_frames[i+1:], i+1):
                distance = abs(end_data['frame'] - start_data['frame'])
                if distance >= min_distance:
                    avg_quality = (start_data['avg_error'] + end_data['avg_error']) / 2
                    recommendations.append({
                        'start_frame': start_data['frame'],
                        'end_frame': end_data['frame'],
                        'distance': distance,
                        'avg_quality': avg_quality,
                        'start_error': start_data['avg_error'],
                        'end_error': end_data['avg_error']
                    })
                    
        # 按质量排序推荐
        recommendations.sort(key=lambda x: x['avg_quality'])
        
        print(f"\n🌟 推荐的插值帧对 (前10个):")
        for i, rec in enumerate(recommendations[:10]):
            print(f"   {i+1:2d}. 帧 {rec['start_frame']:3d} -> {rec['end_frame']:3d} "
                  f"(距离: {rec['distance']:2d}, 质量: {rec['avg_quality']:.6f})")
                  
        return recommendations[:10]
        
    def smart_interpolation(self, start_frame, end_frame, target_smoothness='high', auto_frames=True):
        """智能插值，自动选择最佳参数"""
        print(f"\n🧠 智能插值: {start_frame} -> {end_frame}")
        
        # 验证帧质量
        start_error, _ = self.interpolator.validate_transformation(start_frame)
        end_error, _ = self.interpolator.validate_transformation(end_frame)
        
        print(f"   起始帧质量: {start_error:.6f}")
        print(f"   结束帧质量: {end_error:.6f}")
        
        # 自动选择插值帧数
        if auto_frames:
            distance = abs(end_frame - start_frame)
            if target_smoothness == 'high':
                num_frames = min(distance * 2, 50)  # 高平滑度
            elif target_smoothness == 'medium':
                num_frames = min(distance, 30)     # 中等平滑度
            else:
                num_frames = min(distance // 2, 15)  # 低平滑度
                
            num_frames = max(num_frames, 5)  # 最少5帧
        else:
            num_frames = 15  # 默认值
            
        print(f"   自动选择插值帧数: {num_frames}")
        
        # 选择插值方法
        avg_error = (start_error + end_error) / 2
        method = 'slerp' if avg_error < 0.08 else 'linear'
        print(f"   自动选择插值方法: {method}")
        
        # 执行插值
        result = self.interpolator.interpolate_sequence(
            start_frame, end_frame, num_frames, method
        )
        
        return result, num_frames, method
        
    def batch_interpolation(self, frame_pairs, output_prefix="batch"):
        """批量插值处理"""
        print(f"\n🎭 批量插值: {len(frame_pairs)} 个片段")
        
        all_results = []
        for i, (start, end) in enumerate(frame_pairs):
            print(f"\n📍 处理片段 {i+1}/{len(frame_pairs)}: {start} -> {end}")
            
            result, num_frames, method = self.smart_interpolation(start, end)
            
            # 保存单个片段结果
            output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\final_interpolations"
            segment_name = f"{output_prefix}_segment_{i+1:02d}_{start}to{end}"
            
            results_path = self.interpolator.save_interpolation_results(
                result, output_dir, segment_name
            )
            
            all_results.append({
                'segment_id': i+1,
                'start_frame': start,
                'end_frame': end,
                'num_interpolated': num_frames,
                'method': method,
                'results_path': results_path,
                'validation_errors': (result['start_validation_error'], result['end_validation_error'])
            })
            
        return all_results
        
    def interactive_mode(self):
        """交互式模式"""
        print("\n🎮 进入修正版交互式插值模式")
        print("=" * 60)
        
        while True:
            print("\n📋 可用选项:")
            print("1. 分析帧质量")
            print("2. 推荐插值帧对")
            print("3. 智能单个插值")
            print("4. 批量插值")
            print("5. 退出")
            
            try:
                choice = input("\n请选择操作 (1-5): ").strip()
                
                if choice == '1':
                    self._interactive_analyze_quality()
                elif choice == '2':
                    self._interactive_recommend_pairs()
                elif choice == '3':
                    self._interactive_smart_interpolation()
                elif choice == '4':
                    self._interactive_batch()
                elif choice == '5':
                    print("👋 退出修正插值工具")
                    break
                else:
                    print("❌ 无效选择，请输入1-5")
                    
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出插值工具")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                
    def _interactive_analyze_quality(self):
        """交互式质量分析"""
        print(f"\n🔍 帧质量分析 (总帧数: {self.max_frames})")
        
        mode = input("分析模式 (all/range/specific) [默认: all]: ").strip().lower()
        
        if mode == 'range':
            start = int(input("起始帧: "))
            end = int(input("结束帧: "))
            step = int(input("步长 [默认: 1]: ") or "1")
            frames = list(range(start, min(end+1, self.max_frames), step))
        elif mode == 'specific':
            frames_input = input("指定帧 (用空格分隔): ")
            frames = [int(x) for x in frames_input.split()]
        else:
            # 分析关键帧
            frames = list(range(0, self.max_frames, max(1, self.max_frames // 20)))
            
        self.analyze_frame_quality(frames)
        
    def _interactive_recommend_pairs(self):
        """交互式推荐帧对"""
        print("\n🎯 推荐插值帧对")
        
        quality = input("最低质量要求 (excellent/good/fair/poor) [默认: good]: ").strip().lower()
        if quality not in ['excellent', 'good', 'fair', 'poor']:
            quality = 'good'
            
        distance = int(input("最小帧距离 [默认: 10]: ") or "10")
        
        recommendations = self.recommend_interpolation_pairs(quality, distance)
        
        if recommendations:
            use_rec = input("\n使用推荐进行插值? (y/n): ").strip().lower()
            if use_rec == 'y':
                idx = int(input("选择推荐编号 (1-10): ")) - 1
                if 0 <= idx < len(recommendations):
                    rec = recommendations[idx]
                    result, num_frames, method = self.smart_interpolation(
                        rec['start_frame'], rec['end_frame']
                    )
                    
                    output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\final_interpolations"
                    name = f"recommended_{rec['start_frame']}to{rec['end_frame']}"
                    results_path = self.interpolator.save_interpolation_results(result, output_dir, name)
                    print(f"✅ 插值完成: {results_path}")
                    
    def _interactive_smart_interpolation(self):
        """交互式智能插值"""
        print(f"\n🧠 智能插值 (可用帧: 0-{self.max_frames-1})")
        
        start_frame = int(input("起始帧: "))
        end_frame = int(input("结束帧: "))
        
        smoothness = input("目标平滑度 (high/medium/low) [默认: high]: ").strip().lower()
        if smoothness not in ['high', 'medium', 'low']:
            smoothness = 'high'
            
        output_name = input("输出名称 (可选): ").strip()
        if not output_name:
            output_name = f"smart_{start_frame}to{end_frame}"
            
        result, num_frames, method = self.smart_interpolation(start_frame, end_frame, smoothness)
        
        output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\final_interpolations"
        results_path = self.interpolator.save_interpolation_results(result, output_dir, output_name)
        
        print(f"✅ 智能插值完成: {results_path}")
        print(f"📊 使用了 {num_frames} 帧，方法: {method}")
        
    def _interactive_batch(self):
        """交互式批量处理"""
        print("\n🎭 批量插值")
        
        mode = input("输入模式 (manual/recommended) [默认: manual]: ").strip().lower()
        
        if mode == 'recommended':
            quality = input("最低质量 (excellent/good/fair/poor) [默认: good]: ").strip().lower() or 'good'
            distance = int(input("最小距离 [默认: 15]: ") or "15")
            
            recommendations = self.recommend_interpolation_pairs(quality, distance)
            if recommendations:
                num_pairs = int(input(f"选择前几个推荐 (1-{len(recommendations)}) [默认: 5]: ") or "5")
                frame_pairs = [(rec['start_frame'], rec['end_frame']) for rec in recommendations[:num_pairs]]
            else:
                print("❌ 没有找到合适的推荐")
                return
        else:
            num_segments = int(input("片段数: "))
            frame_pairs = []
            for i in range(num_segments):
                print(f"片段 {i+1}:")
                start = int(input("  起始帧: "))
                end = int(input("  结束帧: "))
                frame_pairs.append((start, end))
                
        output_prefix = input("输出前缀 [默认: batch]: ").strip() or "batch"
        
        results = self.batch_interpolation(frame_pairs, output_prefix)
        
        print(f"\n✅ 批量插值完成，处理了 {len(results)} 个片段")
        for result in results:
            errors = result['validation_errors']
            print(f"   片段 {result['segment_id']}: {result['start_frame']}->{result['end_frame']} "
                  f"({result['num_interpolated']}帧, 误差: {errors[0]:.3f}/{errors[1]:.3f})")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='最终修正版骨骼驱动插值工具')
    
    parser.add_argument('--unified-data', type=str, 
                      default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\advanced_unified_results.pkl",
                      help='统一数据路径')
    parser.add_argument('--weights-data', type=str,
                      default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembones_optimized\dembones_optimized_results.pkl",
                      help='优化权重数据路径')
    
    # 功能选项
    parser.add_argument('--analyze', action='store_true', help='分析帧质量')
    parser.add_argument('--recommend', action='store_true', help='推荐插值帧对')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    
    # 插值参数
    parser.add_argument('--start', type=int, help='起始帧')
    parser.add_argument('--end', type=int, help='结束帧')
    parser.add_argument('--smoothness', type=str, choices=['high', 'medium', 'low'], default='high', help='平滑度')
    parser.add_argument('--output', type=str, help='输出名称')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.unified_data):
        print(f"❌ 找不到统一数据文件: {args.unified_data}")
        return
    if not os.path.exists(args.weights_data):
        print(f"❌ 找不到权重数据文件: {args.weights_data}")
        return
        
    # 创建工具
    tool = FinalInterpolationTool(args.unified_data, args.weights_data)
    
    if args.interactive:
        # 交互模式
        tool.interactive_mode()
    elif args.analyze:
        # 分析模式
        tool.analyze_frame_quality()
    elif args.recommend:
        # 推荐模式
        tool.recommend_interpolation_pairs()
    elif args.start is not None and args.end is not None:
        # 智能插值
        result, num_frames, method = tool.smart_interpolation(
            args.start, args.end, args.smoothness
        )
        
        output_name = args.output or f"final_{args.start}to{args.end}"
        output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\final_interpolations"
        results_path = tool.interpolator.save_interpolation_results(result, output_dir, output_name)
        
        print(f"\n🎯 智能插值完成: {results_path}")
        print(f"📊 参数: {num_frames}帧, {method}方法, {args.smoothness}平滑度")
    else:
        # 显示帮助
        print("🎬 最终修正版骨骼驱动插值工具")
        print("\n使用示例:")
        print("1. 分析帧质量:")
        print("   python final_interpolation_tool.py --analyze")
        print("\n2. 推荐插值帧对:")
        print("   python final_interpolation_tool.py --recommend")
        print("\n3. 智能插值:")
        print("   python final_interpolation_tool.py --start 20 --end 80 --smoothness high")
        print("\n4. 交互模式:")
        print("   python final_interpolation_tool.py --interactive")

if __name__ == "__main__":
    main()
