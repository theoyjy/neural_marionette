#!/usr/bin/env python3
"""
高级骨骼插值工具 - 便捷的交互式插值
基于DemBones优化权重的极高质量插值工具

使用方法:
python advanced_interpolation_tool.py --start 10 --end 50 --frames 20 --method slerp
"""

import os
import argparse
import pickle
import numpy as np
from advanced_skeletal_interpolator import AdvancedSkeletalInterpolator

class InterpolationTool:
    def __init__(self, optimized_data_path):
        """初始化插值工具"""
        self.interpolator = AdvancedSkeletalInterpolator(optimized_data_path)
        self.max_frames = self.interpolator.unified_meshes.shape[0]
        
        print(f"🎬 插值工具已准备就绪!")
        print(f"   可用帧范围: 0 - {self.max_frames - 1}")
        print(f"   总帧数: {self.max_frames}")
        
    def single_interpolation(self, start_frame, end_frame, num_frames, method='slerp', output_name=None):
        """单个插值任务"""
        # 验证输入
        if not (0 <= start_frame < self.max_frames):
            raise ValueError(f"start_frame必须在0-{self.max_frames-1}范围内")
        if not (0 <= end_frame < self.max_frames):
            raise ValueError(f"end_frame必须在0-{self.max_frames-1}范围内")
        if start_frame == end_frame:
            raise ValueError("起始帧和结束帧不能相同")
        if num_frames < 1:
            raise ValueError("插值帧数必须大于0")
            
        print(f"\n🎯 执行插值任务:")
        print(f"   起始帧: {start_frame}")
        print(f"   结束帧: {end_frame}")
        print(f"   插值帧数: {num_frames}")
        print(f"   插值方法: {method}")
        
        # 执行插值
        result = self.interpolator.interpolate_sequence(
            start_frame, end_frame, num_frames, method
        )
        
        # 生成输出名称
        if output_name is None:
            output_name = f"interpolation_{start_frame}_to_{end_frame}_{num_frames}frames_{method}"
            
        # 保存结果
        output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\custom_interpolations"
        results_path = self.interpolator.save_interpolation_results(
            result, output_dir, output_name
        )
        
        return results_path
        
    def multi_segment_interpolation(self, frame_pairs, frames_per_segment, method='slerp', output_name=None):
        """多片段插值"""
        # 验证输入
        for start, end in frame_pairs:
            if not (0 <= start < self.max_frames):
                raise ValueError(f"帧 {start} 超出范围 0-{self.max_frames-1}")
            if not (0 <= end < self.max_frames):
                raise ValueError(f"帧 {end} 超出范围 0-{self.max_frames-1}")
            if start == end:
                raise ValueError(f"片段 ({start}, {end}) 起始帧和结束帧不能相同")
                
        print(f"\n🎭 执行多片段插值:")
        print(f"   片段数: {len(frame_pairs)}")
        print(f"   片段列表: {frame_pairs}")
        print(f"   每片段帧数: {frames_per_segment}")
        print(f"   插值方法: {method}")
        
        # 执行插值
        result = self.interpolator.interpolate_multiple_segments(
            frame_pairs, frames_per_segment, method
        )
        
        # 生成输出名称
        if output_name is None:
            segment_str = "_".join([f"{s}to{e}" for s, e in frame_pairs])
            output_name = f"multi_segment_{segment_str}_{frames_per_segment}each_{method}"
            
        # 保存结果
        output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\custom_interpolations"
        results_path = self.interpolator.save_interpolation_results(
            result, output_dir, output_name
        )
        
        return results_path
        
    def spline_interpolation(self, key_frames, total_frames, output_name=None):
        """样条曲线插值"""
        # 验证输入
        for frame in key_frames:
            if not (0 <= frame < self.max_frames):
                raise ValueError(f"关键帧 {frame} 超出范围 0-{self.max_frames-1}")
                
        if len(key_frames) < 2:
            raise ValueError("至少需要2个关键帧进行样条插值")
            
        if total_frames < len(key_frames):
            raise ValueError("总帧数不能少于关键帧数")
            
        print(f"\n🌊 执行样条曲线插值:")
        print(f"   关键帧: {key_frames}")
        print(f"   总帧数: {total_frames}")
        
        # 执行插值
        result = self.interpolator.smooth_interpolation_with_spline(
            key_frames, total_frames
        )
        
        # 生成输出名称
        if output_name is None:
            key_str = "_".join(map(str, key_frames))
            output_name = f"spline_keys_{key_str}_{total_frames}frames"
            
        # 保存结果
        output_dir = r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\custom_interpolations"
        results_path = self.interpolator.save_interpolation_results(
            result, output_dir, output_name
        )
        
        return results_path
        
    def interactive_mode(self):
        """交互式模式"""
        print("\n🎮 进入交互式插值模式")
        print("=" * 50)
        
        while True:
            print("\n📋 可用选项:")
            print("1. 单个插值 (两帧之间)")
            print("2. 多片段插值")
            print("3. 样条曲线插值")
            print("4. 退出")
            
            try:
                choice = input("\n请选择操作 (1-4): ").strip()
                
                if choice == '1':
                    self._interactive_single()
                elif choice == '2':
                    self._interactive_multi_segment()
                elif choice == '3':
                    self._interactive_spline()
                elif choice == '4':
                    print("👋 退出插值工具")
                    break
                else:
                    print("❌ 无效选择，请输入1-4")
                    
            except KeyboardInterrupt:
                print("\n👋 用户中断，退出插值工具")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
                
    def _interactive_single(self):
        """交互式单个插值"""
        print(f"\n🎯 单个插值 (可用帧: 0-{self.max_frames-1})")
        
        start_frame = int(input("起始帧: "))
        end_frame = int(input("结束帧: "))
        num_frames = int(input("插值帧数: "))
        
        method = input("插值方法 (slerp/linear) [默认: slerp]: ").strip()
        if not method:
            method = 'slerp'
            
        output_name = input("输出名称 (可选): ").strip()
        if not output_name:
            output_name = None
            
        results_path = self.single_interpolation(
            start_frame, end_frame, num_frames, method, output_name
        )
        
        print(f"✅ 插值完成: {results_path}")
        
    def _interactive_multi_segment(self):
        """交互式多片段插值"""
        print(f"\n🎭 多片段插值 (可用帧: 0-{self.max_frames-1})")
        
        num_segments = int(input("片段数: "))
        frame_pairs = []
        
        for i in range(num_segments):
            print(f"片段 {i+1}:")
            start = int(input(f"  起始帧: "))
            end = int(input(f"  结束帧: "))
            frame_pairs.append((start, end))
            
        frames_per_segment = int(input("每片段插值帧数: "))
        
        method = input("插值方法 (slerp/linear) [默认: slerp]: ").strip()
        if not method:
            method = 'slerp'
            
        output_name = input("输出名称 (可选): ").strip()
        if not output_name:
            output_name = None
            
        results_path = self.multi_segment_interpolation(
            frame_pairs, frames_per_segment, method, output_name
        )
        
        print(f"✅ 多片段插值完成: {results_path}")
        
    def _interactive_spline(self):
        """交互式样条插值"""
        print(f"\n🌊 样条曲线插值 (可用帧: 0-{self.max_frames-1})")
        
        key_frames_input = input("关键帧 (用空格分隔): ").strip()
        key_frames = [int(x) for x in key_frames_input.split()]
        
        total_frames = int(input("总输出帧数: "))
        
        output_name = input("输出名称 (可选): ").strip()
        if not output_name:
            output_name = None
            
        results_path = self.spline_interpolation(
            key_frames, total_frames, output_name
        )
        
        print(f"✅ 样条插值完成: {results_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='高级骨骼驱动插值工具')
    
    parser.add_argument('--data', type=str, 
                      default=r"D:\Code\VVEditor\Rafa_Approves_hd_4k\generated_skeletons\dembones_optimized\dembones_optimized_results.pkl",
                      help='DemBones优化数据路径')
    
    # 单个插值参数
    parser.add_argument('--start', type=int, help='起始帧')
    parser.add_argument('--end', type=int, help='结束帧')
    parser.add_argument('--frames', type=int, help='插值帧数')
    parser.add_argument('--method', type=str, default='slerp', choices=['slerp', 'linear'], help='插值方法')
    parser.add_argument('--output', type=str, help='输出名称')
    
    # 多片段插值参数
    parser.add_argument('--segments', type=str, help='片段列表，格式: "0,10;20,30;40,50"')
    parser.add_argument('--segment-frames', type=int, help='每片段插值帧数')
    
    # 样条插值参数
    parser.add_argument('--key-frames', type=str, help='关键帧列表，格式: "0,10,20,30"')
    parser.add_argument('--total-frames', type=int, help='样条插值总帧数')
    
    # 交互模式
    parser.add_argument('--interactive', action='store_true', help='启动交互式模式')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.data):
        print(f"❌ 找不到数据文件: {args.data}")
        return
        
    # 创建工具
    tool = InterpolationTool(args.data)
    
    if args.interactive:
        # 交互模式
        tool.interactive_mode()
    elif args.start is not None and args.end is not None and args.frames is not None:
        # 单个插值
        results_path = tool.single_interpolation(
            args.start, args.end, args.frames, args.method, args.output
        )
        print(f"\n🎯 插值完成: {results_path}")
    elif args.segments and args.segment_frames:
        # 多片段插值
        segments = []
        for segment in args.segments.split(';'):
            start, end = map(int, segment.split(','))
            segments.append((start, end))
            
        results_path = tool.multi_segment_interpolation(
            segments, args.segment_frames, args.method, args.output
        )
        print(f"\n🎭 多片段插值完成: {results_path}")
    elif args.key_frames and args.total_frames:
        # 样条插值
        key_frames = [int(x) for x in args.key_frames.split(',')]
        results_path = tool.spline_interpolation(
            key_frames, args.total_frames, args.output
        )
        print(f"\n🌊 样条插值完成: {results_path}")
    else:
        # 显示帮助
        print("🎬 高级骨骼驱动插值工具")
        print("\n使用示例:")
        print("1. 单个插值:")
        print("   python advanced_interpolation_tool.py --start 10 --end 50 --frames 20")
        print("\n2. 多片段插值:")
        print("   python advanced_interpolation_tool.py --segments \"0,10;20,30;40,50\" --segment-frames 5")
        print("\n3. 样条插值:")
        print("   python advanced_interpolation_tool.py --key-frames \"0,20,40,60\" --total-frames 50")
        print("\n4. 交互模式:")
        print("   python advanced_interpolation_tool.py --interactive")

if __name__ == "__main__":
    main()
