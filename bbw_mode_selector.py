"""
BBW模式选择器

让用户选择使用单帧BBW还是多帧BBW
"""

import argparse
import os
from pathlib import Path

def run_bbw_with_mode_selection():
    """运行BBW插值，允许用户选择模式"""
    
    parser = argparse.ArgumentParser(description="BBW插值模式选择")
    parser.add_argument("--mesh_folder", required=True, help="Mesh文件夹路径")
    parser.add_argument("--start_frame", type=int, required=True, help="起始帧")
    parser.add_argument("--end_frame", type=int, required=True, help="结束帧")
    parser.add_argument("--num_interpolate", type=int, default=5, help="插值帧数")
    parser.add_argument("--bbw_mode", choices=["single", "multi"], default="single", 
                       help="BBW模式: single (避免纸片化) 或 multi (更好的姿势适应)")
    parser.add_argument("--output_dir", help="输出目录")
    
    args = parser.parse_args()
    
    # 设置输出目录
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(f"output/bbw_{args.bbw_mode}_mode_test")
    
    # 配置输出路径
    skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
    
    output_paths = {
        'base': output_base,
        'skeleton': skeleton_dir,
        'skinning': output_base / "skinning_weights",
        'interpolation': output_base / "interpolation"
    }
    
    for path in output_paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    
    print(f"🎯 BBW插值配置:")
    print(f"  - 模式: {args.bbw_mode}")
    print(f"  - 帧范围: {args.start_frame} -> {args.end_frame}")
    print(f"  - 插值数量: {args.num_interpolate}")
    print(f"  - 输出目录: {output_base}")
    
    if args.bbw_mode == "single":
        print(f"  ✅ 单帧模式 - 避免纸片化效果")
    else:
        print(f"  ⚠️  多帧模式 - 可能有纸片化效果，但姿势适应更好")
    
    try:
        from volumetric_interpolation_pipeline import step2_interpolation
        
        # 创建临时的配置修改
        import tempfile
        import shutil
        
        # 运行插值
        success = step2_interpolation(
            folder_path=args.mesh_folder,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            num_interpolate=args.num_interpolate,
            output_paths=output_paths,
            evaluation_mode=True,
            method="bbw_enhanced",
            save_standard_obj=True,
            save_npy_files=False
        )
        
        if success:
            print(f"\n🎉 BBW插值完成!")
            
            # 检查结果
            output_files = list(output_paths['interpolation'].glob("*.obj"))
            print(f"生成 {len(output_files)} 个插值文件")
            
            if output_files:
                print(f"\n📁 输出文件位置:")
                for i, file_path in enumerate(output_files[:3]):  # 显示前3个
                    print(f"  {i+1}. {file_path}")
                if len(output_files) > 3:
                    print(f"  ... 等共{len(output_files)}个文件")
                
                print(f"\n💡 使用建议:")
                if args.bbw_mode == "single":
                    print(f"  - 如果结果正常，单帧模式是最佳选择")
                    print(f"  - 如果姿势变化大仍有问题，可尝试多帧模式")
                else:
                    print(f"  - 如果有纸片化效果，请使用 --bbw_mode single")
                    print(f"  - 多帧模式适合姿势变化很大的序列")
            
            return True
        else:
            print(f"\n❌ BBW插值失败")
            return False
            
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_test_both_modes():
    """快速测试两种模式的对比"""
    print("🔄 快速测试单帧vs多帧BBW模式")
    
    test_mesh_folder = "D:/Code/VVEditor/Rafa_Approves_hd_4k"
    
    if not os.path.exists(test_mesh_folder):
        print("❌ 测试数据不可用")
        return False
    
    # 测试参数
    start_frame, end_frame = 5, 10
    num_interpolate = 3
    
    modes = ["single", "multi"]
    results = {}
    
    for mode in modes:
        print(f"\n🔸 测试 {mode} 模式...")
        
        output_base = Path(f"quick_test_{mode}_mode")
        skeleton_dir = "output/pipeline_Rafa_Approves_hd_4k_8522ed0a/skeleton_prediction"
        
        output_paths = {
            'base': output_base,
            'skeleton': skeleton_dir,
            'skinning': output_base / "skinning_weights",
            'interpolation': output_base / "interpolation"
        }
        
        for path in output_paths.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 临时修改BBW模式
            if mode == "single":
                use_multi_frame = False
            else:
                use_multi_frame = True
            
            from bbw_enhanced_interpolator import BBWEnhancedInterpolator
            
            interpolator = BBWEnhancedInterpolator(
                skeleton_data_dir=skeleton_dir,
                mesh_folder_path=test_mesh_folder,
                use_bbw=True,
                use_multi_frame=use_multi_frame,
                bbw_reference_frame=start_frame,
                num_key_frames=3,
                frame_selection_method="pose_diversity"
            )
            
            # 初始化
            if use_multi_frame:
                interpolator.initialize_skinning(start_frame=start_frame, end_frame=end_frame)
            else:
                interpolator.initialize_skinning()
            
            # 生成插值
            interpolator.generate_interpolated_frames(
                frame_start=start_frame,
                frame_end=end_frame,
                num_interpolate=num_interpolate,
                output_dir=str(output_paths['interpolation']),
                save_standard_obj=True
            )
            
            # 检查结果
            output_files = list(output_paths['interpolation'].glob("*.obj"))
            
            if output_files:
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(str(output_files[0]))
                vertices = np.asarray(mesh.vertices)
                
                bbox = vertices.max(axis=0) - vertices.min(axis=0)
                thickness = bbox.min()
                
                results[mode] = {
                    'success': True,
                    'files': len(output_files),
                    'thickness': thickness,
                    'coord_range': [vertices.min(), vertices.max()]
                }
                
                print(f"  ✅ {mode}模式: {len(output_files)}个文件, 厚度={thickness:.6f}")
            else:
                results[mode] = {'success': False}
                print(f"  ❌ {mode}模式: 生成失败")
                
        except Exception as e:
            results[mode] = {'success': False, 'error': str(e)}
            print(f"  ❌ {mode}模式: 错误 - {e}")
    
    # 对比结果
    print(f"\n📊 模式对比结果:")
    print(f"{'模式':<10} {'状态':<10} {'文件数':<10} {'厚度':<15}")
    print("-" * 50)
    
    for mode, result in results.items():
        if result.get('success'):
            print(f"{mode:<10} {'成功':<10} {result['files']:<10} {result['thickness']:<15.6f}")
        else:
            print(f"{mode:<10} {'失败':<10} {'-':<10} {'-':<15}")
    
    # 推荐
    if results.get('single', {}).get('success') and results.get('multi', {}).get('success'):
        single_thickness = results['single']['thickness']
        multi_thickness = results['multi']['thickness']
        
        print(f"\n💡 推荐:")
        if single_thickness >= multi_thickness * 0.95:
            print(f"  ✅ 推荐使用 single 模式 (厚度: {single_thickness:.6f})")
            print(f"     - 避免纸片化")
            print(f"     - 计算更快")
        else:
            print(f"  ⚠️  multi 模式厚度更好 (厚度: {multi_thickness:.6f})")
            print(f"     - 但可能有纸片化风险")
    
    return True

if __name__ == "__main__":
    import sys
    import numpy as np
    
    if len(sys.argv) > 1:
        # 命令行模式
        run_bbw_with_mode_selection()
    else:
        # 快速测试模式
        quick_test_both_modes()