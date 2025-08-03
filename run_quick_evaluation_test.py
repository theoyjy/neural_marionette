#!/usr/bin/env python3
"""
快速评估测试脚本
测试修复后的评估系统是否能正常工作
"""

import subprocess
import sys
from pathlib import Path

def run_quick_test():
    """运行快速评估测试"""
    print("=== 快速评估测试 ===")
    print("这将测试评估系统是否能正常工作")
    print()
    
    # 测试参数
    cmd = [
        sys.executable, "evaluation/evaluate_interpolation.py",
        "--pairs_dir", "evaluation/data/dfaust/keyframe_pairs/registrations_m/50002_chicken_wings",
        "--results_dir", "evaluation/interpolation", 
        "--methods", "baseline", "dual_reference",
        "--output_dir", "evaluation/results/quick_test",
        "--database_name", "registrations_m",
        "--subject_id", "50002",
        "--sequence_id", "chicken_wings",
        "--k", "5",
        "--fast",
        "--single-process"  # 使用单进程避免pickle问题
    ]
    
    print("运行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        # 运行评估
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 评估成功完成!")
            print()
            print("输出:", result.stdout)
            
            # 检查生成的文件
            output_dir = Path("evaluation/results/quick_test")
            files_to_check = [
                "results.csv",
                "evaluation_report.md"
            ]
            
            print("检查生成的文件:")
            for filename in files_to_check:
                file_path = output_dir / filename
                if file_path.exists():
                    print(f"  ✅ {filename} - 已生成")
                else:
                    print(f"  ❌ {filename} - 未生成")
            
            return True
            
        else:
            print("❌ 评估失败")
            print("错误输出:", result.stderr)
            print("标准输出:", result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 评估超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        return False

def main():
    """主函数"""
    print("快速评估系统测试")
    print("=" * 50)
    
    # 检查前置条件
    pairs_dir = Path("evaluation/data/dfaust/keyframe_pairs/registrations_m/50002_chicken_wings/k5")
    if not pairs_dir.exists():
        print(f"❌ 关键帧对目录不存在: {pairs_dir}")
        print("请确保已生成关键帧对数据")
        return False
    
    interp_dir = Path("evaluation/interpolation/registrations_m/50002_chicken_wings_k5")
    if not interp_dir.exists():
        print(f"❌ 插值结果目录不存在: {interp_dir}")
        print("请确保已运行插值生成")
        return False
    
    print("✅ 前置条件检查通过")
    print()
    
    # 运行测试
    success = run_quick_test()
    
    print()
    print("=" * 50)
    if success:
        print("🎉 测试成功！评估系统工作正常")
        print()
        print("你现在可以:")
        print("1. 查看结果: evaluation/results/quick_test/")
        print("2. 运行完整流水线: python evaluation/run_evaluation_pipeline.py")
        print("3. 使用多进程模式: python evaluation/evaluate_interpolation.py --fast")
    else:
        print("😞 测试失败，请检查错误信息")
        print()
        print("故障排除建议:")
        print("1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("2. 确保已生成插值结果")
        print("3. 检查Python环境配置")
    
    return success

if __name__ == "__main__":
    main()