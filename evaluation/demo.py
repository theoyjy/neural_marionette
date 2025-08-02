#!/usr/bin/env python3
"""
评估系统演示脚本
展示完整的六步评估流程
"""

import os
import sys
import time
from pathlib import Path

def print_banner():
    """打印系统横幅"""
    print("=" * 60)
    print("🎯 插值评估系统演示")
    print("=" * 60)
    print("本系统实现了完整的插值方法评估流水线")
    print("支持对比baseline与dual_reference两种插值方法")
    print("=" * 60)

def check_system_status():
    """检查系统状态"""
    print("🔍 检查系统状态...")
    
    # 检查Python包
    required_packages = ['numpy', 'pandas', 'h5py', 'trimesh', 'matplotlib', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺少包: {', '.join(missing_packages)}")
        print("请运行: python evaluation/setup_evaluation.py")
        return False
    
    # 检查文件
    required_files = [
        "evaluation/data/registrations_m.hdf5",
        "volumetric_interpolation_pipeline.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺少文件: {', '.join(missing_files)}")
        return False
    
    print("\n✅ 系统状态正常")
    return True

def demonstrate_steps():
    """演示六步评估流程"""
    print("\n📋 六步评估流程演示")
    print("-" * 40)
    
    steps = [
        ("步骤1", "生成关键帧对", "从DFAUST数据中每隔k帧抽取首尾帧"),
        ("步骤2", "运行插值", "对每个关键帧对运行baseline和dual_reference方法"),
        ("步骤3", "评估结果", "计算Chamfer、jerk、ARAP等指标"),
        ("步骤4", "可视化结果", "生成Chamfer vs 时间曲线等图表"),
        ("步骤5", "对比方法", "计算改进百分比并生成详细报告"),
        ("步骤6", "结果汇总", "生成CSV和Markdown格式的最终报告")
    ]
    
    for i, (step_num, step_name, description) in enumerate(steps, 1):
        print(f"{step_num}: {step_name}")
        print(f"    {description}")
        print()

def show_metrics():
    """展示评估指标"""
    print("📊 评估指标说明")
    print("-" * 40)
    
    metrics = [
        ("几何质量指标", [
            ("Chamfer距离", "衡量插值结果与真实帧的几何相似性"),
            ("法向一致性", "计算法向量夹角的平均值"),
            ("ARAP误差", "As-Rigid-As-Possible误差，衡量局部刚体变换质量")
        ]),
        ("时间平滑度指标", [
            ("Jerk", "加加速度，衡量运动的时间平滑度"),
            ("平均/最大jerk", "统计jerk的均值和最大值")
        ]),
        ("物理合法性指标", [
            ("骨长标准差", "衡量骨骼长度的一致性"),
            ("自碰撞计数", "检测网格自交情况"),
            ("脚部滑动", "检测脚部与地面的滑动")
        ])
    ]
    
    for category, metric_list in metrics:
        print(f"\n{category}:")
        for metric_name, description in metric_list:
            print(f"  • {metric_name}: {description}")

def show_usage_examples():
    """展示使用示例"""
    print("\n🚀 使用示例")
    print("-" * 40)
    
    examples = [
        ("快速测试（无GT模式）", 
         "python evaluation/run_evaluation_pipeline.py --max_pairs 1 --no_gt"),
        ("完整评估（有GT模式）", 
         "python evaluation/run_evaluation_pipeline.py"),
        ("自定义参数", 
         "python evaluation/run_evaluation_pipeline.py --subject_id 50002 --sequence_id jump --k 10"),
        ("分步运行", 
         "python evaluation/generate_keyframe_pairs.py\npython evaluation/run_interpolation.py\npython evaluation/evaluate_interpolation.py")
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")

def show_output_files():
    """展示输出文件"""
    print("\n📈 输出文件说明")
    print("-" * 40)
    
    files = [
        ("results.csv", "所有评估指标的详细数据"),
        ("evaluation_report.md", "基础评估报告"),
        ("detailed_comparison_report.md", "详细的方法对比分析"),
        ("improvement_summary.csv", "改进百分比汇总"),
        ("chamfer_vs_time.png", "Chamfer距离随时间变化曲线"),
        ("metrics_comparison.png", "各指标对比箱线图"),
        ("method_improvement.png", "方法改进百分比柱状图")
    ]
    
    for filename, description in files:
        print(f"• {filename}: {description}")

def show_performance_features():
    """展示性能特性"""
    print("\n⚡ 性能优化特性")
    print("-" * 40)
    
    features = [
        ("内存优化", [
            "使用流式处理，避免一次性加载所有数据",
            "优化Chamfer距离计算，使用KDTree加速",
            "简化ARAP计算，仅计算关键区域的刚体变换"
        ]),
        ("时间优化", [
            "并行处理多个关键帧对",
            "缓存中间计算结果",
            "使用高效的数值计算库"
        ]),
        ("资源要求", [
            "内存: 16GB RAM",
            "GPU: 单GPU（可选，支持CPU-only）",
            "时间: 30帧测试序列 < 1分钟",
            "存储: 约2GB临时空间"
        ])
    ]
    
    for category, feature_list in features:
        print(f"\n{category}:")
        for feature in feature_list:
            print(f"  • {feature}")

def main():
    """主演示函数"""
    print_banner()
    
    # 检查系统状态
    if not check_system_status():
        print("\n❌ 系统状态检查失败，请先安装依赖包")
        print("运行: python evaluation/setup_evaluation.py")
        return
    
    # 演示各个部分
    demonstrate_steps()
    show_metrics()
    show_usage_examples()
    show_output_files()
    show_performance_features()
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("\n📝 下一步操作:")
    print("1. 安装依赖: python evaluation/setup_evaluation.py")
    print("2. 系统检查: python evaluation/simple_test.py")
    print("3. 快速测试: python evaluation/run_evaluation_pipeline.py --max_pairs 1 --no_gt")
    print("4. 完整评估: python evaluation/run_evaluation_pipeline.py")
    print("5. 查看文档: evaluation/README.md")
    print("=" * 60)

if __name__ == "__main__":
    main() 