# Neural Marionette - 体素视频插值系统

一个基于神经网络的体素视频插值系统，支持骨骼预测、蒙皮权重优化和高质量插值。

## 🚀 快速开始

### 主要Pipeline

```bash
# 运行完整的体素视频插值pipeline
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [--num_interpolate 10] [--skip_skeleton]
```

### 示例用法

```bash
# 基本用法
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20

# 指定插值帧数
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --num_interpolate 15

# 跳过骨骼预测（如果已有数据）
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --skip_skeleton
```

## 📁 项目结构

### 核心模块

- **`volumetric_interpolation_pipeline.py`** - 主pipeline脚本
- **`SkelSequencePrediction.py`** - 骨骼序列预测模块
- **`Interpolate.py`** - 体素插值核心模块
- **`Skinning.py`** - 蒙皮权重优化模块

### 可视化模块

- **`SkelVisualizer.py`** - 骨骼可视化工具
- **`simple_visualize.py`** - 简单可视化工具

### 文档

- **`README_PIPELINE.md`** - Pipeline详细使用说明

## 🔧 功能特性

### ✅ 已完成功能

1. **骨骼预测**
   - 使用Neural Marionette模型预测骨骼序列
   - 多线程网格处理，提高性能
   - 数据缓存和重用机制

2. **蒙皮权重优化**
   - 基于L-BFGS-B的权重优化
   - 自动参考帧选择
   - 权重文件缓存和重用

3. **高质量插值**
   - SLERP旋转插值
   - 相对变换处理
   - 坐标系对齐
   - 体积保持

4. **Pipeline集成**
   - 一键式pipeline
   - 时间性能监控
   - 稳定的输出目录管理
   - 错误处理和恢复

### 📊 性能优化

- **多线程处理**: 网格加载和体素化
- **数据缓存**: 骨骼数据和蒙皮权重
- **时间监控**: 关键步骤性能分析
- **内存优化**: 高效的数据结构

## 📋 输出结构

```
output/
└── pipeline_<name>_<hash>/
    ├── skeleton_prediction/     # 骨骼预测数据
    ├── skinning_weights/        # 蒙皮权重文件
    └── interpolation_results/    # 插值结果
        ├── interpolated_frame_0000.obj
        ├── interpolated_frame_0001.obj
        └── ...
```

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

## 📖 详细文档

更多详细信息请参考：
- [Pipeline使用指南](README_PIPELINE.md)

## 🎯 使用场景

- 体素视频插值
- 动画序列生成
- 骨骼动画处理
- 3D模型变形

## 📝 更新日志

### 最新版本
- ✅ 修复文件生成数量问题
- ✅ 优化pipeline性能
- ✅ 清理冗余代码
- ✅ 改进错误处理
