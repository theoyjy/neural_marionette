# Neural Marionette - 体素视频插值系统

一个基于神经网络的体素视频插值系统，支持骨骼预测、蒙皮权重优化和高质量插值。

## 🚀 快速开始

### 主要Pipeline

```bash
# 运行完整的体素视频插值pipeline
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> [--num_interpolate 10] [--method baseline] [--skip_skeleton]
```

### 示例用法

```bash
# 基本用法（基础插值方法）
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20

# 双参考帧插值方法
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --method dual_reference

# 相似帧自适应插值方法
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --method adaptive_similarity

# 指定插值帧数
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --num_interpolate 15

# 跳过骨骼预测（如果已有数据）
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --skip_skeleton
```

### 插值方法比较

```bash
# 测试所有插值方法
python test_interpolation_methods.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20

# 演示不同方法
python demo_interpolation_methods.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20
```

## 📁 项目结构

### 核心模块

- **`volumetric_interpolation_pipeline.py`** - 主pipeline脚本
- **`SkelSequencePrediction.py`** - 骨骼序列预测模块
- **`Interpolate.py`** - 体素插值核心模块（支持多种插值方法）
- **`Skinning.py`** - 蒙皮权重优化模块

### 测试和演示模块

- **`test_interpolation_methods.py`** - 插值方法比较测试
- **`demo_interpolation_methods.py`** - 插值方法演示

### 可视化模块

- **`SkelVisualizer.py`** - 骨骼可视化工具
- **`simple_visualize.py`** - 简单可视化工具

### 文档

- **`README_PIPELINE.md`** - Pipeline详细使用说明
- **`README_INTERPOLATION_METHODS.md`** - 插值方法详细说明

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

3. **多种插值方法**
   - **基础插值方法 (Baseline)**: 使用单一参考帧，计算效率高
   - **双参考帧插值 (Dual Reference)**: 使用起始帧和结束帧作为双参考，提供更自然的过渡
   - **相似帧自适应插值 (Adaptive Similarity)**: 为每个插值帧找到最相似的原始骨骼，提供最高质量结果

4. **高质量插值**
   - SLERP旋转插值
   - 相对变换处理
   - 坐标系对齐
   - 体积保持

5. **Pipeline集成**
   - 一键式pipeline
   - 时间性能监控
   - 稳定的输出目录管理
   - 错误处理和恢复

### 📊 性能优化

- **多线程处理**: 网格加载和体素化
- **数据缓存**: 骨骼数据和蒙皮权重
- **时间监控**: 关键步骤性能分析
- **内存优化**: 高效的数据结构

## 🎯 插值方法选择指南

| 方法 | 计算复杂度 | 内存使用 | 插值质量 | 适用场景 |
|------|------------|----------|----------|----------|
| Baseline | 低 | 低 | 中等 | 简单插值，姿态差异小 |
| Dual Reference | 中等 | 中等 | 高 | 中等复杂度，姿态差异大 |
| Adaptive Similarity | 高 | 高 | 最高 | 复杂场景，需要最高质量 |

## 🐛 最近修复

### Unicode编码错误修复 (最新)
- ✅ 修复了Windows GBK编码下的Unicode编码错误
- ✅ 移除了所有emoji字符，避免编码问题
- ✅ 双参考帧方法现在可以正常工作
- ✅ 所有插值方法都已测试通过

### 双参考帧方法修复
- ✅ 修复了双参考帧插值方法中的 `KeyError: 'keypoints'` 错误
- ✅ 添加了缺失的关键点数据生成
- ✅ 修复了自适应相似性插值器中的相同问题
- ✅ 优化了权重文件路径生成逻辑
- ✅ 改进了可视化功能的错误处理

**修复内容：**
1. 在 `_generate_dual_reference_frame` 函数中添加了 `interpolate_keypoints` 调用
2. 在 `_generate_adaptive_frame` 函数中添加了相同的关键点生成
3. 修复了权重文件保存路径问题
4. 确保所有插值方法都能正确生成可视化所需的完整数据
5. 移除了所有emoji字符，避免Windows GBK编码问题

**测试结果：**
```bash
# 双参考帧方法测试成功
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 0 5 --num_interpolate 3 --method dual_reference

# 生成文件：
# - 3个OBJ文件 (插值网格)
# - 3个PNG文件 (可视化图像)
# - 权重文件保存在统一目录
```

**测试方法：**
```bash
# 运行修复测试
python test_dual_reference_fix.py

# 直接测试双参考帧方法
python volumetric_interpolation_pipeline.py data/demo/source 0 5 --method dual_reference
```

## 📋 输出结构

```
output/
└── pipeline_<name>_<hash>/
    ├── skeleton_prediction/     # 骨骼预测数据 (共享)
    │   ├── keypoints.npy
    │   ├── transforms.npy
    │   └── parents.npy
    ├── skinning_weights/        # 蒙皮权重文件 (共享)
    │   ├── skinning_weights_ref0_opt0-10_step1.npz
    │   ├── dual_ref_start_frame_0_weights.npz
    │   ├── dual_ref_end_frame_10_weights.npz
    │   └── adaptive_similar_frame_5_weights.npz
    ├── interpolation_baseline/   # 基础插值结果
    │   ├── interpolated_frame_0000.obj
    │   ├── interpolated_frame_0001.obj
    │   └── ...
    ├── interpolation_dual_reference/  # 双参考帧插值结果
    │   ├── interpolated_frame_0000.obj
    │   ├── interpolated_frame_0001.obj
    │   └── ...
    └── interpolation_adaptive_similarity/  # 自适应相似性插值结果
        ├── adaptive_frame_0000.obj
        ├── adaptive_frame_0001.obj
        └── ...
```

## 🛠️ 安装依赖

```
```