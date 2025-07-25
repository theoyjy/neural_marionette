# Volumetric Video Interpolation Pipeline

完整的体素视频插值流水线，支持骨骼预测、蒙皮权重优化和插值生成。

## 🚀 快速开始

### 基本用法

```bash
# 基本插值
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame>

# 指定插值帧数
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> --num_interpolate 20

# 跳过骨骼预测（如果已经存在）
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> --skip_skeleton

# 启用可视化
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> --visualization
```

### 示例

```bash
# 对Rafa_Approves_hd_4k文件夹的帧10-20进行插值
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20

# 生成20个插值帧
python volumetric_interpolation_pipeline.py "D:/Code/VVEditor/Rafa_Approves_hd_4k" 10 20 --num_interpolate 20
```

## 📁 输出结构

```
output/
└── pipeline_[folder_name]_[hash]/
    ├── skeleton_prediction/          # 骨骼预测数据
    │   ├── keypoints.npy
    │   ├── transforms.npy
    │   ├── parents.npy
    │   └── ...
    ├── skinning_weights/            # 蒙皮权重文件
    │   ├── skinning_weights_ref10_opt10-20_step1.npz
    │   └── ...
    └── interpolation_results/        # 插值结果
        ├── interpolated_frame_0000.obj
        ├── interpolated_frame_0001.obj
        ├── debug_frame_0000.png
        └── ...
```

## 🔧 参数说明

### 必需参数
- `folder_path`: 输入网格文件夹路径（包含.obj文件）
- `start_frame`: 起始帧索引
- `end_frame`: 结束帧索引

### 可选参数
- `--num_interpolate`: 插值帧数（默认: 10）
- `--skip_skeleton`: 跳过骨骼预测步骤
- `--visualization`: 启用可视化（默认: 关闭）

## 🎯 工作流程

### 步骤1: 骨骼预测
- 检查是否已存在骨骼数据
- 如果不存在，调用 `SkelSequencePrediction.py`
- 使用Neural Marionette模型预测骨骼
- 保存到 `skeleton_prediction/` 目录

### 步骤2: 插值生成
- 初始化 `VolumetricInterpolator`
- 检查蒙皮权重文件
- 如果不存在，调用 `Skinning.py` 优化权重
- 生成插值帧并保存结果

## 📊 文件命名规则

### 蒙皮权重文件
```
skinning_weights_ref{reference_frame}_opt{start}-{end}_step{step}.npz
```

示例：
- `skinning_weights_ref10_opt10-20_step1.npz`: 参考帧10，优化帧10-20，步长1

### 插值结果文件
```
interpolated_frame_{frame_idx:04d}.obj
debug_frame_{frame_idx:04d}.png
```

## 🔍 调试和可视化

### 启用可视化
```bash
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> --visualization
```

### 查看调试信息
- 检查 `debug_frame_*.png` 文件
- 查看控制台输出的详细信息
- 分析生成的OBJ文件

## ⚡ 性能优化

### 缓存机制
- 骨骼数据会被缓存，重复使用相同文件夹时跳过预测
- 蒙皮权重文件会被缓存，相同参数时跳过优化
- 输出目录基于文件夹路径哈希，避免冲突

### 内存优化
- 限制最大处理帧数（默认200帧）
- 优化帧数限制（默认5帧）
- 自动清理临时文件

## 🐛 故障排除

### 常见问题

1. **依赖项缺失**
   ```bash
   pip install torch numpy open3d scipy matplotlib trimesh pygltflib imageio opencv-python
   ```

2. **内存不足**
   - 减少 `--num_interpolate` 参数
   - 减少 `max_frames` 参数

3. **文件路径问题**
   - 确保输入文件夹包含.obj文件
   - 检查文件权限

4. **GPU内存不足**
   - 减少批处理大小
   - 使用CPU模式

### 调试模式

```bash
# 详细输出
python volumetric_interpolation_pipeline.py <folder_path> <start_frame> <end_frame> --visualization

# 检查中间结果
ls output/pipeline_*/skeleton_prediction/
ls output/pipeline_*/skinning_weights/
ls output/pipeline_*/interpolation_results/
```

## 📈 性能指标

### 典型性能
- 骨骼预测: ~2-5秒/帧
- 权重优化: ~30-60秒
- 插值生成: ~1-2秒/帧

### 内存使用
- 骨骼预测: ~2-4GB
- 权重优化: ~4-8GB
- 插值生成: ~1-2GB

## 🔄 模块化设计

每个模块都是独立的，便于调试和维护：

- `SkelSequencePrediction.py`: 骨骼预测
- `Skinning.py`: 蒙皮权重优化
- `Interpolate.py`: 插值生成
- `volumetric_interpolation_pipeline.py`: 主流水线

## 📝 更新日志

- v1.0: 初始版本，支持基本插值功能
- v1.1: 添加坐标系修复和可视化
- v1.2: 添加缓存机制和性能优化 