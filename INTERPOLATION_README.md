# 体素视频插值系统 (Volumetric Video Interpolation System)

基于骨骼的体素视频帧插值系统，使用SLERP插值和Linear Blend Skinning技术实现高质量的网格插值。

## 🎯 功能特性

### 核心功能
- **骨骼SLERP插值**: 使用四元数SLERP对骨骼旋转进行平滑插值
- **自动蒙皮权重优化**: 基于LBS损失函数自动优化蒙皮权重
- **多格式导出**: 支持OBJ、PLY、STL格式导出
- **可视化对比**: 实时可视化插值结果与原始帧的对比
- **批量处理**: 支持批量插值多个序列

### 技术特点
- **高质量插值**: 基于骨骼变换的物理合理插值
- **权重优化**: 使用中间帧优化蒙皮权重质量
- **内存高效**: 支持大网格序列的流式处理
- **可扩展**: 模块化设计，易于扩展新功能

## 📋 系统要求

### 依赖包
```bash
pip install numpy torch open3d scipy scikit-learn tqdm matplotlib
```

### 数据要求
- 骨骼预测数据 (由 `SkelSequencePrediction.py` 生成)
- 网格序列文件 (OBJ格式)
- 预计算蒙皮权重 (可选，由 `Skinning.py` 生成)

## 🚀 快速开始

### 1. 基本插值
```bash
# 使用命令行界面
python interpolate_cli.py --start 10 --end 50 --num 20

# 或直接运行主程序
python Interpolate.py
```

### 2. 可视化插值结果
```bash
python interpolate_cli.py --start 0 --end 100 --num 50 --visualize
```

### 3. 导出不同格式
```bash
python interpolate_cli.py --start 5 --end 25 --num 10 --format ply
```

### 4. 使用配置文件
```bash
python interpolate_cli.py --config my_config.json
```

## 📖 详细使用说明

### 命令行参数

#### 基本参数
- `--start`: 起始帧索引
- `--end`: 结束帧索引  
- `--num`: 插值帧数

#### 路径参数
- `--skeleton-dir`: 骨骼数据目录 (默认: `output/skeleton_prediction`)
- `--mesh-dir`: 网格文件目录 (默认: `D:/Code/VVEditor/Rafa_Approves_hd_4k`)
- `--weights`: 预计算蒙皮权重路径 (默认: `output/skinning_weights_auto.npz`)
- `--output`: 输出目录路径 (默认: `output/interpolation_results`)

#### 功能参数
- `--format`: 输出格式 (`obj`, `ply`, `stl`, 默认: `obj`)
- `--visualize`: 可视化插值结果
- `--no-optimize`: 跳过权重优化
- `--save-animation`: 保存动画帧

#### 高级参数
- `--max-iter`: 权重优化最大迭代次数 (默认: 300)
- `--regularization`: 正则化系数 (默认: 0.01)

### 配置文件格式

```json
{
  "skeleton_dir": "output/skeleton_prediction",
  "mesh_dir": "D:/Code/VVEditor/Rafa_Approves_hd_4k", 
  "weights": "output/skinning_weights_auto.npz",
  "output": "output/interpolation_results",
  "format": "obj",
  "visualize": true,
  "save_animation": true,
  "max_iter": 300,
  "regularization": 0.01,
  "interpolation_examples": [
    {
      "name": "short_sequence",
      "start": 10,
      "end": 50, 
      "num": 20
    }
  ]
}
```

## 🔧 API 使用

### 基本用法

```python
from Interpolate import VolumetricInterpolator

# 初始化插值器
interpolator = VolumetricInterpolator(
    skeleton_data_dir="output/skeleton_prediction",
    mesh_folder_path="path/to/mesh/folder",
    weights_path="output/skinning_weights_auto.npz"  # 可选
)

# 生成插值帧
interpolated_frames = interpolator.generate_interpolated_frames(
    frame_start=10,
    frame_end=50,
    num_interpolate=20,
    max_optimize_frames = max_optimize_frames,
    optimize_weights=True
)

# 导出插值序列
interpolator.export_interpolation_sequence(
    frame_start=10,
    frame_end=50,
    num_interpolate=20,
    max_optimize_frames = 10,
    output_dir="output/interpolation",
    format='obj'
)

# 可视化插值结果
interpolator.visualize_interpolation(
    frame_start=10,
    frame_end=50,
    num_interpolate=20,
    output_dir="output/interpolation",
    save_animation=True
)
```

### 高级用法

```python
# 自定义权重优化参数
weights, loss = interpolator.optimize_skinning_weights_for_frame(
    target_frame_idx=25,
    reference_frame_idx=10,
    max_iter=500,
    regularization_lambda=0.01
)

# 手动插值骨骼变换
interpolated_transforms = interpolator.interpolate_skeleton_transforms(
    frame_start=10,
    frame_end=50,
    t=0.5  # 插值参数 [0, 1]
)

# 插值关键点
interpolated_keypoints = interpolator.interpolate_keypoints(
    frame_start=10,
    frame_end=50,
    t=0.5
)
```

## 🎨 输出文件说明

### 插值序列文件
```
output/interpolation_results/
├── frame_000000.obj          # 起始帧
├── frame_000001.obj          # 插值帧 1
├── frame_000002.obj          # 插值帧 2
├── ...
├── frame_000020.obj          # 结束帧
└── interpolation_metadata.json # 元数据
```

### 可视化文件
```
output/interpolation_results/
├── animation_frames/
│   ├── frame_0000.png        # 动画帧 1
│   ├── frame_0001.png        # 动画帧 2
│   └── ...
└── interpolated_frame_0000.obj  # 中间结果
```

### 元数据文件
```json
{
  "frame_start": 10,
  "frame_end": 50,
  "num_interpolate": 20,
  "total_frames": 22,
  "format": "obj",
  "skeleton_data_dir": "output/skeleton_prediction",
  "mesh_folder_path": "path/to/mesh/folder",
  "interpolation_method": "skeleton_slerp_lbs",
  "optimization_frames": [10, 11, 12, ..., 50]
}
```

## 🔬 技术原理

### 1. 骨骼SLERP插值

使用四元数SLERP (Spherical Linear Interpolation) 对骨骼旋转进行插值：

```python
# 四元数SLERP插值
quat_start = R.from_matrix(R_start).as_quat()
quat_end = R.from_matrix(R_end).as_quat()

# 确保四元数在同一半球
if np.dot(quat_start, quat_end) < 0:
    quat_end = -quat_end

# SLERP插值
quat_interp = (1-t) * quat_start + t * quat_end
quat_interp = quat_interp / np.linalg.norm(quat_interp)
R_interp = R.from_quat(quat_interp).as_matrix()
```

### 2. Linear Blend Skinning

使用LBS变换将骨骼变换应用到网格顶点：

```python
def apply_lbs_transform(self, rest_vertices, weights, transforms):
    """应用Linear Blend Skinning变换"""
    num_vertices = rest_vertices.shape[0]
    num_joints = transforms.shape[0]
    
    rest_vertices_homo = np.hstack([rest_vertices, np.ones((num_vertices, 1))])
    transformed_vertices = np.zeros((num_vertices, 3))
    
    for j in range(num_joints):
        joint_transform = transforms[j]
        transformed_homo = (joint_transform @ rest_vertices_homo.T).T
        transformed_xyz = transformed_homo[:, :3]
        joint_weights = weights[:, j:j+1]
        transformed_vertices += joint_weights * transformed_xyz
    
    return transformed_vertices
```

### 3. 权重优化

使用L-BFGS-B优化器最小化LBS重建损失：

```python
def compute_lbs_loss(self, weights_flat, rest_vertices, target_vertices, transforms, 
                    regularization_lambda=0.01):
    """计算LBS损失函数"""
    weights = weights_flat.reshape(num_vertices, num_joints)
    weights = np.maximum(weights, 0)
    weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-8)
    
    predicted_vertices = self.apply_lbs_transform(rest_vertices, weights, transforms)
    reconstruction_loss = np.mean(np.sum((predicted_vertices - target_vertices)**2, axis=1))
    sparsity_loss = np.mean(np.sum(weights**2, axis=1))
    
    total_loss = reconstruction_loss + regularization_lambda * sparsity_loss
    return total_loss
```

## 📊 性能优化

### 内存优化
- 流式处理大网格序列
- 采样优化减少计算量
- 缓存中间结果

### 计算优化
- 并行处理多个关节
- 向量化LBS计算
- 高效的四元数运算

### 质量优化
- 多帧权重平均
- 自适应正则化
- 渐进式优化

## 🐛 故障排除

### 常见问题

#### 1. 骨骼数据加载失败
```
❌ 无法加载骨骼数据: [Errno 2] No such file or directory
```
**解决方案**: 确保先运行 `SkelSequencePrediction.py` 生成骨骼数据

#### 2. 网格文件不匹配
```
⚠️ 网格文件数 (100) 与骨骼帧数 (120) 不匹配
```
**解决方案**: 检查网格文件和骨骼数据的一致性

#### 3. 权重优化失败
```
❌ 权重优化过程中出现错误: Optimization failed
```
**解决方案**: 
- 减少 `max_iter` 参数
- 增加 `regularization` 参数
- 检查网格质量

#### 4. 内存不足
```
❌ MemoryError: Unable to allocate array
```
**解决方案**:
- 减少插值帧数
- 使用 `--no-optimize` 跳过权重优化
- 分批处理大序列

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 工作流程

### 完整流程
1. **骨骼预测**: 运行 `SkelSequencePrediction.py` 生成骨骼数据
2. **蒙皮权重**: 运行 `Skinning.py` 生成蒙皮权重 (可选)
3. **插值设置**: 配置插值参数 (帧范围、插值数量等)
4. **执行插值**: 运行插值系统生成中间帧
5. **结果验证**: 可视化检查插值质量
6. **导出结果**: 保存插值序列到指定格式

### 示例工作流
```bash
# 1. 生成骨骼数据
python SkelSequencePrediction.py

# 2. 生成蒙皮权重 (可选)
python Skinning.py

# 3. 执行插值
python interpolate_cli.py --start 10 --end 50 --num 20 --visualize

# 4. 检查结果
ls output/interpolation_results/
```

## 📈 质量评估

### 评估指标
- **重建误差**: 插值帧与原始帧的顶点距离
- **平滑度**: 相邻帧之间的变化连续性
- **物理合理性**: 骨骼变换的物理约束

### 可视化评估
- 颜色编码误差分布
- 骨骼轨迹可视化
- 网格变形对比

## 🔮 未来扩展

### 计划功能
- [ ] 支持更多插值算法 (Catmull-Rom, B-spline)
- [ ] 实时插值预览
- [ ] 多分辨率处理
- [ ] GPU加速计算
- [ ] 自动质量评估

### 贡献指南
欢迎提交Issue和Pull Request来改进系统！

## 📄 许可证

本项目基于MIT许可证开源。

## 🤝 致谢

感谢以下开源项目的支持：
- Open3D: 3D数据处理和可视化
- SciPy: 科学计算和优化
- NumPy: 数值计算
- PyTorch: 深度学习框架 