# 插值评估系统

本系统实现了完整的插值方法评估流水线，支持对比baseline与dual_reference两种插值方法。

## 系统特性

- ✅ **支持有/无GT两种模式**
- ✅ **输出多种评估指标**: Chamfer距离、jerk、ARAP误差、骨长标准差、自碰撞计数等
- ✅ **自动对比分析**: baseline vs dual_reference
- ✅ **生成详细报告**: CSV + Markdown + 可视化图表
- ✅ **性能优化**: 可在16GB RAM单GPU环境下1分钟内完成30帧测试序列评估

## 文件结构

```
evaluation/
├── data/                          # 数据目录
│   ├── registrations_m.hdf5       # DFAUST数据
│   └── keyframe_pairs/            # 生成的关键帧对
├── results/                       # 结果目录
│   ├── baseline/                  # baseline方法结果
│   ├── dual_reference/            # dual_reference方法结果
│   ├── results.csv                # 评估结果CSV
│   ├── evaluation_report.md       # 评估报告
│   ├── detailed_comparison_report.md  # 详细对比报告
│   └── *.png                      # 可视化图表
├── generate_keyframe_pairs.py     # 步骤1: 生成关键帧对
├── run_interpolation.py           # 步骤2: 运行插值
├── evaluate_interpolation.py      # 步骤3: 评估结果
├── visualize_results.py           # 步骤4: 可视化结果
├── compare_methods.py             # 步骤5: 对比方法
├── run_evaluation_pipeline.py     # 主控制脚本
├── utils_mesh.py                  # 网格处理工具
└── README.md                      # 本文件
```

## 快速开始

### 1. 一键运行完整评估

```bash
# 有GT模式（推荐）
python evaluation/run_evaluation_pipeline.py

# 无GT模式（仅计算内部指标）
python evaluation/run_evaluation_pipeline.py --no_gt

# 快速测试（仅处理3对关键帧）
python evaluation/run_evaluation_pipeline.py --max_pairs 3
```

### 2. 分步运行

```bash
# 步骤1: 生成关键帧对
python evaluation/generate_keyframe_pairs.py

# 步骤2: 运行插值
python evaluation/run_interpolation.py

# 步骤3: 评估结果
python evaluation/evaluate_interpolation.py

# 步骤4: 可视化结果
python evaluation/visualize_results.py

# 步骤5: 对比方法
python evaluation/compare_methods.py
```

## 评估指标

### 几何质量指标
- **Chamfer距离**: 衡量插值结果与真实帧的几何相似性
- **法向一致性**: 计算法向量夹角的平均值
- **ARAP误差**: As-Rigid-As-Possible误差，衡量局部刚体变换质量

### 时间平滑度指标
- **Jerk**: 加加速度，衡量运动的时间平滑度
- **平均/最大jerk**: 统计jerk的均值和最大值

### 物理合法性指标
- **骨长标准差**: 衡量骨骼长度的一致性
- **自碰撞计数**: 检测网格自交情况
- **脚部滑动**: 检测脚部与地面的滑动

## 输出文件说明

### 主要结果文件
- `results.csv`: 所有评估指标的详细数据
- `evaluation_report.md`: 基础评估报告
- `detailed_comparison_report.md`: 详细的方法对比分析
- `improvement_summary.csv`: 改进百分比汇总

### 可视化图表
- `chamfer_vs_time.png`: Chamfer距离随时间变化曲线
- `metrics_comparison.png`: 各指标对比箱线图
- `method_improvement.png`: 方法改进百分比柱状图

## 性能优化

### 内存优化
- 使用流式处理，避免一次性加载所有数据
- 优化Chamfer距离计算，使用KDTree加速
- 简化ARAP计算，仅计算关键区域的刚体变换

### 时间优化
- 并行处理多个关键帧对
- 缓存中间计算结果
- 使用高效的数值计算库

### 资源要求
- **内存**: 16GB RAM
- **GPU**: 单GPU（可选，支持CPU-only）
- **时间**: 30帧测试序列 < 1分钟
- **存储**: 约2GB临时空间

## 命令行参数

### 主控制脚本参数
```bash
python evaluation/run_evaluation_pipeline.py [选项]

选项:
  --gt_hdf5 PATH          GT数据HDF5文件路径
  --subject_id ID         DFAUST subject ID (默认: 50002)
  --sequence_id ID        DFAUST sequence ID (默认: jump)
  --k INT                 每隔k帧抽取一对关键帧 (默认: 10)
  --max_pairs INT         最大处理的关键帧对数量 (默认: 3)
  --no_gt                 无GT模式，仅计算内部指标
  --skip_steps INT...     跳过的步骤编号（1-5）
  --timeout INT           每个步骤的超时时间（秒）(默认: 60)
```

### 评估脚本参数
```bash
python evaluation/evaluate_interpolation.py [选项]

选项:
  --pairs_dir PATH        关键帧对目录
  --results_dir PATH      插值结果目录
  --methods METHOD...     要评估的插值方法
  --output_dir PATH       评估结果输出目录
  --no_gt                 无GT模式
  --gt_hdf5 PATH         GT数据HDF5文件路径
```

## 故障排除

### 常见问题

1. **缺少依赖包**
   ```bash
   pip install numpy pandas trimesh h5py matplotlib seaborn scipy scikit-learn
   ```

2. **内存不足**
   - 减少`--max_pairs`参数
   - 使用`--no_gt`模式
   - 增加系统虚拟内存

3. **超时错误**
   - 增加`--timeout`参数
   - 检查GPU内存是否充足
   - 使用CPU-only模式

4. **文件路径错误**
   - 确保DFAUST数据文件存在
   - 检查文件权限
   - 使用绝对路径

### 调试模式

```bash
# 仅运行特定步骤
python evaluation/run_evaluation_pipeline.py --skip_steps 2 3 4 5

# 详细输出
python evaluation/evaluate_interpolation.py --debug

# 检查数据
python evaluation/generate_keyframe_pairs.py --check_only
```

## 扩展功能

### 添加新的评估指标
1. 在`utils_mesh.py`中添加计算函数
2. 在`evaluate_interpolation.py`中集成新指标
3. 更新可视化脚本

### 支持新的插值方法
1. 确保新方法输出格式兼容
2. 更新`run_interpolation.py`中的方法列表
3. 在对比脚本中添加新方法

### 批量评估
```bash
# 评估多个动作序列
for sequence in jump run walk; do
    python evaluation/run_evaluation_pipeline.py --sequence_id $sequence
done
```

## 贡献指南

1. 遵循现有代码风格
2. 添加适当的文档和注释
3. 确保新功能通过测试
4. 更新README文档

## 许可证

本项目遵循MIT许可证。 