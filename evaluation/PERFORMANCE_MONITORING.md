# 性能监控功能

插值评估现在包含详细的系统资源监控，可以帮助分析性能瓶颈和优化插值算法。

## 🎯 监控的指标

### ⏱️ 时间指标
- `interpolation_time_seconds` - 总插值时间
- `time_per_frame_seconds` - 每帧平均处理时间
- `monitoring_samples` - 监控采样次数

### 💻 系统资源
- `cpu_usage_avg_percent` / `cpu_usage_max_percent` - CPU使用率
- `memory_usage_avg_gb` / `memory_usage_max_gb` - 内存使用量
- `gpu_usage_avg_percent` / `gpu_usage_max_percent` - GPU使用率
- `gpu_memory_avg_gb` / `gpu_memory_max_gb` - GPU内存使用量

## 📂 输出文件

### 性能数据文件
每次插值会生成性能监控文件：
```
evaluation/method/database_name/subjectid_sequenceid_k{k}/pair_xxx/intp/
├── performance_baseline_pair_000.json     # baseline方法性能数据
├── performance_dual_reference_pair_000.json  # dual_reference方法性能数据
├── interpolated_frame_*.obj                # 插值结果
└── ...
```

### 集成到评估结果
性能数据会自动集成到最终的评估CSV和报告中，包含在：
- `results.csv` - 汇总所有指标
- `evaluation_report.md` - 详细报告
- `summary_statistics.md` - 统计汇总

## 📊 示例输出

### 控制台输出
```
插值成功: dual_reference - pair_000 (耗时: 12.34s)
性能数据已保存: evaluation/dual_reference/registrations_m/50002_jumping_jacks_k10/pair_000/intp/performance_dual_reference_pair_000.json
性能汇总: CPU平均45.2%, 内存峰值8.67GB, GPU平均78.1%, GPU内存峰值6.42GB
```

### JSON性能文件结构
```json
{
  "summary": {
    "total_time_seconds": 12.34,
    "sample_count": 25,
    "cpu_usage_percent": {"avg": 45.2, "max": 87.3, "min": 12.1},
    "memory_usage_gb": {"avg": 7.89, "max": 8.67, "min": 6.23},
    "gpu_usage_percent": {"avg": 78.1, "max": 95.6, "min": 45.2},
    "gpu_memory_gb": {"avg": 5.87, "max": 6.42, "min": 4.93},
    "timestamp": "2025-08-02T05:59:16.736734",
    "psutil_available": true,
    "gpu_available": true
  },
  "detailed_data": {
    "timestamps": [...],
    "cpu_usage": [...],
    "memory_usage": [...],
    "gpu_usage": [...],
    "gpu_memory": [...]
  }
}
```

### 评估结果文件中的性能列
```csv
method,pair,mean_chamfer,interpolation_time_seconds,cpu_usage_avg_percent,memory_usage_max_gb,gpu_usage_avg_percent
baseline,pair_000,0.0062,8.45,42.1,6.78,65.3
dual_reference,pair_000,0.0059,12.34,45.2,8.67,78.1
```

## 🔧 技术要求

### 依赖包（可选）
为了获得完整的监控数据，建议安装：
```bash
pip install psutil              # CPU和内存监控
pip install nvidia-ml-py3       # NVIDIA GPU监控
```

### 自动降级
如果某些监控工具不可用，系统会自动适应：
- 没有`psutil`：CPU和内存指标为0
- 没有GPU驱动：GPU指标为0
- 系统会继续正常工作，只是监控数据不完整

## 🎯 使用场景

### 性能优化
- 比较不同方法的资源使用
- 识别内存泄漏或CPU瓶颈
- 优化GPU利用率

### 基准测试
- 跨不同硬件配置的性能对比
- 方法性能的定量分析
- 扩展性研究

### 调试
- 超时问题诊断
- 资源不足问题定位
- 异常情况分析

## 📈 数据分析

可以使用这些数据进行：
1. **性能对比**：不同方法在相同硬件上的表现
2. **资源效率**：单位质量改进的资源消耗
3. **扩展性分析**：处理时间与数据复杂度的关系
4. **硬件要求**：为生产环境估算所需资源

这个监控系统为Neural Marionette的性能分析和优化提供了完整的数据基础！