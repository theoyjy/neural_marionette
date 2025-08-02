# DFAUST数据集批量评估使用指南

## 🎯 功能概述

批量评估系统能够自动对DFAUST数据集的男性(registrations_m.hdf5)和女性(registrations_f.hdf5)数据进行充分的插值评估测试，支持多种参数组合，生成详细的对比报告。

## 📁 目录结构

确保数据按以下结构组织：

```
evaluation/
├── data/
│   └── dfaust/
│       ├── registrations_m.hdf5     # 男性数据
│       ├── registrations_f.hdf5     # 女性数据
│       ├── keyframe_pairs/          # 生成的关键帧对
│       ├── all_sequences.json       # 序列信息
│       ├── male_sequences.json      # 男性序列信息
│       └── female_sequences.json    # 女性序列信息
├── results/
│   └── batch_evaluation/            # 批量评估结果
└── batch_evaluation.py             # 主评估脚本
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所有依赖：
```bash
C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/setup_evaluation.py
```

### 2. 检查数据集

首先检查DFAUST数据集中的可用序列：
```bash
C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/check_dfaust_sequences.py
```

这将生成：
- `evaluation/data/dfaust/all_sequences.json` - 完整序列信息
- `evaluation/data/dfaust/male_sequences.json` - 男性序列信息  
- `evaluation/data/dfaust/female_sequences.json` - 女性序列信息

### 3. 运行批量评估

#### 测试模式（推荐首次使用）

```bash
C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/batch_evaluation.py --test_mode
```

测试模式特点：
- 仅测试4个序列：50002_jumping_jacks, 50004_jumping_jacks, 50007_running_on_spot, 50020_running_on_spot
- 参数组合：k=[10,20], max_pairs=[1,2], no_gt=[True,False]
- 总测试数：32个
- 预计耗时：约20分钟

#### 完整模式

```bash
C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/batch_evaluation.py
```

完整模式特点：
- 测试所有129个序列（65个男性 + 64个女性）
- 参数组合：k=[5,10,15,20], max_pairs=[1,2,3], no_gt=[True,False]
- 总测试数：约3000+个
- 预计耗时：几十小时

#### 自定义评估

```bash
# 仅无GT模式
C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/batch_evaluation.py --no_gt_only

# 自定义k值和max_pairs
C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/batch_evaluation.py --k_values 10 20 --max_pairs 1 2 --test_mode
```

## 📊 数据集统计

根据最新检查结果：

### 男性数据 (registrations_m.hdf5)
- **受试者数量**: 5人 (50002, 50007, 50009, 50026, 50027)
- **序列总数**: 65个
- **主要动作**: jumping_jacks, running_on_spot, punching, shake_hips等

### 女性数据 (registrations_f.hdf5)  
- **受试者数量**: 5人 (50004, 50020, 50021, 50022, 50025)
- **序列总数**: 64个
- **主要动作**: jumping_jacks, running_on_spot, punching, shake_hips等

### 总计
- **受试者数量**: 10人
- **序列总数**: 129个
- **网格顶点数**: 6890
- **序列长度**: 142-1251帧不等

## 📈 评估结果

### 测试模式结果示例

最新测试模式结果（32个测试）：
- **成功率**: 100% (32/32)
- **男性数据**: 100% (16/16)
- **女性数据**: 100% (16/16)
- **平均耗时**: 40.8秒
- **最大耗时**: 51.1秒
- **最小耗时**: 31.0秒

### 输出文件

批量评估完成后，在 `evaluation/results/batch_evaluation/` 目录下生成：

1. **batch_results_YYYYMMDD_HHMMSS.json** - 详细结果数据
   - 每个测试的完整信息
   - 包含命令输出、错误信息、耗时等

2. **batch_summary_YYYYMMDD_HHMMSS.csv** - 汇总结果表格
   - 测试时间、参数、成功状态、耗时等
   - 便于Excel分析

3. **batch_report_YYYYMMDD_HHMMSS.md** - 评估报告
   - 成功率统计
   - 参数组合分析
   - 性能统计
   - 失败案例分析

## 🎛️ 参数说明

### 命令行参数

- `--test_mode`: 测试模式，仅运行少量测试
- `--python_path`: Python解释器路径（默认：C:\Users\sky\miniconda3\envs\nmario\python.exe）
- `--k_values`: k值列表（默认：[10, 20]）
- `--max_pairs`: max_pairs值列表（默认：[1, 2]）
- `--no_gt_only`: 仅运行无GT模式

### 评估参数

- **k**: 关键帧间隔，k越大序列越稀疏
- **max_pairs**: 最大关键帧对数量，控制测试规模
- **no_gt**: 是否使用Ground Truth模式
  - True: 无GT模式，仅计算内部指标
  - False: 有GT模式，计算与真实帧的误差

## 🔧 性能优化

### 系统要求
- **内存**: 16GB RAM
- **处理器**: 支持多核处理
- **存储**: 足够空间存储中间结果
- **GPU**: 可选，用于加速计算

### 优化策略
- 自动清理中间文件节省空间
- 并行处理多个关键帧对
- 内存流式处理避免OOM
- 超时控制防止死锁

## 🐛 故障排除

### 常见问题

1. **UnicodeEncodeError**: 
   - 已修复emoji字符编码问题
   - 确保使用最新版本脚本

2. **文件路径错误**:
   - 确保数据在 `evaluation/data/dfaust/` 目录下
   - 运行 `check_dfaust_sequences.py` 验证数据

3. **内存不足**:
   - 减少max_pairs值
   - 使用test_mode进行小规模测试

4. **评估超时**:
   - 检查硬件性能
   - 增加timeout参数

### 调试技巧

1. 查看详细日志：
   ```bash
   C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/batch_evaluation.py --test_mode > log.txt 2>&1
   ```

2. 单独测试某个序列：
   ```bash
   C:\Users\sky\miniconda3\envs\nmario\python.exe evaluation/run_evaluation_pipeline.py --subject_id 50002 --sequence_id jumping_jacks --max_pairs 1 --no_gt
   ```

## 📝 最佳实践

1. **首次使用**：先运行test_mode确保系统正常
2. **长时间评估**：使用后台运行，定期检查进度
3. **结果分析**：重点关注CSV文件中的成功率和耗时
4. **存储管理**：定期清理output目录释放空间
5. **参数调优**：根据硬件性能调整k值和max_pairs

## 🎯 下一步扩展

1. **新数据集支持**: 添加AMASS数据集评估
2. **更多指标**: 增加新的评估指标
3. **可视化**: 生成更丰富的图表分析
4. **并行优化**: 进一步提升评估效率
5. **云端部署**: 支持分布式评估

---

## 📞 技术支持

如遇问题，请检查：
1. Python环境和依赖包
2. 数据文件完整性
3. 系统资源使用情况
4. 错误日志详细信息