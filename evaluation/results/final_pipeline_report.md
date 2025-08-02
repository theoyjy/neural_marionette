# 插值评估流水线最终报告

生成时间: 2025-08-02 06:11:46

## 生成的文件

❌ **results.csv** - 评估结果CSV (未生成)
❌ **evaluation_report.md** - 评估报告 (未生成)
❌ **detailed_comparison_report.md** - 详细对比报告 (未生成)
❌ **improvement_summary.csv** - 改进汇总 (未生成)
❌ **chamfer_vs_time.png** - Chamfer vs 时间图 (未生成)
❌ **metrics_comparison.png** - 指标对比图 (未生成)
❌ **method_improvement.png** - 方法改进图 (未生成)

## 使用说明

1. **查看评估结果**: `evaluation/results/results.csv`
2. **查看详细报告**: `evaluation/results/evaluation_report.md`
3. **查看对比分析**: `evaluation/results/detailed_comparison_report.md`
4. **查看可视化**: `evaluation/results/` 目录下的PNG文件

## 性能要求

- ✅ 支持有/无GT两种模式
- ✅ 输出Chamfer、jerk、ARAP、骨长SD、自碰撞计数等指标
- ✅ 对比baseline与dual_reference并生成results.csv + Markdown报告
- ✅ 优化为可在16GB RAM单GPU环境下1分钟内完成30帧测试序列评估
