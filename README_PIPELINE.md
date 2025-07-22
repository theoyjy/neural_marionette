# Neural Marionette - 体积视频插值管道

完整的体积视频插值管道，集成了修复后的DemBones蒙皮权重计算。

## 🚀 功能特性

1. **智能帧处理**: 每个.obj文件通过voxel化后交给NeuralMarionette预测skeleton
2. **自动Rest Pose检测**: 基于多重指标自动选择最佳rest pose
3. **骨骼引导网格统一**: 解决VV数据中顶点数量和拓扑不一致的问题
4. **修复的DemBones集成**: 使用正确的API进行可靠的蒙皮权重预测
5. **高质量插值**: 生成任意两帧之间的指定数量插值mesh

## 📋 使用方法

### 基本命令
```bash
python complete_vv_pipeline.py <folder_path> [选项]
```

### 参数说明
- `folder_path`: **必需** - 包含.obj文件的文件夹路径
- `--start_frame`: 处理起始帧索引 (默认: 0)
- `--end_frame`: 处理结束帧索引 (默认: 自动检测最后一帧)
- `--interp_from`: 插值起始帧索引 (默认: 0)
- `--interp_to`: 插值结束帧索引 (默认: 1)
- `--num_interp`: 插值帧数 (默认: 5)
- `--output_dir`: 输出目录 (默认: 输入文件夹下的vv_complete_output)

### 使用示例

#### 1. 处理所有帧并在前两帧间插值
```bash
python complete_vv_pipeline.py "path/to/obj/folder"
```

#### 2. 处理指定范围的帧
```bash
python complete_vv_pipeline.py "path/to/obj/folder" --start_frame 10 --end_frame 50
```

#### 3. 在特定两帧之间生成插值
```bash
python complete_vv_pipeline.py "path/to/obj/folder" --interp_from 20 --interp_to 30 --num_interp 10
```

#### 4. 完整示例
```bash
python complete_vv_pipeline.py "D:/VV_Data/sequence1" \
    --start_frame 0 --end_frame 100 \
    --interp_from 25 --interp_to 75 \
    --num_interp 20 \
    --output_dir "D:/Results/sequence1_output"
```

## 🔧 管道步骤

### 步骤1: 帧数据处理
- 加载指定范围的.obj文件
- 每帧通过voxel化预处理
- 使用NeuralMarionette预测24关节skeleton
- 保存处理结果供后续使用

### 步骤2: Rest Pose检测
- 多重指标评估每帧作为rest pose的适合度:
  - 关节-网格对齐度
  - 骨骼长度一致性  
  - 姿态展开度 (T-pose检测)
  - 旋转矩阵规律性
- 自动选择得分最高的帧

### 步骤3: 网格拓扑统一
- 基于骨骼结构进行顶点对应关系分析
- 智能重映射不同顶点数的mesh到统一拓扑
- 保持几何结构的同时统一顶点数量

### 步骤4: DemBones蒙皮计算
- 智能数据子采样以确保DemBones成功运行
- 使用修复后的正确API进行蒙皮权重计算
- 扩展结果到完整分辨率并保存

### 步骤5: 插值生成
- 使用SLERP插值骨骼变换
- 应用线性混合蒙皮(LBS)生成变形
- 保存为.obj格式的插值mesh

## 📁 输出结构

```
<output_dir>/
├── Frame_XXXXX_data.pkl          # 每帧的处理数据
├── skinning_results.pkl          # 蒙皮权重和骨骼变换
└── interpolated_XXX_XXX_XXX.obj  # 插值结果mesh
```

## ⚠️ 注意事项

1. **数据格式**: 输入文件夹应包含编号的.obj文件 (如 frame_001.obj, frame_002.obj)
2. **内存需求**: 大型mesh (>30k顶点) 会自动进行智能子采样
3. **计算时间**: DemBones计算时间取决于顶点数和帧数，可能需要几分钟到几十分钟
4. **GPU要求**: NeuralMarionette需要CUDA支持的GPU

## 🐛 问题排查

### 常见错误
- **"No .obj files found"**: 检查文件夹路径和文件名格式
- **CUDA out of memory**: 减少处理的帧数范围或使用更少顶点的mesh
- **DemBones失败**: 通常会自动重试，如持续失败检查数据质量

### 性能优化
- 对于大型数据集，建议分批处理
- 可以先处理小范围测试效果
- 使用SSD存储可显著提升I/O性能

## 🎯 技术亮点

✅ **修复的DemBones**: 解决了原始API的兼容性问题  
✅ **智能子采样**: 自动处理大型数据集  
✅ **骨骼引导统一**: 解决VV数据拓扑不一致问题  
✅ **自动Rest Pose**: 无需手动选择最佳参考帧  
✅ **高质量插值**: SLERP + LBS生成平滑过渡  

---

**开发者**: GitHub Copilot  
**版本**: 1.0 - 完整集成版本  
**最后更新**: 2025年7月
