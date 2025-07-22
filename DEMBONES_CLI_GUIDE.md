# DemBones C++ CLI 版本集成说明

## 概述

我们已经成功将 `complete_vv_pipeline.py` 从 Python 绑定版本的 DemBones 切换到了 C++ CLI 版本。这样可以避免 Python 绑定的各种问题，直接使用官方的 C++ 实现。

## 主要改动

### 1. 导入模块更改
```python
# 之前
import py_dem_bones as pdb

# 现在  
import subprocess
import tempfile
```

### 2. 新增方法

- `_find_demBones_executable()`: 查找 DemBones 可执行文件
- `_write_demBones_input()`: 写入 DemBones 输入文件
- `_read_demBones_output()`: 读取 DemBones 输出文件
- 更新了 `_try_demBones_with_timeout()`: 使用 CLI 版本

### 3. 支持的可执行文件格式

系统会自动查找以下文件：
- `demBones.exe` (Windows)
- `DemBones.exe` (Windows)
- `demBones.bat` (Windows 批处理文件)
- `demBones` (Linux/Mac)

## 安装选项

### 选项1: 使用测试存根（推荐用于测试）
```bash
python setup_dembones.py
# 选择选项 2 创建测试存根
```

### 选项2: 编译真实的 DemBones
```bash
# 1. 安装依赖
# - CMake: https://cmake.org/download/
# - Git: https://git-scm.com/download/  
# - Visual Studio: https://visualstudio.microsoft.com/downloads/

# 2. 下载源码
git clone https://github.com/electronicarts/dem-bones.git

# 3. 编译
cd dem-bones
mkdir build && cd build
cmake ..
cmake --build . --config Release

# 4. 复制可执行文件到 neural_marionette 目录
copy Release/DemBones.exe /path/to/neural_marionette/demBones.exe
```

### 选项3: 直接使用简化蒙皮权重
无需任何安装，系统会自动使用基于距离的简化蒙皮权重算法。

## 测试验证

运行以下命令验证安装：
```bash
python test_dembones_integration.py
```

## 使用方法

一切设置完成后，正常使用 pipeline：
```bash
python complete_vv_pipeline.py <folder_path> --start_frame 0 --end_frame 10 --num_interp 5
```

## 文件格式

### DemBones 输入文件格式
```
# DemBones Input File
nV 8         # 顶点数
nB 2         # 骨骼数  
nF 2         # 动画帧数
nS 1         # subject数量
nIters 5     # 迭代次数
...
parent 0 -1  # 骨骼层次结构
parent 1 0
fStart 0     # 帧起始索引
subjectID 0  # subject ID
v 0.5 0.5 0.5  # rest pose顶点
...
a 0.6 0.5 0.5  # 动画顶点
...
```

### DemBones 输出文件格式
```
# DemBones Output
w 0.8 0.2    # 每行一个顶点的权重
w 0.7 0.3
...
```

## 优势

1. **稳定性**: 直接使用官方 C++ 实现，避免 Python 绑定问题
2. **性能**: C++ 版本通常比 Python 绑定更快
3. **调试**: 更容易调试和诊断问题
4. **灵活性**: 可以自定义输入/输出格式
5. **回退机制**: 如果 DemBones 不可用，自动使用简化算法

## 故障排除

### 问题1: 找不到 DemBones 可执行文件
- 确保可执行文件在当前目录或 PATH 中
- 运行 `python setup_dembones.py` 创建测试存根
- 或者手动编译 DemBones

### 问题2: DemBones 执行失败
- 检查输入数据是否有效
- 查看错误输出信息
- 系统会自动回退到简化蒙皮权重

### 问题3: 输出解析失败
- 系统会自动回退到简化蒙皮权重
- 检查 DemBones 输出格式是否正确

## 下一步

现在你可以正常使用 `complete_vv_pipeline.py` 了！系统会：

1. 尝试使用 DemBones CLI 版本（如果可用）
2. 如果失败，自动使用简化蒙皮权重算法
3. 继续完成整个插值流程

这样既保证了最佳性能（当 DemBones 可用时），又确保了系统的健壮性（当 DemBones 不可用时）。
