目标：实现一套独立的 Python 评估脚本，对比
①「固定起始权重」与 ②「起始+结束权重分段」两种插值结果。
输出 CSV/Markdown 表和可选曲线图，便于写论文。

## 注意：本项目虚拟环境 python 路径
C:\Users\sky\miniconda3\envs\nmario\python.exe

1. 准备阶段
下载 DFAUST Seq “50002_jump” 与 AMASS SMPL meshes（任选 1–2 条长序列作测试）:
- 已下载DFAUST的Registration males and famales 在 `evaluation/data` 下，请解压到 `evaluation/data/dfaust`
- MASS 下载可自行运行
`python amass_download.py --datasets CMU ACCAD BMLrub TotalCapture --out_dir evaluation/data/amass`

写脚本 `generate_keyframe_pairs.py`

每隔 k 帧抽首尾帧，保存 OBJ + skeleton（父子关系、T-pose 骨长）。

记录对应真实中间帧索引，用于后续误差对比。

2. 运行插值产生结果
```bash
    python volumetric_interpolation_pipeline.py  \
        --start_frame startIdx --end_frame endIdx \
        --method baseline         # 固定权
    python volumetric_interpolation_pipeline.py  \
        --start_frame startIdx --end_frame endIdx \
        --method dual_reference   # 分段权
```

输出 interpolated_frame_XXXX.obj。


3. 实现评估脚本 evaluate_interpolation.py
3.1 几何误差（需真实中间帧）
```python
    def chamfer(a_pts, b_pts):
        d_ab = cKDTree(a_pts).query(b_pts)[0]  # to a
        d_ba = cKDTree(b_pts).query(a_pts)[0]  # to b
        return (d_ab.mean() + d_ba.mean()) * 0.5
```
对每个中间帧计算 Chamfer-L2；求均值、方差、最大值。

3.2 法向一致 & ARAP
读取法向，用 `np.einsum('ij,ij->i', n1, n2)` 求夹角。

对每个骨骼 `influence` 区域执行 `rigid_transform = svd_fit()`，取残差作为 ARAP 误差。

3.3 时间平滑度
```python
    vel = (V[t+1] - V[t]) / dt
    acc = (vel[1:] - vel[:-1]) / dt
    jerk = (acc[1:] - acc[:-1]) / dt
    mean_jerk = np.linalg.norm(jerk, axis=-1).mean()
```
3.4 物理/合法性检查
骨长 SD：`np.std(np.linalg.norm(joints[:,child]-joints[:,parent], axis=-1))`

自碰撞：用 `trimesh.collision` 检查每帧自交数。

端效器滑动（若地面 y=0）：`np.sum(np.abs(foot_y[frames_ground] - 0) > ε)`。

4. 结果汇总
对每种方法输出：

`mean_chamfer`, `max_chamfer`, `mean_normal_angle`,

`mean_jerk`, `max_jerk`,

`avg_arap_error`,

`bone_length_sd`,

`self_intersection_count`,

`foot_slide_px`.

保存为 `results.csv` 并绘制 **Chamfer vs 时间** 曲线 (matplotlib)。

5. 对比与结论自动化
```python
    df = pd.read_csv('results.csv')
    improvement = df[df.method=='dual_reference'] - df[df.method=='baseline']
    print(improvement[['mean_chamfer','mean_jerk','avg_arap_error']])
```
若改进值为负，说明分段权重优于固定权。生成 Markdown 报告。

6. 可选扩展
批量跑不同动作，自动汇总平均提升百分比。

提供命令行 `flag --no_gt` 切换到“无 GT”模式，仅算内部指标。

文件结构建议
```bash
    evaluation/
    ├── data/           # 下载的 DFAUST / AMASS
    ├── generate_keyframe_pairs.py
    ├── evaluate_interpolation.py
    ├── utils_mesh.py   # 读取 obj, 计算法向、Chamfer 等
    └── results/
        ├── baseline/
        ├── dual_reference/
        └── results.csv
```

任务: 在 evaluation/ 文件夹下按上文六步实现自动评估脚本，确保：

支持有/无 GT 两种模式；

至少输出 Chamfer、jerk、ARAP、骨长 SD、自碰撞计数等指标；

对比 baseline 与 dual_reference 并生成 results.csv + Markdown 报告；

保证可在 16 GB RAM 单 GPU（或 CPU-only）环境下 1 分钟内完成一次 30 帧测试序列评估。