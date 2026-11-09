# 对比有序 RGB-D 算法

这个实验入口用于比较积分图法向、投影配准、法向方向采样和深度/法向边缘分类。它生成独立报告和诊断数组，帮助决定是否值得改变当前重建流程。

实验只使用现有 NumPy、OpenCV、Open3D，算法按公开方法独立实现。没有安装或绑定 PCL。当前工作台仍使用原重建与规划算法；实验结果没有进入方案保存或机械臂接口。

## 运行合成数据对照

在项目目录使用本项目虚拟环境。下面命令生成三帧 640×480 曲面 RGB-D，重复三次对照，并细分当前精配准和重建函数的耗时：

```powershell
Set-Location 'E:\测试3D人体关节坐标-ui'
.\.venv\Scripts\python.exe -X utf8 tools\compare_organized_rgbd.py --synthetic --width 640 --height 480 --output outputs\organized_rgbd_review --repeats 3 --profile-current
```

输出目录必须尚不存在。再次测试时更换目录名；程序不会覆盖已有结果。正常完成后，终端输出 `summary.json` 的绝对路径。

默认分析缩放为 0.5，所以此例的算法输入为 320×240。输入 RGB-D 仍为完整的 640×480，原始深度和位姿不修改。生成器保存真实渲染位姿，给后续帧的输入位姿注入 1.5、−0.8、0.5 mm 的三轴平移误差；这属于测试条件，不代表相机误差。

## 对比已有采集

输入应是含 `manifest.json`、`frames.jsonl`、RGB 标定和合法 T_WC 的已接纳帧数据集。实验支持 1–20 帧；单帧只比较法向和边缘，不运行配准。原始深度必须为 uint16 毫米值。

```powershell
.\.venv\Scripts\python.exe -X utf8 tools\compare_organized_rgbd.py --dataset 'POSED_DATASET' --output outputs\organized_rgbd_real_review --repeats 3 --scale 0.5
```

将 `POSED_DATASET` 替换为实际 `local_posed` 数据集目录。输出必须位于输入目录之外。`--scale` 可设为 0.25–1，改变算法对照的采样规模，不覆盖采集证据。`--profile-current` 额外运行现有精配准及全场景重建函数，会生成派生模型和分析文件，但不会发布到用户历史索引；它没有传入涂抹选区，不能代表选区重建耗时。

## 读取报告

| 文件 | 内容 |
|---|---|
| `summary.json` | 重复次数、阶段耗时中位数、法向覆盖率、解析面误差、三种配准的通过/拒绝统计 |
| `run_N/comparison.json` | 每帧来源摘要、参数、每对帧的原始位姿、候选位姿、双向核验和拒绝原因 |
| `run_N/frame_XXXXXX_diagnostics.npz` | 法向、有效性、边缘标签、拟合/检查掩码，仅作诊断 |
| `current_profile/profile.json` | 使用 `--profile-current` 时生成的当前流程分段计时 |
| `current_profile/*.pstats` | Python 分析器补充资料；部分 pybind 调用栈可能缺失，以显式壁钟计时为准 |

每次运行的 `state` 应为 `complete`，`source_unchanged` 应为 `true`。来源文件或诊断数组的 SHA-256 可用于核对本次输入和输出；若来源改变，实验失败，不发布成功结论。

边缘标签为位组合：1 表示有效深度旁的缺失边界，2 表示前景遮挡边，4 表示背景被遮挡边，8 表示相邻法向角度突变。缺失像素本身没有可靠三维位置。8 是角度代理，不是主曲率测量；RGB 颜色边缘没有参与这些几何标签。

法向报告中的 KD 差异只表示两种估计器的差异。`analytic_normal_comparison` 另外使用已知平面、球面及三次独立的 0.2 mm 高斯噪声，给出相对解析真值的角误差和重复噪声下的法向变化。该噪声是实验设定，不是传感器精度指标。

## 理解配准比较

三个实验分支使用同一份标定后的 RGB-D 和初始位姿：

- `baseline_icp`：复用现有多尺度稳健点到平面 ICP。
- `projective`：依据相机内外参投影寻找对应点，检查视野、正深度、深度/空间残差、目标法向、重复像素投票和几何退化。
- `normal_space_icp`：按法向方向桶平衡选点后，调用原 ICP 求解器。只选择实际观测点，不移动或生成点。

为避免拟合与检查相互混入，实验使用互不重叠的 32×32 图像块。拟合法向的窗口不能跨到保留检查块。这个实验划分与产品当前逐点交替划分不同，因此不能直接把实验耗时和通过率当作产品改善量。

所有分支调用现有 `approve`，继续检查双向一致性、修正范围、空间范围、独立样本误差和颜色退化。近似平面与不足观测会拒绝；求解器收敛不等于通过。任意一帧失败时，整组不能采用混合位姿。`would_adopt_all_views` 只表示是否满足实验核验，运行器始终 `published=false`，不会替换原始位姿或工作台模型。

## 判断性能收益

先比较法向有效覆盖率、解析误差和配准拒绝原因，再比较耗时。积分图法向在缺口及遮挡附近会保留未知，不能靠增加有效数量掩盖缺失。

`median_pair_elapsed_s` 包含双向求解与独立核验；拟合法向和法向采样准备开销单独报告，必须一并考虑。重复三次中的六个配准尝试实际来自两对帧，不能理解为六份独立采集。

当前流程的 `wall_stage_totals` 是显式 `perf_counter` 壁钟计时，部分阶段存在包含关系，不能直接相加。它含文件输出及分析器开销，不含相机采集、路线规划或方案保存。一次实验较快不足以修改默认算法，仍需实采曲面、缺失、遮挡和位姿扰动数据复测。

## 参考方法

- [PCL 积分图法向](https://pointclouds.org/documentation/tutorials/normal_estimation_using_integral_images.html)
- [PCL 有序投影对应点](https://pointclouds.org/documentation/classpcl_1_1registration_1_1_correspondence_estimation_organized_projection.html)
- [PCL 法向空间采样](https://pointclouds.org/documentation/classpcl_1_1_normal_space_sampling.html)
- [PCL 有序边缘分类](https://pointclouds.org/documentation/classpcl_1_1_organized_edge_from_r_g_b_normals.html)
