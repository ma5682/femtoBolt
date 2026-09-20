# 真实 Femto 多视角闭环：仅剩的人工步骤

更新：2026-09-08。手持移动采集与后续处理入口已恢复，40 项相关测试及 24 帧未知位姿合成演练通过；真实端到端状态仍为 **BLOCKED / accepted=false**，等待实拍多视角和实物尺度验收。

软件不需要重新安装。相机已实际读取同步 RGB-D（最近四次 smoke 2026-09-05 00:36、01:36、02:59、03:24 通过，manifest 记录 `connection_type=USB2.1`，尚未在 USB3 下复测；最新持久日志 `outputs/final_femto_gui_camera_smoke_20260905_0325.log` 记录 4 次读取、前三次 `align_none` 后成功；2026-09-04 22:25 一次在同样连接下未取到对齐帧，属间歇性失败），direct gsplat/CUDA、离线 LPIPS 和官方 `spz==1.1.0` binding 已通过检查。现有五套移动模式记录的背景位移都未通过门禁，最新实拍也是固定机位；不能继续把这些数据当成真实多视角输入。旧合成训练指标不属于本阶段验收。

## 1. 摆放一次静态场景

- 第一轮选一张有纹理的桌子，上面放纸箱、书本、哑光杯等不同深度的静止物体，旁边保留桌角或椅背作为空间结构。不要选活动人体、纯白墙、玻璃、镜子或大面积黑色吸光物。
- Femto 起点离主要物体约 **0.8–1.2 m**，斜向俯拍桌面，目标与周边背景同时在画面内。
- 在物体旁固定一块哑光硬板，用尺子核实并标记两个相距 **300 mm** 的清晰点；尺子/硬板在整个扫描中不得移动。若实际距离不是 300 mm，记录真实数值，不能按期望值填入。
- 两个标记点必须在 RGB 和有效 Depth 区域内。这个长度只用于几何尺度研发验证，不表示临床或外部设备定位精度。

## 2. 页面采集

1. 先把 Femto 接到 USB3 口，再打开 `start_femto_app.cmd`，关闭其他占用相机的采集程序。四次成功 smoke 的照片 manifest 都记录 `connection_type=USB2.1`；最新一次证实启动期可能连续出现 `align_none` 后恢复，另一次完全失败则没有照片 manifest，失败证据是 `outputs/femto_gui_camera_smoke_20260904_222500.log` 中的 SDK 对齐/配对帧错误。页面现在按真实 15 秒截止时间等待首个完整 aligned RGB-D，并在结果 JSON 中保存读取诊断计数；正式 15 秒 / 6 FPS 采集前仍建议在 `photos/latest_photo.txt` 指向的 `manifest.json` 里核对 `connection_type`。若采集中出现丢帧或无对齐帧，先换口/换线再重采，不要用软件填补。
2. 模式选 **手持移动多视角（场景静止）**，按钮显示“开始移动采集并重建”。原单帧关键帧模式继续保留。
3. 当前深度固定为 **WFOV_UNBINNED / 0.25–2.21 m / 1024×1024@15**，RGB 为 1920×1080 MJPG@15；无需切档。选择移动模式默认时长 **15 秒**、保存 **6 FPS**、目标 **90 帧**。3DGS 档位选 **low_vram**。
4. 确认目标没有过近提示，主要物体和两端标记都有深度返回。Depth=0 保持未知，不用软件填成假深度。
5. 点击当前模式的采集建模按钮。沿桌子一侧缓慢平移约 **30–50 cm**，同时小角度朝向同一组物体，形成约 **20–40°** 的观察变化；每张画面与上一张保持约 **70% 或更多重叠**。
6. 不要只原地转相机，不要遮挡镜头或快速甩动。多帧对背景预检不会仅因结束时回到起点而误判为固定机位，但本轮顺序里程计没有回环，绕回起点不能消除累计漂移，仍不建议在结束时回到与起点完全相同的机位。
7. 采集过程中页面会约每 1 秒用起始 RGB 帧和当前低分辨率 RGB 做“运动辅助提示”，显示 **尚未移动 / 已移动**。初始“尚未移动”只表示尚未取得可靠移动证据，不表示已经得到固定机位 Pose；pending、纹理不足等不可靠结果或 worker 异常都不能把它误报为“已移动”。一旦某次可靠结果确认移动，后续固定或不可靠抽样不会把状态回退，报告也保留与该状态匹配的可靠证据。这是光流+单应性的操作反馈，不是位姿或定位，也不会写入 `T_WC`；相机独占线程只提交抽样 RGB 副本并轮询单线程异步 worker，计算不占住相机读取或改变帧保存速率。
8. 自动停止时若仍没有可靠移动证据、页面显示“尚未移动”，会在开始重建前提示；如需固定相机，可改用 **关键帧建模**。不会自动切换模式，也不会生成单位矩阵伪 Pose。随后仍按原门禁执行，最终是否固定/可估 Pose 由正式背景预检和里程计报告决定，失败证据写入本次派生目录。
9. 自动停止后等待 Pose→验证→重建→转换。原始数据进入 `datasets/scene_YYYYMMDD_HHMMSS/`，Pose 只写该目录下的 `derived/posed_rgbd_odometry_*/`。

期望留下 30–90 个有效移动视角；硬门禁仍为 ≥12 帧、位移跨度 ≥0.05 m 或旋转 ≥5°。里程计单步仍为 ≤0.25 m、≤25°、fitness ≥0.15、RMSE ≤0.06 m；没有为本次放宽。

## 3. 成功后的软件动作

几何重建成功后，页面点击 **训练写实 3DGS**。默认正式链为：

```text
femto_control_app → gsplat_backend → .venv-gsplat/scene_gsplat_trainer
→ outputs/direct_gsplat_*/model/trained_gaussians.ply
→ evaluation.json / acceptance_report.json / model/model_manifest.json
```

也可在项目根目录执行下列命令（`--require-real` 阻断合成/未知来源；仍执行所有原门禁）：

```powershell
.\.venv\Scripts\python.exe gsplat_backend.py train --require-real --profile low_vram --downscale 4 --max-init-points 5000
```

如果页面重建失败，先读它显示的原因，不要直接训练旧结果。需要重跑刚采集的原始数据时执行：

```powershell
.\.venv\Scripts\python.exe complete_demo.py --mode latest --no-viewer
```

这条命令没有合成回退，也没有失败 Pose→单位矩阵回退。完成后再运行上面的训练命令。

## 4. 几何尺度选点

重建完成后，只在融合点云上选择刚才标记的两个端点。运行：

```powershell
.\.venv\Scripts\python.exe scene_scale_validation.py --pick --ground-truth-mm 300 --label ruler_300mm
```

按住 Shift 左键选两个端点，关掉选点窗口。这里唯一需要人工辨认的是实际标记端点，不能由软件猜测。若实体距离不是 300 mm，命令中的数值必须换成尺子实际读数。

输出在该次 `reconstruction/scale_validation_spec.json` 与 `scale_validation.json`。报告保存 `data_kind`、实际几何顶点 World 米制坐标、几何文件 SHA-256、来源、距离和误差；不使用 Gaussian 中心做测量。不规定未经确认的毫米容差，也不自动宣称尺度 PASS。

背景预检报告会保留兼容字段 `median_background_displacement_px`，并额外列出全部相邻保存帧对（另含首末端点对）的索引、位移、跟踪数、单应性内点比例和 `appears_fixed`。单应性共识低于 0.85 的帧对只能标为不可靠，不能误提示“已移动”。因此即使操作者短暂移动后绕回起点，中间移动也不会被首末帧抵消或落入稀疏采样间隙。

随后从主界面打开 **3DGS 验收报告**，它会刷新汇总并关联相同 Pose 数据集的尺度报告。**查看写实 3DGS** 打开专用 gsplat 渲染器；原 Open3D 查看器继续负责 Mesh、点云及轨迹。

## 5. 完成人工采集后给我的信息

只需告诉我“移动采集已完成”和标定两点的真实距离。软件侧我会继续读取：

- `datasets/latest_scene.txt` 指向的原始目录及其 `manifest.json`、`frames.jsonl`、标定；
- 该目录 `derived/` 的 Pose/失败报告和 `latest_posed_dataset.txt`；
- `outputs/latest_complete_demo.txt` 对应的几何、转换与门禁报告；
- `outputs/latest_direct_gsplat.txt` 对应的训练、指标、模型血缘和查看器日志；
- `reconstruction/scale_validation_spec.json`、`scale_validation.json`。

SPZ 只在训练完成后作为可选压缩交付格式；训练主文件仍是完整 PLY。若需要手动复核，可用 `scene_gsplat_export.py`，报告会校验官方回读 Gaussian 数量与 PLY 一致。

本阶段不需要机械臂、ROS、MoveIt 或任何治疗控制。没有真实移动数据、实测尺寸或可靠端点选择时，真实训练/尺度验收继续保持 BLOCKED，不填造数字。
