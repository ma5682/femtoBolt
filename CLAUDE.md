# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库是未来聚焦超声治疗设备的"真实场景 / 患者体表三维采集、重建、显示、测量与空间数据基础"。硬件基线：Orbbec Femto Bolt RGB-D、Windows、NVIDIA CUDA GPU、Python。

Femto Bolt → RGB-D 原始数据 → 标定 / Pose / 坐标 → Point Cloud / TSDF / Mesh → 标注 / 测量 / 区域规划（研发演示）→ direct gsplat 3DGS → 查看器。

本仓库不是 Gaussian Splatting Demo。3DGS 只是表现层之一，治疗区域、测量、路径规划都建立在 PointCloud / Mesh / Geometry 层之上。当前阶段明确不实现机械臂控制、ROS、MoveIt、轨迹执行和超声治疗控制，但坐标、单位、Transform、Scene、ROI、Device、Dataset 的设计必须为未来接入留边界。

`技术路线.md` 是唯一技术基线（Source of Truth）；README、本文件和源码与其冲突时以它为准。它的顶部是按日期倒序的最新变更，第 13 节是变更记录。

## 硬性项目规则

来自 `AGENTS.md`，改动前必读：

1. **修改采集、对齐、Mask、关键点、坐标、筛帧、融合、TSDF、Mesh、格式或依赖时，必须同步更新 `技术路线.md`**（参数、状态、变更记录）。这是最容易被忽略的一步。
2. **原始采集数据不可变**。`datasets/scene_*/{rgb,depth,pose,calibration,frames.jsonl,manifest.json}` 是证据层；估计位姿、融合、关键帧、点云、Mesh、标注、路径只能写入 `derived/` 或 `outputs/`。
3. **相机移动时禁止写单位外参**。`T_WC=I` 只对物理固定机位成立。位姿不合格时阻断并写质量报告，不得输出伪 Pose，不得用单位矩阵、旧位姿或放宽门限拼接。
4. **完整场景 RGB-D 是主数据**；人体 Pose/Mask/语义是可删除重算的派生层，不得裁剪或覆盖原始画面。
5. **Graphdeco `gaussian-splatting` 源码禁止进入产品树**。正式 3DGS 只走 Apache-2.0 `gsplat` 公共 API。`commercial_compliance_check.py` 会 AST 扫描并阻断 `diff_gaussian_rasterization` / `simple_knn` / `pymeshlab` 导入。
6. 新增第三方依赖先登记 `THIRD_PARTY_NOTICES.md` 和 `licenses/`，否则合规检查失败。
7. 新增 3DGS、SPZ 或 Web 显示时，必须记录训练输入、相机坐标到渲染坐标的轴转换和模型血缘（`model_manifest.json`）。
8. 关键区域坐标、测距、覆盖路线仅供研发可视化，不得声称临床靶点、接触安全或驱动机械臂；所有相关 JSON 固定 `execution_ready=false`、`physical_accuracy_verified=false`。
9. 不引入 ROS、MoveIt、机械臂厂商 SDK、轨迹规划或真实运动控制。
10. 不得将 gsplat / Gaussian Model 设计为唯一 Scene 数据源。
11. 所有三维数据必须明确 coordinate frame、unit、transform direction、source。不允许无法判断属于 Camera / W0 / World / Model 的裸 XYZ。
12. 硬件通过 Adapter / Provider 抽象（`orbbec_rgbd.OrbbecAlignedCamera` 是唯一 SDK 封装）。
13. 离线商业部署约束：不依赖运行时隐式联网下载模型、权重或资源。
14. 新增可分发运行时依赖时必须考虑 LICENSE、版本锁定、来源、SHA-256、是否允许再分发。
15. 失败时的正确行为是**阻断并写报告**，不是降级输出伪结果；数值缺失写 `null`，不沿用合成指标或历史数字。

## 并行开发与工作方式

- 本目录自 2026-09-20 起纳入 Git，远端为私有库 `origin = https://gitee.com/madegao/femtoBolt.git`（分支 `master`）。账号、邮箱、私人令牌与恢复步骤集中在 `git/README.md` 和 `git/gitee_credentials.json`；整个 `git/` 目录被 `.gitignore` 排除，明文令牌只允许存在那里和 Windows 凭据管理器，不得写进 URL、被跟踪的文件或文档。`.gitignore` 还排除三个 venv、`datasets/`、`outputs/`、`photos/`、`Log/`、`*.pt/*.pth/*.spz` 与 `.claude/settings.local.json`；`.gitattributes` 统一以 LF 入库、`.cmd/.ps1` 以 CRLF 检出。
- 另一个 agent（OpenAI Codex CLI）可能与本会话**同时编辑**本仓库（规划/路径/固定 RGB 模块尤其频繁）。开始改动前先看 `git status`、根目录和 `tests/` 最近几分钟的修改时间及 `Get-Process codex`；测试失败若落在正在被并行修改的文件上，先报告，不要抢改。
- 有 Git 之后，修改前的备份与差异以提交历史为准，不再新建 `outputs/*_backup_<stamp>/`。已有的 `outputs/*_backup_*` / `outputs/*_validation/` 目录是 Git 之前的遗留（含 SHA-256 manifest 与 `changes.diff`），里面的源码副本不是 live code，grep 命中它们要忽略。
- 每轮交付记录写入 `doc/历史开发记录/`，使用说明写入 `doc/使用说明/`，并把要点合并进根目录 `技术路线.md` 顶部。
- 报告口径：软件/合成/历史回放验证与实物验收分开写；开发合规通过 ≠ release_ready。

## 环境

三套互相隔离的虚拟环境，**不要合并**：

| 环境 | Python | 用途 | 安装 |
|---|---|---|---|
| `.venv` | 3.12 | 采集、几何、查看器、标注、规划、单元测试。所有非 gsplat 的 `.cmd` 默认用它 | `setup_environment.cmd`（版本硬编码在脚本里，不读 requirements.txt） |
| `.venv-gsplat` | 3.10 x64 | **产品 3DGS 训练/查看运行时**：PyTorch 2.7.1+cu128、`gsplat==1.5.3`、`spz==1.1.0`。没有 open3d/cv2/trimesh，跑不了完整测试 | `setup_direct_gsplat_environment.cmd` |
| `.venv-3dgs` | 3.12（已死） | 内部 Splatfacto 对照。目录存在但无 `ns-train`，安装脚本要求 3.10 且拒绝就地修复；因传递依赖 GPL-3.0 PyMeshLab **禁止随产品分发** | `setup_3dgs_environment.cmd` |

`.venv`：只装 `opencv-contrib-python`，`pyorbbecsdk2` 用 `--no-deps`，因此 `pip check` 会名义性报缺 `opencv-python`，属预期，不要"修复"。`psutil` 由 open3d 传递带入并被 `scene_temporal_fusion.py` 可选使用，但未登记到合规材料。

`.venv-gsplat`：
- 安装脚本建立目录联接 `C:\FemtoGsplatRuntime` → `.venv-gsplat`。所有 gsplat `.cmd` 和 `scene_gsplat_trainer.runtime_python()` 优先走这个 ASCII 路径，因为 CUDA JIT 在中文路径下失败；联接缺失时静默回退到中文路径并报出令人困惑的编译错误。
- 需要 CUDA Toolkit 12.8 `nvcc` + VS2022 C++ Build Tools；`TORCH_CUDA_ARCH_LIST=12.0`（RTX 5060 / sm_120）。
- `patch_gsplat_windows.py` 对已安装 gsplat 1.5.3 的 `cuda/_backend.py` 做精确字符串补丁；重装 gsplat 后必须重跑，版本不是 1.5.3 会拒绝。
- LPIPS 需要 torchvision 官方 AlexNet 权重 `alexnet-owt-7be5be79.pth`。训练器**不会**联网下载：手动下载到 `%USERPROFILE%\Downloads\` 后运行 `install_lpips_weights.cmd`，校验 SHA-256 后放进 torch hub 缓存。缺失或哈希不符时训练在优化前阻断；`--skip-lpips` 仅冒烟且不能验收通过。`TORCH_HOME` 与安装时不同会导致"缺失"。
- SPZ 由 `setup_spz_environment.cmd` 从 `C:\FemtoSpzDeps` 的固定源码缓存构建（tag v3.0.0，zlib 1.3.1、zstd 1.5.6、nanobind 3.0.1），PEP 517 后端未离线锁定，不能称发布级可复现。

版本被 `commercial_compliance_check.py` 的 `EXPECTED_PACKAGES` / `EXPECTED_GSPLAT_PACKAGES` 钉死。升级依赖时要同步改：该字典、`setup_*.cmd` 中的硬编码、`requirements*.txt`、`THIRD_PARTY_NOTICES.md`、`licenses/`。

## 常用命令

全部从仓库根目录运行。

```powershell
# 单元测试（约 700 个用例，unittest；.venv 里固定 8 个因缺 torch 跳过；会弹出真实 Tk 窗口，需交互桌面会话）
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests

# 单个文件 / 单个用例（只用这两种形式；点分模块名 `-m unittest tests.test_x` 会因兄弟 import 风格不一致而 ModuleNotFoundError）
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_local_capture.py" -v
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -k test_known_pose_rgbd_reconstructs_and_converts

# 8 个 torch/gsplat 门控测试只能在产品运行时跑
C:\FemtoGsplatRuntime\Scripts\python.exe -m unittest discover -s tests -p "test_gsplat_guard.py" -v

# 商用合规门禁 → 会无条件覆盖 compliance/current_report.json，不要随手跑
.venv\Scripts\python.exe commercial_compliance_check.py     # 或 .\check_commercial_readiness.cmd

# 不接相机的端到端回归：合成真值 RGB-D → 校验 → TSDF → Nerfstudio → Gaussian 初值（改几何链路后最有价值）
.venv\Scripts\python.exe -m femto.pipeline.complete_demo --mode synthetic --no-viewer
# 已采集真实原始数据的重新处理（失败会阻断，不回退到 synthetic）
.venv\Scripts\python.exe -m femto.pipeline.complete_demo --mode latest --no-viewer

# 相机连通性冒烟测试（拍一张 RGB-D 后退出）
.venv\Scripts\python.exe -m femto.app.femto_control_app --smoke-test

# 数据集门禁 / Nerfstudio 转换 / 实拍验收报告 / 尺度验证
.venv\Scripts\python.exe -m femto.dataset.scene_dataset_validator         # 默认 datasets/latest_scene.txt
.venv\Scripts\python.exe -m femto.pipeline.scene_to_nerfstudio
.\生成实拍验收报告.cmd <结果目录> --spec <field_acceptance_spec.json>
.venv\Scripts\python.exe -m femto.reports.scene_scale_validation --pick --ground-truth-mm 300 --label ruler_300mm

# 正式产品 3DGS（.venv-gsplat，.cmd 会加载 VsDevCmd + CUDA 环境）
.\start_direct_gsplat_status.cmd                            # runtime + LPIPS 权重 + 最近输入是否达标
.\start_direct_gsplat_train.cmd --profile low_vram          # low_vram | balanced | quality
.venv\Scripts\python.exe -m femto.gsplat.gsplat_backend train --require-real --profile low_vram   # UI 等价入口，会自动补做严格分区转换
C:\FemtoGsplatRuntime\Scripts\python.exe -m femto.gsplat.scene_gsplat_export outputs\direct_gsplat_xxx   # 已有 PLY 的官方 SPZ 转换
.\start_direct_gsplat_viewer.cmd outputs\direct_gsplat_xxx --orbit 3 --width 320 --data-kind synthetic
.venv\Scripts\python.exe -m femto.gsplat.scene_gsplat_acceptance

# 分层检查（新增 import 前跑；输出向上依赖边，期望 0）
.venv\Scripts\python.exe tools\check_layering.py

# 离线局部工具（不开相机）
.venv\Scripts\python.exe -m femto.pipeline.local_comparison <local_posed_dataset> <new_dir>      # 同源 A/B/C 几何对照
.venv\Scripts\python.exe -m femto.pipeline.local_interrupted <session_dir> --output <dir>      # 失跟会话的合格前缀重建
.venv\Scripts\python.exe -m femto.tracking.local_tracking_replay <session_dir> --output <dir>  # 只读跟踪回放
.venv\Scripts\python.exe -m femto.pipeline.depth_quality_replay <photo_or_scene_dir>           # 深度补洞回放
.\start_fixed_rgb_viewer.cmd [结果目录|*.region.zip]                            # 固定 RGB 标注 / 规划工作台
.\start_region_planner.cmd                                                       # 选 .region.zip 打开规划工作台
```

需要相机的 GUI 入口：`start_femto_app.cmd`（主界面，`pythonw`）、`start_scene_capture.cmd`、`start_real_scene_demo.cmd`、`start_temporal_fusion.cmd`、`start_top_scan.cmd`。`一键清理历史.cmd` / `清理历史数据.ps1` 把 `datasets/scene_*`、`outputs/complete_demo_*` 及三个 marker 移入回收站（需输入 `DELETE`），不会清 `datasets/latest_scene.txt` 和 `latest_keyframe_reconstruction.txt`，会留下悬空指针。

训练器排错：`set FEMTO_GSPLAT_DEBUG=1` 后 `scene_gsplat_trainer` 打印完整 traceback，否则只输出 `{"status":"blocked","error":...}`。

## 架构

2026-09-20 起源码位于 `femto/` 包（约 130 个模块、16 个子包），根目录只保留 `commercial_compliance_check.py`、`.cmd` 入口、`tests/` 与 `tools/`。**文件基名沿用迁移前的名字**（例如 `scene_capture.py` 现在是 `femto/capture/scene_capture.py`），文档和历史记录里的模块名仍然可以直接对应到文件。所有入口一律 `python -m femto.<pkg>.<module>`，从仓库根目录运行；子进程也全部改为 `-m` 加 `cwd=REPO_ROOT`，没有任何按脚本路径启动的地方。`femto/paths.py` 提供 `REPO_ROOT / DATASETS_ROOT / OUTPUTS_ROOT / PHOTOS_ROOT / MODELS_ROOT / LOG_ROOT`，模块内不要再写 `Path(__file__).resolve().parent` 去找数据树。

子包按依赖层次自下而上排列，**只允许上层 import 下层**（含函数内延迟 import）。`tools/check_layering.py` 按下表顺序检查，当前为 0 条向上依赖；新增 import 前先跑它，若必须让下层调用上层功能，把那段纯函数往下搬而不是加延迟 import：

| 层 | 子包 | 职责 | 主要模块 |
|---|---|---|---|
| 0 | `femto` | 叶子工具 | `io_utils`（唯一的 JSON / JSONL / 图片读写实现；默认 `ensure_ascii=False`，兼容旧文件字节时显式传序列化参数；`cv2` 延迟导入，gsplat 运行时可用）、`ui_i18n`（`tr(language, key)`，只覆盖高级页、Open3D 查看器和 gsplat 查看器；其余窗口用内联 `(zh, en)` 元组，`test_ui_i18n` 要求 zh/en 键集一致）、`paths`（`REPO_ROOT` 等数据树位置与 `child_path`） |
| 1 | `femto.depth` | 逐帧深度质量 | `depth_quality`（保守补洞，`DepthQualityConfig` 只能收紧）、`depth_error_check`、`temporal_depth_fusion`、`temporal_depth_layer`（时域中间层，含 `prepared_depth` / `load_temporal_confidence`）；置信度层 1.0 原始 / 0.3 修复 / 0 未知，每次派生深度都有同名 float32 confidence |
| 2 | `femto.dataset` | 数据集 schema 与门禁 | `scene_dataset_validator`、`scene_data_kind`（只有原始 manifest `camera.device=="Orbbec Femto Bolt"` 才是 `real_femto`）、`local_reference`（`CALIBRATION_FILES`、参考证据）、`calibration_files.write_calibration`、`scene_gsplat_split`（严格 train/val/test 分区）、`single_frame_source.resolve_source`（关键帧模型 / 照片的单帧来源解析） |
| 3 | `femto.geometry` | 反投影与网格原语 | `pose_geometry`（人体分支无畸变针孔）、`depth_cloud`（全项目 K6 反投影的正典实现）、`rgbd_frames`（`distortion_maps` / `prepare_rgbd` / `load_poses`）、`fusion`（Open3D 原生 IO 兜底）、`multiview_texture`、`single_frame_mesh` / `single_frame_quality`、`scene_gaussian`（未训练初值）、`depth_confidence_cloud` |
| 4 | `femto.hardware` | 唯一 SDK 封装 | `orbbec_rgbd.OrbbecAlignedCamera`。`DEPTH_MODE_SPECS` 四档（`wide_detail` 1024×1024@15、`wide`、`near`、`far`）；主界面固定 `APP_DEPTH_MODE="wide_detail"`（0.25–2.21 m）+ RGB 1920×1080；构造默认却是 `near`（500–3860 mm），`scene_capture` / `top_scan` 用的是默认。`AlignedFrame` 数组可能别名 SDK 缓冲，保留必须 copy |
| 5 | `femto.reports` | 研发报告 | `scene_quality_report`（实拍验收 spec）、`scene_scale_validation`（尺度测距，`accepted=null` 除非给容差）、`research_validation`（重复测量记录） |
| 6 | `femto.annotation` | 标注与固定 RGB 几何 | `scene_annotations`（绑定网格 SHA-256 与 W0 来源；`validate_exclusions`）、`scene_annotation_ui`、`scene_surface_picker`、`fixed_rgb_geometry`（K6 去畸变初始照片 → 射线打到 W0 网格）、`measurement_audit`、`region_texture`、`square_brush` |
| 7 | `femto.tracking` | 里程计与运动检查 | `moving_quality`（`POLICY` 离线与在线共用门限、`pose_diversity`）、`scene_pose_estimator`（顺序 Hybrid RGB-D 里程计 + 双向一致性 + 关键帧筛选 + 背景运动预检）、`scene_live_tracker`（在线跟踪、W0）、`local_tracking_*`、`local_frame_selection`（视角接纳 + `local_workspace` 采集包络 30 mm / 10°）、`local_stability*` |
| 8 | `femto.temporal` | 固定机位时域 burst | `scene_temporal_fusion`、`temporal_keyframe`（逐帧深度质量 → 连续 confidence → 加权中值 → 关键帧派生数据集；关键帧按钮走这条链）、`scene_static_fusion` / `scene_keyframe`（只剩测试与 temporal 链调用） |
| 9 | `femto.local` | 局部补采离线阶段 | `local_pose_refinement`（精配准候选）、`local_temporal_support`（五帧支持 Mask）、`local_geometry`（观测支持三角面过滤）、`local_processing`（处理报告 schema 与 HTML）、`local_geometry_diagnostics` |
| 10 | `femto.planning` | 区域规划工作台 | `region_*`、`metric_region_path`、`serpentine_path`、`route_*`、`path_presentation`、`planning_*`、`coverage_*`、`surface_spacing*`，以及宿主窗口 `fixed_rgb_viewer`（也是 `.region.zip` 规划工作台）和 `export_history` |
| 11 | `femto.viewer` | 独立进程的查看器 | `scene_demo_viewer`（Open3D 几何查看器 + 表面选点/测距/圈选）、`live_point_cloud`（实时点云子进程）、`view_latest`、`segmented_cloud_viewer`、`depth_debug_viewer`、`export_preview` |
| 12 | `femto.pipeline` | 端到端编排 | `scene_reconstruction`（K6 去畸变 → 世界变换 → TSDF，补洞帧 10:3 重复积分）、`complete_demo.run_demo`（校验 → 位姿 → 重建 → 转换 → HTML/质量报告 → 查看器）、`synthetic_scene_demo`、`scene_to_nerfstudio`（RDF→RUB 转换、严格分区）、`local_offline`（Windows Job Object 子进程、发布凭证）、`local_comparison` / `local_comparison_viewer`（A/B/C 对照）、`local_parameter_study`、`local_interrupted`、`local_history`（结果目录）、`reconstruction_sources`（关键帧/移动两类成功来源的 marker）、`depth_quality_replay` |
| 13 | `femto.gsplat` | 产品 3DGS | `scene_gsplat_trainer`（`PROFILES`、`assess_training_input`、`runtime_python`）、`scene_gsplat_data` / `_export` / `_metrics` / `_guard` / `_viewer` / `_acceptance`、`gsplat_backend`（UI 跨环境控制）、`scene_single_gs*`（单帧固定视角 GS）、`scene_3dgs`（内部 Splatfacto 对照，不可运行）。这些模块及 viewer 只能在函数内 `import torch` |
| 14 | `femto.capture` | 数据集写入与采集工作线程 | `scene_capture`（`create_session`、`depth_preview`、CLI `near` 档 JPEG 采集）、`scan_session`（`ActiveScan`、`save_scan_frame` / `finalize_scan` / `save_rgbd_photo`、运动提示状态机、`wait_for_aligned_frame`）、`local_raw_writer`、`local_live_preview`、`local_capture.LocalCapture`（局部补采工作线程）、`scene_segmented_capture`（历史四档采集）、`top_scan`（人体 MediaPipe 分支） |
| 15 | `femto.app` | Tk 主界面 | `femto_control_app`（977 行；保留界面编排与兼容转发）、`camera_worker.CameraWorker`（单一 `FemtoCamera` 线程独占 SDK，命令队列 + 事件队列）、`app_model_panel.ModelPanel`（3DGS 训练、查看与验收操作）；`ttk.Notebook` 两页：tab 0 默认为**局部移动补采** `app_local_panel.LocalPanel`，tab 1 为固定采集与高级功能；`SCAN_MODE_KEYS=("keyframe","moving")`，**没有 static 模式**；`scan_session` 的名字仍可从主模块 import，供旧写法的测试使用。另有 `app_local_history`、`app_local_comparison`、`app_temporal` / `app_temporal_panel` |

`tools/` 存放实验与安装脚本（`download_model`、`download_verified_file`、`patch_gsplat_windows`、五个对照实验脚本）、分层检查 `check_layering.py`，以及已执行完毕的迁移脚本 `_migrate_packages.py` / `_restructure_layers.py`；它们在文件头把仓库根目录加入 `sys.path`，以 `python tools\<name>.py` 运行。

### 三条采集/建模链

主界面的选择会一路传播到 manifest、位姿、重建参数、marker 和报告口径，改任一环节都要三条都想清楚。

**局部移动补采（默认页）**：`LocalPanel.start` → 相机线程构造 `local_capture.LocalCapture` → 每帧 `scene_live_tracker.LiveTracker`（Open3D Hybrid RGB-D 里程计，正反向，1/4 分辨率）→ 首个有效帧定义 **W0**，`T_W0_Ct` 把当前彩色相机 RDF 变到 W0 → 停稳补采接纳视角 → 结束后 `local_offline.LocalOfflineJob` 在 Windows Job Object（600 s / 4096 MiB）子进程里执行 `local_processing`（精配准候选 → 五帧支持 Mask）→ `complete_demo.run_demo(publish_latest=False, publication_id=…)` → `publish_result` 把 `publication.json` 从 `pending` 翻成 `published` 并原子写 `outputs/latest_complete_demo.txt` 与 `latest_moving_reconstruction.txt`。原始帧写 `datasets/scene_<ts>_local_<8hex>/`（RGB 无损 PNG，`pose_valid=false`），在线 Pose 只在 `derived/live_tracking.jsonl` 和 `derived/local_posed/`。`scene_pose_estimator` 不参与这条链。<12 视角时跳过 Nerfstudio 转换（`demo_summary.nerfstudio=null`），此时"训练写实 3DGS"按钮会以 `TypeError` 失败，真实多视角训练仍需高级页移动采集。

**固定视角关键帧（高级页 keyframe）**：`toggle_scan` 发 `temporal_start(count∈{1,5,10,15}, mode='quality', build_model=True)` → 相机线程填 `TemporalBurst` → 非 daemon 线程 `process_keyframe_burst` → `temporal_keyframe.build_temporal_keyframe`（每帧 `process_depth_quality` → 运动评估 → 置信度加权时域融合）→ `complete_demo.run_demo(outputs_root=<capture>/derived/temporal_fusion/<stamp>/models)`。结果目录在 `datasets/temporal_capture_*/…` 下，**不在 `outputs/`**，通过 `reconstruction_sources.remember` 登记到 `outputs/latest_keyframe_reconstruction.txt`。派生数据集 `frame_count=1`、`T_WC=I`、`pose_source=single_fixed_camera_keyframe_identity`；`manifest.temporal_fusion` 存在的数据集只能走 `single_frame_quality` 像素网格路径，走 TSDF 会 `ValueError`。这类结果只喂 `scene_single_gs`（单帧固定视角 GS）。

**手持移动多视角（高级页 moving）**：`start_scan(fixed_camera=False)` 默认 15 s / 6 FPS → 原始帧 `pose_valid=false`，`derived/capture_quality.json` 记录约每 1 秒一次的运动/清晰度提示（"尚未移动"只表示尚无可靠移动证据，可靠确认后粘滞，不可靠结果不能制造"已移动"）→ 结束时仍"尚未移动"则弹窗提示但不自动切换 → `complete_demo.run_demo(mode='latest')` → `estimate_poses`（先背景固定预检，判固定则阻断）→ `derived/posed_rgbd_odometry_*` → TSDF/纹理 → `nerfstudio/` → `outputs/complete_demo_*` → `latest_moving_reconstruction.txt`。只有这一条链的产物能进多视角 3DGS。

CLI `scene_capture.py`（`near` 档、JPEG）与 GUI 采集（`wide_detail`、含 `parameters.processing_mode`）写出的数据集深度范围不同，下游从 `manifest.camera.depth_range_m` 读取。

### 3DGS 分派

- 主界面"训练写实 3DGS"只用 `latest_moving_result`；"关键帧 3DGS"只用 `latest_keyframe_result`；模式下拉只决定下一次采集和"查看最新三维模型"打开哪个，不重定向训练来源。
- 链：`femto_control_app._train_3dgs_worker`（兼容转发）→ `app_model_panel.ModelPanel.train_3dgs_worker` → `gsplat_backend.prepare_training_input`（缺 `dataset_split` 的旧转换自动重转到同级 `nerfstudio_strict_<ns>/`）→ `assess_training_input(require_real=True)`（委托 `scene_gsplat_trainer.assess_training_input`）→ runtime ready 且 lpips_ready → 子进程 `C:\FemtoGsplatRuntime\...\python.exe -m femto.gsplat.scene_gsplat_trainer train …`（`cwd=REPO_ROOT`），stdout 每行一个含 `step` 的 JSON 作为进度。CLI 训练器不会自动重转，缺严格分区直接 blocked。
- `scene_3dgs.py` 的旧门禁（读 manifest 跨度、不知道分区）只服务已死的 Splatfacto 路径，不是正式口径。
- 单帧 GS：`scene_single_gs.launch` 在 `.venv` 里 `prepare`（要求来源 `demo_summary.frame_count==1 && fixed_camera`，源 manifest `status=derived_fixed_camera_keyframe`），再子进程在 gsplat 运行时 `train`：冻结几何，只优化 SH0/opacity/scale，500 步、≤300k 点、1920 宽；输出 `outputs/single_gs_*`，marker `outputs/latest_single_gs.txt`；`viewmat=I`、`T_WC=null`，永远不是多视角验收。
- torch 延迟导入规则：`gsplat_backend` 被 `femto_control_app` 顶层 import，而它顶层 import `scene_gsplat_trainer`（进而 guard/data/export/metrics），因此这些模块及 viewer / single_gs 都只能在函数内 `import torch`。测试里需要 torch 的文件用 `try/except` + `skipIf`。

### 质量门禁（改参数前先看这里）

- 里程计（离线 `scene_pose_estimator` 与在线 `scene_live_tracker` 共用 `moving_quality.POLICY`）：单步位移 ≤0.25 m、旋转 ≤25°、fitness ≥0.15、inlier RMSE ≤0.06 m、正反向一致性 ≤0.01 m / 2°；帧质量有效深度占比 ≥0.20、Laplacian 清晰度 ≥20；连续 5 帧拒绝判丢失；关键帧筛选 ≥0.01 m 或 ≥2°；离线整条轨迹至少 3 帧且跨度 ≥20 mm 或 ≥2°。背景单应性内点比例 <0.85 是"不可靠"而非"已移动"。
- 局部补采：RGB/Depth 同步 ≤5 ms；帧时效/跟踪间隔 ≤2 s；原点 15 s 内建立；会话 ≤300 s；补采请求 3 s 内有效；停稳 ≤3 mm / 1°；视角接纳需 ≥60% 深度一致重叠、≤15% 前方冲突、新样本 ≥64；工作范围半径 ≤0.03 m、总旋转 ≤10°（超出继续跟踪但不接纳）；结束需 ≥3 个接纳视角且最后一帧仍在跟踪；精配准全部视角通过才采用（修正 ≤5 mm / 1°）；五帧支持 Mask 需 5 帧内 1 mm / 0.2° 静止窗口，像素 4/5 帧在 10 mm 内，保留 ≥80% 才应用；网格三角面须三顶点+中心在某原始视角 10 mm 内有支持。ORB+深度+PnP 初值、同会话关键帧锚定和最长 15 s 有证据重定位均是显式实验开关，默认关闭；恢复不改 W0，不跨会话，恢复帧不直接接纳，无回环/位姿图。
- 深度补洞（`DepthQualityConfig` 只能收紧）：无效率 ≤3% 自动修小孔（≤64 px），≤10% 仅局部高置信（≤24 px），>10% 只报告；孔宽高 ≤12、不触边；邻域原始有效 ≥60%、标准差 <30 mm、极差 ≤60 mm；修复像素与原始邻居最大差 ≤50 mm，否则整块回滚；`repaired_confidence` 必须等于 0.30。
- 时域融合：帧数恰为 5/10/15（关键帧链另许 1）；内点 |v−加权中值| ≤30 mm、至少 max(3, ⌈0.6N⌉)、纯度 ≥0.8、MAD ≤15 mm；RGB 背景位移 ≤1.0 px 且每对可靠，否则整段拒绝。
- 3DGS 输入（`scene_gsplat_trainer.assess_training_input`）：≥12 个合格移动视角、非固定机位、位移跨度 ≥0.05 m 或旋转 ≥5°（从实际矩阵算）、`transforms.json` 含严格 `dataset_split`（1-based 每 8 帧第 8 个 test、第 4 个 validation，Seed PLY 只用 train 帧且 SHA 匹配）、几何与转换引用同一 Pose 数据集、UI 强制 `data_kind=real_femto`。
- 训练器 `PROFILES`：low_vram 6000 步 / downscale 4 / 初值 25k / 上限 400k；balanced 30000 / 1 / 250k / 800k；quality 45000 / 1 / 500k / 1M。所有档位 CUDA reserved ≥5120 MiB 或 free <1536 MiB 时 `ResourceLimitError` 写 `resource_stop_*.json`、保留最近检查点、不发布。验收：留出 PSNR ≥20、SSIM ≥0.65、LPIPS ≤0.40、Depth RMSE ≤0.12 m、覆盖 ≥0.90；评估深度用 `depth_original`（仅原始观测），未渲染像素按 0 m 满误差计。`--resume` 校验数据指纹、downscale/sh_degree/packed、iterations/lr 等超参和点数上限，任一不同即拒绝。
- gsplat 查看器只在 `torch.cuda.OutOfMemoryError` 或消息含 `out of memory` 时逐级降分辨率/临时采样（磁盘 PLY 不变）；其他 CUDA 错误原样抛出。
- TSDF 默认（`complete_demo`/`scene_reconstruction`）：voxel 0.005 m、trunc 0.02 m、image_scale 1.0、stride 1；有补点的帧原始深度积分 10 次 + 仅修复像素 3 次，无补点的帧 1 次；`local_measured` 数据集绝不积分修复像素。
- 深度覆盖 <35% 时结果标 `warning_low_depth_coverage`，不能标 PASS。`reconstruct_scene` / `estimate_poses` / `scene_to_nerfstudio.convert` / `process_depth_quality` 对已存在输出目录 `FileExistsError`，从不覆盖。
- 尺度验证只报 numeric error，未给工程容差时 `accepted=null`；实拍验收 spec 要求 ≥3 次独立采集（按原始 RGB-D 字节指纹去重）才允许 PASS。

### 坐标与单位

- 磁盘约定：`+X` 右、`+Y` 下、`+Z` 前（Orbbec/OpenCV RDF）。Depth PNG 为 `uint16` **毫米**，0 未知；点云、Mesh、关节、标注一律**米**；UI 显示 mm（1 位小数只是格式，不是精度）。
- `T_AB` 把 B 系坐标变到 A 系；`p_W = T_WC · p_C`；TSDF 积分用 `inv(T_WC)`。
- **W0**：局部补采会话首个有效彩色相机的位置与朝向，参考帧 Pose 精确为单位矩阵（atol 1e-9）。`T_W0_Ct` 把当前彩色相机 RDF 变到 W0。最终网格已经在 W0，查看器与标注**不再乘任何相机变换**。`origin_reference.kind` 只有在 `local_geometry.json`、`reconstruction_report.json`、源 manifest、跟踪报告、首帧 Pose 全部一致时才是 `initial_color_camera`，否则是 `model_coordinate_origin / unknown`，固定 RGB 窗口会拒绝打开。这些校验用绝对路径，复制/移动结果目录会让来源降级为 unknown；`.region.zip` 工程原样携带来源不再校验。
- Nerfstudio/OpenGL 导出用 `CV_CAMERA_TO_OPENGL_CAMERA = diag(1,-1,-1,1)` 右乘做 RDF→RUB；`scene_gsplat_data.opengl_c2w_to_rdf` 训练前转回，所以训练后 Gaussian PLY、Seed PLY、SPZ 声明都在 RDF 世界系。区域/路径 GLB 通过场景节点 `diag(1,-1,-1,1)` 转 glTF，PLY/CSV 保持 W0 RDF 米。
- 显示层翻转只作用于副本：`view_latest.py` 翻 Y，`export_preview.py` 对 PLY 乘 `[1,-1,-1]`，`scene_demo_viewer.py` 不翻几何而是把相机 up 设为 `(0,-1,0)`。
- 两套反投影并存：场景链（`femto.geometry.depth_cloud`、`live_point_cloud`、`scene_reconstruction`）用 K6 去畸变射线；人体分支（`pose_geometry` / `fusion` / `top_scan`）用无畸变针孔。同一像素两者点不同，不要混用。
- 时域支持 Mask 在 K6 去畸变**之前**的原始对齐像素格；`scene_to_nerfstudio` 把补洞深度与 confidence 按最近邻重映射到去畸变格。
- 角度显示 `R = Rz(roll)·Ry(yaw)·Rx(pitch)`，奇异时不显示。

### 未训练 vs 已训练 Gaussian

`gaussian_init_untrained.ply` 只是点云生成的 splat 初值，状态固定 `initialization_only_untrained`，gsplat 查看器按文件名/manifest 拒绝它（属性集与训练 PLY 部分重合，不能靠属性判断）。写实效果来自 `outputs/direct_gsplat_*`（多视角）或 `outputs/single_gs_*`（单帧同视角拟合，`multiview_accepted=False`）。

## Windows / 中文路径注意事项

仓库路径含中文，新代码沿用同样写法：

- **不要用 `cv2.imread` / `cv2.imwrite`**。统一用 `io_utils.imread_unicode` / `imwrite_unicode`（`np.fromfile` + `cv2.imdecode` / `imencode` + `write_bytes`），或 PIL。不要再在模块里复制私有实现，也不要为公共 IO / 几何函数创建 `_read_image`、`json_dump` 等兼容别名；调用点和测试 patch 目标都直接使用公开名。
- **Open3D 原生读写**走 `fusion.write_point_cloud_safe` / `read_geometry_safe`，先落到 `%LOCALAPPDATA%\FemtoPatientPose\native_io` 的 ASCII 临时路径再搬运。`fusion.py` 在 import 时就读 `os.environ["LOCALAPPDATA"]`，无该变量的环境 import 即崩。
- 带 `confidence` 属性的自定义 PLY（`depth_confidence_cloud`）被 Open3D 重读时会丢掉该属性，下游融合必须读 depth + `.npy`，不能读回 PLY。
- **Open3D Filament 资源**在中文路径下加载失败；`scene_demo_viewer._open3d_ascii_resources()` 镜像 wheel 资源到 `%LOCALAPPDATA%\FemtoSceneDemo\open3d_resources_*`，`live_point_cloud` 子进程也依赖它。
- **CUDA JIT / SPZ C++ reader** 怕中文路径：gsplat 走 `C:\FemtoGsplatRuntime` 联接，SPZ 导出先复制到 ASCII 临时目录（回退 `C:/FemtoSpzTmp`）。MediaPipe 模型复制到 `%LOCALAPPDATA%\FemtoPatientPose\models`。
- `start_femto_app.cmd` 用 `pythonw.exe`，`femto_control_app.main()` 仅在流为 None 时把 stdout/stderr 追加到 `Log/femto_control_*.log`；GUI 崩溃先看 `Log/`（还有 `OrbbecSDK.log.txt`、`live_point_cloud.log`、`export_preview.log`）。
- JSON 新写入默认 `ensure_ascii=False`，既有产物必须通过 `io_utils` 的显式参数保留原转义、缩进和 JSONL 分隔符字节；统一用 `encoding="utf-8-sig"` 读（部分 marker 带 BOM）。`.cmd` 里的中文提示需要 `chcp 65001`；gsplat `.cmd` 另设 `PYTHONUTF8=1`、`VSLANG=1033`。
- `commercial_compliance_check.py` 的 `EXCLUDED_SCAN_DIRS` 含三个 venv 及 `outputs`、`datasets`、`references`、`__pycache__`；`femto/`、`tools/` 和 `tests/` 都会被扫描。AST 扫描用 `ast.parse(read_bytes())`，BOM 不能藏受限导入。`release_ready` 是常量 False；缺 SPZ zlib DLL 是无条件 error，无 SPZ 运行时的机器过不了开发基线。

## 目录产物与 marker 文件

这些文本 marker 是各入口之间的唯一耦合点，改输出结构时要一起改：

- `datasets/scene_<ts>/`（高级页/CLI 采集）、`datasets/scene_<ts>_local_<8hex>/`（局部补采）、`datasets/temporal_capture_<ts>/`（时域 burst，`raw/frame_NNN_*.png`，无 `pose/`，校验器会报缺文件）。`datasets/latest_scene.txt` 由 `finalize_scan` 写到 **`session.parent`**（曾经误写全局索引），每次结束含失败会话都会更新。
- `datasets/scene_*/derived/`：`posed_rgbd_odometry_*`、`fixed_keyframe_*`、`local_posed/`、`processing_<8hex>/`、`live_tracking.jsonl`、`local_reference.json`、`local_capture_report.json`、`local_failure.json`、`live_preview_report.json`、`capture_quality.json`、`temporal_fusion_<ts>/`（独立时域）、`temporal_fusion/<ts>/{keyframe,models}/`（一键关键帧），以及 `latest_keyframe_dataset.txt` / `latest_fixed_dataset.txt` / `latest_posed_dataset.txt`。
- `outputs/complete_demo_*/`：`demo_summary.json`（`data_kind`、`fixed_camera`、`frame_count`、`nerfstudio`、`publication_*`、`local_measured`、`nerfstudio_skipped_reason`）、`publication.json`、`demo_report.html`、`result_quality.html/json`、`field_acceptance_spec.json`、`processing_report.json/html`、`reconstruction/`（`scene_tsdf_mesh_m.ply/.glb/.obj`、`texture_multiview.png` + `mesh_triangle_uv.npy`、`scene_fused_points_m.ply`、`initial_observed_points_m.ply`、`local_geometry.json`、`reconstruction_report.json`、`gaussian/`、`derived/depth_quality/`）、`nerfstudio/` 或 `nerfstudio_strict_<ns>/`。`reconstruction/` 下还会累积永不覆盖的 `annotations/<stamp>_<uuid8>/`、`measurement_validation/`、`region_paths/path_*/`、`region_projects/*.region.zip`、`region_project_history/`。
- `outputs/latest_complete_demo.txt`（→ `…/reconstruction`）、`outputs/latest_moving_reconstruction.txt` 与 `latest_keyframe_reconstruction.txt`（→ complete_demo 根，仅 `real_femto` 且发布凭证有效）、`outputs/latest_direct_gsplat.txt`、`outputs/latest_single_gs.txt`、`outputs/local_history_deleted.json`（历史窗口软删除索引，只隐藏不删文件）。
- `outputs/direct_gsplat_*/`（`training_report.json`、`acceptance_report.json`、`checkpoints/latest.pt`、`model/{trained_gaussians.ply,.spz,model_manifest.json}`、`viewer_stdout.log`、`orbit_render/`）、`outputs/single_gs_*/`、`outputs/local_comparisons/comparison_*/`、`outputs/region_workspaces/<uuid>/`（打开 `.region.zip` 的工作副本）、`outputs/planning_drafts/`。
- `photos/femto_photo_*/`（`femto-rgbd-photo-v1`，含 `derived/depth_quality/`）、`photos/latest_photo.txt`；`outputs/patient_top_*/`、`outputs/latest_session.txt`（人体分支）。
- `compliance/current_report.json`（只有合规脚本写，无代码读）、`Log/`、`%LOCALAPPDATA%\FemtoSceneDemo\ui_preferences.json`（语言偏好）。
- 训练器 `--input` 既可指向 `nerfstudio/` 也可指向 complete demo 根目录（通过 `demo_summary.nerfstudio` 解析）；不传则取 `latest_complete_demo.txt`。

`reconstruction_sources.mode_of` 分类规则：`keyframe` = `real_femto` + `fixed_camera True` + `frame_count 1` + 单帧来源可解析；`moving` = `fixed_camera False` + `frame_count>1` + （`nerfstudio/transforms.json` 存在 或 局部 <12 视角跳过标记）；带 `publication_required` 的结果必须 `published`。启动时 `restore()` 扫描 `outputs/*/demo_summary.json` 重建，不信任目录名。

## 标注、测量与区域规划

- `annotations.json` `version=1`：绑定源网格绝对路径、几何 SHA-256、`world_rdf`/`metre`、`local_context`、`origin_reference`、`data_layers`（`femto-data-layers-v1`，按字典相等作为绑定键，改它会让所有旧标注/路径/记录不可回读）。每次保存新建目录；回读时 `distance_to_origin_m` 与两点距离总是从 XYZ 重算，磁盘不回写。区域以 `triangle_indices` 为准，边界每次重算，孔洞保留，`exclusion_polygons` ≤32 个、每个 3–512 角点。
- 固定 RGB 窗口（`fixed_rgb_viewer`）：K6 去畸变初始照片 → 射线打到 W0 网格首次命中；参考深度在量程内时要求 |z_hit − depth| ≤0.01 m 否则 `depth_conflict`；无参考深度但有网格命中记为 `new_support`（青色，不是覆盖率证明）。单 worker、120 s 预算、代际计数丢弃迟到结果。它也是 `.region.zip` 规划工作台的宿主（传文件而非目录时启用规划页签）。
- 区域规划：`region_project.save_project/open_project` 原子写 ZIP（`femto-region-project-v1`，≤1 GiB，逐文件 SHA），工作副本在 `outputs/region_workspaces/`；路线算法 `region_path.raster_path`（像素行）、`metric_region_path`（网格边最短距离等值线，≈5 mm）、`serpentine_path`（水平蛇形 v1/v2）、`region_coverage`；结果带 `plan_binding`（`femto-plan-binding-v1`：区域指纹、几何 SHA、参数、算法版本、结果摘要），`planning_state.state()` 决定按钮可用；名称/备注不进区域指纹，禁区或三角面变化使旧路径失效；采样预算 200000 点。`route_motion` 只是按弦长恒速的播放演示。
- `measurement_audit`：手工尺量记录（`femto-manual-distance-test-v1`）与只读证据审计，`physical_accuracy_verified` 恒为 False。

## 测试约定

- `tests/` 没有 `__init__.py`，也没有 conftest。必须从根目录 `python -m unittest discover -s tests`（不要 `-t .`、不要进 `tests/` 里跑、不要点分模块名）：7 个文件用 `from tests.test_x import …`，5 个用裸 `from test_x import …`，只有这种跑法两种都能解析。产品代码一律 `from femto.<pkg>.<module> import …`；`mock.patch` 的字符串目标也必须是完整包路径（如 `"femto.temporal.scene_temporal_fusion.imwrite_unicode"`）。`outputs/*_validation/run_regression.py` 是交付轮留下的子集跑法（同时把根目录和 `tests/` 塞进 sys.path）。
- 8 个跳过是结构性的（`test_gsplat_guard` 6 个 + `test_depth_confidence_integration` 2 个，缺 torch）；`.venv` 里跳过数 ≠8 说明环境变了。
- 大量测试会打开真实 Tk 窗口并按 1120×720 / 1000×720 / 760×520 / 600×400 断言布局，不要在无桌面会话跑，跑时别碰鼠标键盘；`test_local_comparison` 与 `test_planning_recovery` 会真正起子进程。Tk 测试结束前要 `after_cancel` 所有定时器再 `destroy` + `gc.collect()`。
- 驱动 `_camera_worker` 的测试必须 patch `femto.app.femto_control_app.OrbbecAlignedCamera` 与 `DATASETS_ROOT`；`Class.__new__(Class)` + 手工设属性构造"裸实例"是常见写法，改构造函数时要同步测试里的 `bare()` helper。
- 共享 fixture：`tests/test_segmented_capture.make_frame/make_camera`、`tests/test_fixed_rgb_experience.make_model`、`tests/test_scene_annotation_origin.local_evidence`、`femto.pipeline.synthetic_scene_demo.generate_dataset`。fixture 一律显式标 synthetic，只有原始 manifest `camera.device` 精确等于 `Orbbec Femto Bolt` 才能得到 `real_femto`。
- `test_scene_gsplat_product` 等在 `.venv` 跑是靠 torch 延迟导入；不要把 torch 提到模块顶层。

## 当前状态（读最新事实请看 `技术路线.md` 顶部）

- 真实端到端 3DGS 仍为 BLOCKED：没有通过门禁的真实 ≥12 视角移动采集，也没有实物尺度端点。旧 CLAUDE.md、`doc/历史开发记录/REAL_FEMTO_NEXT_STEPS.md` 和 `THIRD_PARTY_NOTICES.md` 引用的 2026-09-05 证据目录（`outputs/complete_demo_20260905_*`、`direct_gsplat_*`、`final_*.log`、`photos/femto_photo_20260905_*`、`datasets/scene_20260903_145140`）已被清理，`outputs/` 目前只有 2026-09-16/17 的局部补采结果与规划验证目录，不要把它们当成现存证据。
- 最近成功发布的局部结果是 `outputs/complete_demo_20260917_102931`（3 视角、`real_femto`、已 published，无 `nerfstudio/`）；`latest_scene.txt` 指向对应 `scene_20260917_102907_local_3e8ddeec`。
- 合规：`compliance/current_report.json` 开发基线通过、`release_ready=false`，7 条固定的 `manual_release_gate` 警告（权利人、EULA/LGPL、隐私、医疗器械分类等）。
- 2026-09-16 记录的完整回归为 501 项（493 通过 / 8 跳过）；2026-09-17 本地实跑 597 项，失败只落在 Codex 当时正在改的规划/固定 RGB 文件上。引用总数前先重新跑一遍。
