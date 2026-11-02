# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库是未来聚焦超声治疗设备的"真实场景 / 患者体表三维采集、重建、显示、测量与空间数据基础"。硬件基线：Orbbec Femto Bolt RGB-D、Windows、NVIDIA CUDA GPU、Python。

Femto Bolt → RGB-D 原始数据 → 标定 / Pose / 坐标 → Point Cloud / TSDF / Mesh → 标注 / 测量 / 区域规划（研发演示）→ direct gsplat 3DGS → 查看器。

本仓库不是 Gaussian Splatting Demo。3DGS 只是表现层之一，治疗区域、测量、路径规划都建立在 PointCloud / Mesh / Geometry 层之上。当前不实现真实机械臂控制、ROS、MoveIt、轨迹执行和超声输出；硬件预备迭代新增独立软件模拟任务，不赋予真实执行资格。坐标、单位、Transform、Scene、ROI、Device、Dataset 保留未来接入边界。

`技术路线.md` 是唯一技术基线（Source of Truth）；README、本文件和源码与其冲突时以它为准。它顶部只保留当前状态和稳定约束，完整日期记录按月归档在 [`doc/变更记录/`](doc/变更记录/)（2026-09 与 2026-10 两个文件）；被瘦身的历史技术章节原文在 [`doc/技术资料/技术路线_历史技术章节_2026-09.md`](doc/技术资料/技术路线_历史技术章节_2026-09.md)。写得非常长且逐轮追加，改动前至少读顶部状态段和与本次改动相关的段落。

## 硬性项目规则

来自 `AGENTS.md`，改动前必读：

1. **修改采集、对齐、Mask、关键点、坐标、筛帧、融合、TSDF、Mesh、格式或依赖时，必须同步更新 `技术路线.md` 的参数与状态，并把完整变更追加到 `doc/变更记录/` 当月文件**。这是最容易被忽略的一步。
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

- Git 远端是私有 Gitee `madegao/femtoBolt`。账号和令牌只允许在被忽略的 `git/` 或 Windows 凭据管理器中，不得进入 URL、跟踪文件或文档；提交前排除 venv、`datasets/`、`outputs/`、`photos/`、`Log/`、模型文件和本地设置。
- 另一个 agent（OpenAI Codex CLI）可能与本会话**同时编辑**本仓库（规划/路径/固定 RGB 模块尤其频繁）。开始改动前先看 `git status`、根目录和 `tests/` 最近几分钟的修改时间及 `Get-Process codex`；测试失败若落在正在被并行修改的文件上，先报告，不要抢改。
- 主检出 `E:\测试3D人体关节坐标` 之外还有同级 git worktree（`git worktree list`），其他 Claude 会话可能在别的 worktree 工作；改动和提交只针对当前 worktree 与分支。`datasets/`、`outputs/` 被 git 忽略、每个 worktree 各有一份，marker 里写的是本 worktree 的绝对路径。
- 备份与差异以 Git 为准，不再新建 `outputs/*_backup_*`；遗留备份中的源码不是 live code，搜索时忽略。
- 每轮交付记录写入 `doc/历史开发记录/`，使用说明写入 `doc/使用说明/`；当前状态和稳定约束同步到根目录 `技术路线.md`，完整日期记录追加到 `doc/变更记录/` 对应月份文件。
- 报告口径：软件/合成/历史回放验证与实物验收分开写；开发合规通过 ≠ release_ready。

## 环境

- 三套环境严格隔离：`.venv`（采集/几何/UI/完整测试）、`C:\FemtoRuntimes\88d813bb44d9a282\gsplat`（产品 gsplat/torch/SPZ）、`.venv-3dgs`（已死且禁止产品分发的旧对照）。不得合并或跨环境补依赖。
- 产品 gsplat 必须走 工作区专属 ASCII 环境、CUDA 12.8 和 VS2022 Build Tools；不得联网下载 LPIPS，版本、许可证、来源和 SHA-256 必须同步锁定。
- `.venv` 的 `pyorbbecsdk2 --no-deps` 会让 `pip check` 名义缺少 `opencv-python`，这是预期，不能擅自“修复”。
- 当前 UI 工作区拥有独立 `.venv`；所有采集入口只使用本目录解释器，不再回退同级主检出。不得在主目录环境安装本分支依赖。
- PySide6 6.11.2 与 VTK 9.5.2 均安装在当前工作区独立采集/产品环境。
- 完整环境、权重、SPZ 构建和安装约定见 [`开发环境与架构约定.md`](doc/技术资料/开发环境与架构约定.md)。

## 常用验证命令

每次提交前从仓库根目录执行全部六项；不得跳过、改写或放宽：

```powershell
.venv\Scripts\python.exe tools\check_layering.py
.venv\Scripts\python.exe -X utf8 -c "import importlib,pathlib;ms=[p.with_suffix('').as_posix().replace('/','.') for p in pathlib.Path('femto').rglob('*.py')];[importlib.import_module(m) for m in ms];print(len(ms),'modules OK')"
C:\FemtoRuntimes\88d813bb44d9a282\gsplat\Scripts\python.exe -X utf8 -c "import femto.gsplat.scene_gsplat_trainer,femto.gsplat.scene_gsplat_viewer,femto.gsplat.gsplat_backend,femto.gsplat.scene_single_gs,femto.gsplat.scene_single_gs_viewer,femto.gsplat.scene_gsplat_acceptance;print('gsplat runtime OK')"
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
C:\FemtoRuntimes\88d813bb44d9a282\gsplat\Scripts\python.exe -m unittest discover -s tests -p "test_gsplat_guard.py"
.venv\Scripts\python.exe -X utf8 -c "from pathlib import Path;from commercial_compliance_check import imported_modules,BLOCKED_IMPORT_PREFIXES;bad=[(f,[m for m in imported_modules(f) if m.startswith(BLOCKED_IMPORT_PREFIXES)]) for f in list(Path('femto').rglob('*.py'))+list(Path('tools').glob('*.py'))];print([b for b in bad if b[1]] or 'clean')"
```

- **不要沿用任何旧的全量用例数**。`技术路线.md` 顶部、`doc/历史开发记录/` 和 `doc/项目维护与后续迭代任务清单_20261011.md` 里出现的 842 / 881 / 1520 / 1547 都是各自当时观测条件下的快照，测试树一直在长。引用总数前必须自己重新跑第一、二、四条命令，以当前源码的模块导入和 unittest 日志为准；分层须保持 `upward edges: 0`。
- `.venv` 全量耗时以当前实际日志为准，P4/P5 规划与保存回读包含完整几何复算，不能沿用旧的”约 4 分钟”估计。环境跳过项按最终汇总记录，不增加或隐藏跳过条件。单文件使用 `unittest discover -s tests -p “test_x.py”`，不要用点分模块名。Qt 界面测试走 offscreen，需要桌面的手工验收另行执行。
- `prototype/` 不在上述范围：`.venv\Scripts\python.exe prototype\tests\test_workflow.py` 与 `.venv\Scripts\python.exe prototype\tests\test_visual_contract.py` 必须分两个进程跑（前者建 `QCoreApplication`，后者需要 `QApplication`）。
- **不要随手运行 `commercial_compliance_check.py` 主程序**：它会覆盖 `compliance/current_report.json`。这里只允许第六条的受限 import 干扫描。

## 架构

源码位于 `femto/` 包；入口从仓库根以 `python -m femto.<pkg>.<module>` 运行，数据树路径来自 `femto.paths`。

- 分层由低到高固定为：`femto` → `depth` → `dataset` → `geometry` → `hardware` → `reports` → `annotation` → `tracking` → `temporal` → `local` → `planning` → `viewer` → `pipeline` → `gsplat` → `capture` → `execution` → `app`。
- **只允许上层 import 下层，函数内延迟 import 也算**。新增 import 前必须运行 `tools/check_layering.py` 并保持 `upward edges: 0`；下层需要上层功能时应下移纯函数，不能用延迟 import 绕过。
- `femto.gsplat.*`、viewer 和 single-GS 只能在函数内 `import torch`；`femto.io_utils` 不能顶层 import `cv2`。
- 17 层逐包职责、模块清单和 `tools/` 约定见 [`doc/技术资料/开发环境与架构约定.md`](doc/技术资料/开发环境与架构约定.md)。`execution` 管理模拟任务、检查、协调、资源占用与独立事件库；hardware 不导入 execution。新代码统一绝对导入。

### PySide6 工作台（唯一 GUI）

- 入口 `python -m femto.app.qt_main [--ui-preview] [--project <结果/reconstruction 目录或 .zip/.json 方案>]`，`--ui-preview` 不连相机。`start_femto_app.cmd` 先做 import 检查（失败写 `outputs\qt_startup_check.log`）；pythonw 下 stdout/stderr 进 `Log/qt_workstation_{stdout,stderr}.log`。
- `qt_main.WorkstationWindow` 包含采集、规划、只读回看和“设备与模拟”四页。新增 `qt_execution.ExecutionPage` 使用一个非 daemon 后台所有者串行处理模拟协调与 IO；构造不连接设备、不创建数据库。切换项目及关窗纳入各页 `can_leave()` / `shutdown()`，结束未完成按定时重试等待。标题栏保持“研发预览 · 不可执行”。
- `qt_planning_io` 的共享方案来源组合已下移到 `planning.saved_source`，Qt 和无界面模拟均调用原来源/完整性/目标/覆盖核验。模拟记录单独写 `outputs/history_metadata/execution.sqlite3`，当前写 schema v2、只读 v1/v2；原 `HistoryStore` 独占的 `history.sqlite3` schema v1 保持不变。`app.execution_presentation` 仅转换中文摘要，不决定业务状态；保存回看传递修订与路线摘要，候选核验及冻结由模拟后台所有者执行。
- `qt_capture.CameraHost` 在线程里跑 `camera_worker.CameraWorker` 子类，是相机 SDK 唯一持有者；界面只经 `command_queue` / `event_queue` 通信（如 `send('local_start')`）。
- Tk 已全部移除。`femto_control_app` 只剩后端兼容再导出（`FemtoControlApp(...)` 返回 `WorkstationWindow`，`-m` 启动转发到 `qt_main`），Qt 页面不 import 它；`camera_worker.py` 只保留命令队列、帧处理和后台任务，旧 Tk 事件消费、扫描按钮和重建完成回调已于 2026-09-28 删除。
- 只有 `qt_*` 模块在顶层 import PySide6；viewer、gsplat 查看器、planning 对话框和 local 诊断都在函数内 import（gsplat 运行时没有 PySide6）。样式在 `femto/app/styles/medical.qss` 与 `assets/`。
- `prototype/` 是独立的 PySide6 Mock 界面原型（`python prototype/main.py`，`--replay` 只读回放已有项目），与 `femto` 互不 import，QSS 是独立副本；`prototype/standalone.py` 由 `build_standalone.py` 生成，不要手改。

### 三条采集/建模链

Qt 工作台只暴露第一条。后两条的后端、manifest 口径和测试都还在，但目前没有 Qt 入口：旧 Tk 命令发送与重建完成回调已删除，`CameraWorker` 的 `start_scan` / `temporal_start` 命令后端仍保留；当前工作台没有相应入口，也不会通过旧回调写两类重建 marker。测试经 `tests/camera_worker_fixture` 往命令队列直接发这两个命令。改 manifest、位姿、重建参数、marker 或报告口径时三条都要想清楚。

**局部移动补采（采集工作台）**：`CapturePage.start` → `send('local_start')` → 相机线程 `CameraWorker._cmd_local_start` 构造 `local_capture.LocalCapture` → 每帧 `scene_live_tracker.LiveTracker`（Open3D Hybrid RGB-D 里程计，正反向，1/4 分辨率）→ 首个有效帧定义 **W0**，`T_W0_Ct` 把当前彩色相机 RDF 变到 W0 → 停稳补采接纳视角 → 结束后 `local_offline.LocalOfflineJob` 在 Windows Job Object（`TIME_BUDGET_S=600` / `MEMORY_BUDGET_MB=4096`）子进程 `-m femto.pipeline.local_offline --worker` 里执行 `local_processing`（精配准候选 → 五帧支持 Mask）→ `complete_demo.run_demo(publish_latest=False, publication_id=…)` → `publish_result` 把 `publication.json` 从 `pending` 翻成 `published`，并原子写 `outputs/latest_complete_demo.txt`；只有 `real_femto`、非固定机位且 ≥3 帧时才同时写 `latest_moving_reconstruction.txt`。原始帧写 `datasets/scene_<ts>_local_<8hex>/`（RGB 无损 PNG，`pose_valid=false`），在线 Pose 只在 `derived/live_tracking.jsonl` 和 `derived/local_posed/`。`scene_pose_estimator` 不参与这条链。<12 视角时跳过 Nerfstudio 转换（`demo_summary.nerfstudio=null`），`gsplat_backend` 状态检查明确返回不可训练原因，准备/训练在启动进程前以 ValueError 拒绝，不借用残留转换目录；真实多视角训练仍需合格移动视角和其他正式门禁。

**固定视角关键帧（仅后端）**：相机线程收到 `temporal_start(count∈{1,5,10,15}, mode='quality', build_model=True)` 后填 `TemporalBurst` → 非 daemon 线程 `app_temporal.process_keyframe_burst` → `temporal_keyframe.build_temporal_keyframe`（每帧 `process_depth_quality` → 运动评估 → 置信度加权时域融合）→ `complete_demo.run_demo(outputs_root=<capture>/derived/temporal_fusion/<stamp>/models)`。结果目录在 `datasets/temporal_capture_*/…` 下，**不在 `outputs/`**，原设计由已退役 GUI 回调调用 `reconstruction_sources.remember` 登记到 `outputs/latest_keyframe_reconstruction.txt`；当前 Qt 不提供该流程，也不会自动登记此 marker。派生数据集 `frame_count=1`、`T_WC=I`、`pose_source=single_fixed_camera_keyframe_identity`；`manifest.temporal_fusion` 存在的数据集只能走 `single_frame_quality` 像素网格路径，走 TSDF 会 `ValueError`。这类结果只喂 `scene_single_gs`（单帧固定视角 GS）。

**手持移动多视角（仅后端 / CLI）**：`CameraWorker._cmd_start_scan`（payload `duration_s` / `fps` / `processing_mode`，非 `keyframe` 即移动），或 CLI `start_real_scene_demo.cmd`（`scene_capture --capture-seconds 10 --capture-fps 6` → `complete_demo --mode latest`，不写 `latest_moving_reconstruction.txt`）→ 原始帧 `pose_valid=false`，`derived/capture_quality.json` 记录约每 1 秒一次的运动/清晰度提示（"尚未移动"只表示尚无可靠移动证据，可靠确认后粘滞，不可靠结果不能制造"已移动"）→ `complete_demo.run_demo(mode='latest')` → `estimate_poses`（先背景固定预检，判固定则阻断）→ `derived/posed_rgbd_odometry_*` → TSDF/纹理 → `nerfstudio/` → `outputs/complete_demo_*`。只有这一条链的产物能进多视角 3DGS。

CLI `scene_capture.py`（`OrbbecAlignedCamera` 默认 `near` 档、JPEG）与相机线程采集（`CameraWorker`，`wide_detail`；`start_scan` 会话另写 `parameters.processing_mode`）写出的数据集深度范围不同，下游从 `manifest.camera.depth_range_m` 读取。

### 规划算法链（P0–P6）

这是当前工作树的主要开发面，`femto/planning/` 下 70+ 个模块几乎都服务这条链。README 的"规划能力与完成状态"表是它的对外口径，契约文档在 `doc/技术资料/`。

- **P0/P1 原目标台账与计算修订**：选区像素 → 原目标单元建账（像素裁剪、遮挡判定、子单元生成、深度与质量原因、原目标身份保留）。每次独立保存/回读都**重新建账**，不继承上次结果；上游输入变化会让旧修订失效。契约 [P0_P1_目标台账与计算修订契约](doc/技术资料/P0_P1_目标台账与计算修订契约.md)。
- **P2 名义模型覆盖**（B1 验证 / B2 产品接入）：用名义模型核验覆盖，支持原因定位与同输入优先复证。契约 [P2_B1](doc/技术资料/P2_B1_名义覆盖验证契约.md)、[P2_B2](doc/技术资料/P2_B2_名义覆盖产品接入契约.md)。
- **P3 固定模型补充规划**：只有"原确认集合无损 + 新扫描见证增益达标"才接纳。契约 [P3_补充规划与覆盖增益契约](doc/技术资料/P3_补充规划与覆盖增益契约.md)。
- **P4 严格路径与严格补充**：保守支持判定、覆盖重算、两级恢复。契约 [P4_严格路径构造与支持契约](doc/技术资料/P4_严格路径构造与支持契约.md)。
- **P5 名义变宽模型与变宽补充**：逐段半宽上下界、有效期、独立覆盖与两级恢复。两份契约：[P5_名义变宽模型与回读契约](doc/技术资料/P5_名义变宽模型与回读契约.md)、[P5_变宽补充规划与独立回读契约](doc/技术资料/P5_变宽补充规划与独立回读契约.md)。
- **P6 跨会话继承**：**未开发**。当前保持独立会话，第二级坐标变换与目标对应都不存在。不要照着"下一阶段"的语气写它。

改这条链时必须同时守住的三件事：

1. **算法版本字符串是持久契约**。路线生成是 `quality-constrained-surface-v3`，P5 路线是 `nominal-variable-width-surface-v1`，P5 补充是 `nominal-variable-width-refined-surface-v1`，P3 覆盖分区是 `interleaved-partition-anchor-ball-coverage-v5`。旧版本代码**按版本回放历史证明，不按"已有新版本"删除**——v4 逐叶入口至今保留。
2. **证明资产与预算不能被顺手放宽**。工作台核心 30 s、整个操作 120 s、轨迹 JSON 64 MiB、`target` 建账 45 s / 1,000,000 候选 / 200,000 单元 / 500,000 成对检查 / 512 MiB 资产限额。这些是证明语义的一部分，改之前先开契约文档。
3. **不缓存接纳结果、不跨请求复用几何或覆盖资格**。可以复用的只有"本轮新建、按当前可用内存配置的只读缓冲"（`PreparedResidualEngine`、区间缓冲、四叉系数），且不跨叶/跨请求。`femto.resource_policy` 按实际可用内存决定 `cache_budget_bytes` 和 `candidate_cache_entries`，它**只调节可丢弃缓冲，不调节数值批次、证明顺序或 worker 数**——把资源适配当成放宽预算的借口是错的。

判定语义上最容易搞混的一点：**"未确认"不等于"已覆盖"，也不等于"漏了"**。缺面、无深度、深度冲突、未定位范围一律保持待确认，**不得用补洞或跨块连线伪装完成**；补充规划只能补"未确认"区，且必须证明原确认集合损失为零。`femto.planning.coverage_*` 与 `variable_*` 系列的分工就围绕这个语义展开。

### 离线工具（`tools/`）

`tools/README.md` 有索引。常用的：`check_layering.py`（分层门禁）、`check_workspace_runtime.py`（环境归属）、`check_body_surface_replay.py`、`check_nominal_coverage.py`、`diagnose_route_continuity.py`、`diagnose_local_span.py`、`benchmark_surface_coverage.py`、`benchmark_depth_quality.py`。工具只读或写独立诊断目录，**不得成为运行时代码的下层依赖**（分层检查不覆盖 `tools/`，但要靠自觉）。

### 3DGS 分派

- Qt 工作台目前没有 3DGS 训练、单帧 GS 或 gsplat 查看入口；`camera_worker` 中旧 GUI 的 `latest_moving_result` / `latest_keyframe_result` 状态赋值随废弃回调移除；`femto_control_app` 的 gsplat 再导出暂作为兼容接口保留。
- 多视角训练 CLI：`python -m femto.gsplat.gsplat_backend {status,train,view} --input <结果目录> [--require-real] [--profile low_vram|balanced|quality]`。`train` → `prepare_training_input`（缺 `dataset_split` 的旧转换自动重转到同级 `nerfstudio_strict_<ns>/`）→ `assess_training_input(require_real)`（委托 `scene_gsplat_trainer.assess_training_input`；`require_real` 默认 False，要显式 `--require-real` 才阻断合成/未知来源）→ 子进程 `runtime_python() -m femto.gsplat.scene_gsplat_trainer train …`（`cwd=REPO_ROOT`），stdout 每行一个含 `step` 的 JSON 作为进度；这里只检查运行时 Python 存在，不检查 lpips_ready。`start_direct_gsplat_train.cmd` 直接调训练器，不会自动重转，缺严格分区直接 blocked。
- `runtime_python()` 是 `C:\FemtoRuntimes\88d813bb44d9a282\gsplat\Scripts\python.exe`（回退 `<repo>\.venv-gsplat`）；训练器、`gsplat_backend.open_viewer`、`scene_single_gs_viewer.open_viewer` 和 `start_direct_gsplat_viewer.cmd` 都在这个运行时里跑。
- `scene_3dgs.py` 的旧门禁（读 manifest 跨度、不知道分区）只服务已死的 Splatfacto 路径，不是正式口径。
- 单帧 GS（`python -m femto.gsplat.scene_single_gs build <路径>`）：`scene_single_gs.launch` 在 `.venv` 里 `prepare`（要求来源 `demo_summary.frame_count==1 && fixed_camera`，源 manifest `status=derived_fixed_camera_keyframe`），再子进程在 gsplat 运行时 `train`：冻结几何，只优化 SH0/opacity/scale，500 步、≤300k 点、1920 宽；输出 `outputs/single_gs_*`，marker `outputs/latest_single_gs.txt`；`viewmat=I`、`T_WC=null`，永远不是多视角验收。旧单帧快捷 CMD 已移除，须明确提供固定视角来源；局部补采结果是多视角，会被来源检查阻断。
- torch 延迟导入规则：`femto_control_app` 顶层 import `gsplat_backend`，它又顶层 import `scene_gsplat_trainer`（进而 guard/data/export/metrics），而 `.venv` 没有 torch，因此这些模块及 viewer / single_gs 都只能在函数内 `import torch`。测试里需要 torch 的文件用 `try/except` + `skipIf`。

### 质量门禁（改参数前先看这里）

- 里程计（离线 `scene_pose_estimator` 与在线 `scene_live_tracker` 共用 `moving_quality.POLICY`）：单步位移 ≤0.25 m、旋转 ≤25°、fitness ≥0.15、inlier RMSE ≤0.06 m、正反向一致性 ≤0.01 m / 2°；帧质量有效深度占比 ≥0.20、Laplacian 清晰度 ≥20；连续 5 帧拒绝判丢失；关键帧筛选 ≥0.01 m 或 ≥2°；离线整条轨迹至少 3 帧且跨度 ≥20 mm 或 ≥2°。背景单应性内点比例 <0.85 是"不可靠"而非"已移动"。
- 局部补采：RGB/Depth 同步 ≤5 ms；帧时效/跟踪间隔 ≤2 s；原点 15 s 内建立；会话 ≤300 s；补采请求 3 s 内有效；停稳 ≤3 mm / 1°；视角接纳需 ≥60% 深度一致重叠、≤15% 前方冲突、新样本 ≥64；工作范围半径 ≤0.03 m、总旋转 ≤10°（超出继续跟踪但不接纳）；结束需 ≥3 个接纳视角且最后一帧仍在跟踪；精配准全部视角通过才采用（修正 ≤5 mm / 1°）；五帧支持 Mask 需 5 帧内 1 mm / 0.2° 静止窗口，像素 4/5 帧在 10 mm 内，保留 ≥80% 才应用；网格三角面须三顶点+中心在某原始视角 10 mm 内有支持。ORB+深度+PnP 初值、同会话关键帧锚定和最长 15 s 有证据重定位均是显式实验开关，默认关闭；恢复不改 W0，不跨会话，恢复帧不直接接纳，无回环/位姿图。
- 深度补洞（`DepthQualityConfig` 只能收紧）：无效率 ≤3% 自动修小孔（≤64 px），≤10% 仅局部高置信（≤24 px），>10% 只报告；孔宽高 ≤12、不触边；邻域原始有效 ≥60%、标准差 <30 mm、极差 ≤60 mm；修复像素与原始邻居最大差 ≤50 mm，否则整块回滚；`repaired_confidence` 必须等于 0.30。
- 时域融合：帧数恰为 5/10/15（关键帧链另许 1）；内点 |v−加权中值| ≤30 mm、至少 max(3, ⌈0.6N⌉)、纯度 ≥0.8、MAD ≤15 mm；RGB 背景位移 ≤1.0 px 且每对可靠，否则整段拒绝。
- 3DGS 输入（`scene_gsplat_trainer.assess_training_input`）：≥12 个合格移动视角、非固定机位、位移跨度 ≥0.05 m 或旋转 ≥5°（从实际矩阵算）、`transforms.json` 含严格 `dataset_split`（1-based 每 8 帧第 8 个 test、第 4 个 validation，Seed PLY 只用 train 帧且 SHA 匹配）、几何与转换引用同一 Pose 数据集、`require_real` 时强制 `data_kind=real_femto`。
- 训练器 `PROFILES`：low_vram 6000 步 / downscale 4 / 初值 25k / 上限 400k；balanced 30000 / 1 / 250k / 800k；quality 45000 / 1 / 500k / 1M。所有档位 CUDA reserved ≥5120 MiB 或 free <1536 MiB 时 `ResourceLimitError` 写 `resource_stop_*.json`、保留最近检查点、不发布。验收：留出 PSNR ≥20、SSIM ≥0.65、LPIPS ≤0.40、Depth RMSE ≤0.12 m、覆盖 ≥0.90；评估深度用 `depth_original`（仅原始观测），未渲染像素按 0 m 满误差计。`--resume` 校验数据指纹、downscale/sh_degree/packed、iterations/lr 等超参和点数上限，任一不同即拒绝。
- gsplat 查看器只在 `torch.cuda.OutOfMemoryError` 或消息含 `out of memory` 时逐级降分辨率/临时采样（磁盘 PLY 不变）；其他 CUDA 错误原样抛出。
- TSDF 默认（`complete_demo`/`scene_reconstruction`）：voxel 0.005 m、trunc 0.02 m、image_scale 1.0、stride 1；有补点的帧原始深度积分 10 次 + 仅修复像素 3 次，无补点的帧 1 次；`local_measured` 数据集绝不积分修复像素。
- 深度覆盖 <35% 时结果标 `warning_low_depth_coverage`，不能标 PASS。`reconstruct_scene` / `estimate_poses` / `scene_to_nerfstudio.convert` / `process_depth_quality` 对已存在输出目录 `FileExistsError`，从不覆盖。
- 尺度验证只报 numeric error，未给工程容差时 `accepted=null`；实拍验收 spec 要求 ≥3 次独立采集（按原始 RGB-D 字节指纹去重）才允许 PASS。

### 坐标与单位

- 磁盘约定：`+X` 右、`+Y` 下、`+Z` 前（Orbbec/OpenCV RDF）。Depth PNG 为 `uint16` **毫米**，0 未知；点云、Mesh、关节、标注一律**米**；UI 显示 mm（1 位小数只是格式，不是精度）。
- `T_AB` 把 B 系坐标变到 A 系；`p_W = T_WC · p_C`；TSDF 积分用 `inv(T_WC)`。
- **W0**：局部补采会话首个有效彩色相机的位置与朝向，参考帧 Pose 精确为单位矩阵（atol 1e-9）。`T_W0_Ct` 把当前彩色相机 RDF 变到 W0。最终网格已经在 W0，查看器与标注**不再乘任何相机变换**。`origin_reference.kind` 只有在 `local_geometry.json`、`reconstruction_report.json`、源 manifest、跟踪报告、首帧 Pose 全部一致时才是 `initial_color_camera`，否则是 `model_coordinate_origin / unknown`，固定 RGB 选区（`annotation.fixed_rgb_geometry.FixedRGBModel`）会拒绝加载。这些校验用绝对路径，复制/移动结果目录会让来源降级为 unknown；`.region.zip` 工程原样携带来源不再校验。
- Nerfstudio/OpenGL 导出用 `CV_CAMERA_TO_OPENGL_CAMERA = diag(1,-1,-1,1)` 右乘做 RDF→RUB；`scene_gsplat_data.opengl_c2w_to_rdf` 训练前转回，所以训练后 Gaussian PLY、Seed PLY、SPZ 声明都在 RDF 世界系。区域/路径 GLB 通过场景节点 `diag(1,-1,-1,1)` 转 glTF，PLY/CSV 保持 W0 RDF 米。
- 显示层翻转只作用于副本：`view_latest.py` 翻 Y，`export_preview.py` 对 PLY 乘 `[1,-1,-1]`，`scene_demo_viewer.py` 不翻几何而是把相机 up 设为 `(0,-1,0)`。
- 两套反投影并存：场景链（`femto.geometry.depth_cloud`、`live_point_cloud`、`scene_reconstruction`）用 K6 去畸变射线；人体分支（`pose_geometry` / `fusion` / `top_scan`）用无畸变针孔。同一像素两者点不同，不要混用。
- 时域支持 Mask 在 K6 去畸变**之前**的原始对齐像素格；`scene_to_nerfstudio` 把补洞深度与 confidence 按最近邻重映射到去畸变格。
- 角度显示 `R = Rz(roll)·Ry(yaw)·Rx(pitch)`，奇异时不显示。

### 未训练 vs 已训练 Gaussian

`gaussian_init_untrained.ply` 只是点云生成的 splat 初值，状态固定 `initialization_only_untrained`，gsplat 查看器按文件名/manifest 拒绝它（属性集与训练 PLY 部分重合，不能靠属性判断）。写实效果来自 `outputs/direct_gsplat_*`（多视角）或 `outputs/single_gs_*`（单帧同视角拟合，`multiview_accepted=False`）。

## Windows / 中文路径注意事项

- 中文路径图片 IO 统一走 `io_utils.imread_unicode` / `imwrite_unicode` 或 PIL；Open3D 原生读写走 ASCII 临时目录安全包装。不得复制私有 IO 实现或恢复旧兼容别名。
- CUDA JIT、SPZ、MediaPipe 和 Open3D Filament 的 ASCII 镜像/联接规则不得删除；JSON 统一 `utf-8-sig` 读，既有写入字节格式必须显式保持。
- `commercial_compliance_check.py` 会扫描 `femto/`、`tools/`、`tests/`；BOM 不能隐藏受限 import，`release_ready` 固定为 False。
- 完整路径、日志、编码和运行时细节见 [`doc/技术资料/开发环境与架构约定.md`](doc/技术资料/开发环境与架构约定.md)。

## 目录产物与 marker 文件

- 原始 `datasets/scene_*` 证据不可变；估计位姿、融合、Mesh、标注和规划只能进入 `derived/` 或 `outputs/`，不得覆盖 RGB、Depth、Pose、标定、`frames.jsonl` 或 manifest。
- `datasets/latest_scene.txt`、派生数据集 marker、`outputs/latest_*` 和 `publication.json` 是入口间受控耦合点；改目录或 marker 时必须同步所有生产者、消费者、恢复逻辑和测试。带 `publication_required` 的结果只有有效 `published` 凭证才能恢复。
- 每个产物必须保留来源、坐标系、单位、绑定哈希和数据血缘；失败结果与软删除索引不得伪装成成功模型。
- 完整目录树、文件名、marker 分类与恢复规则见 [`doc/技术资料/数据产物与marker约定.md`](doc/技术资料/数据产物与marker约定.md)。

## 标注、测量与区域规划

- 标注、测距、固定 RGB、区域工程和路径必须绑定 W0 / RDF / 米、源几何 SHA-256、原点与 `data_layers`；绑定变化或来源不一致必须拒绝回读，不能重解释裸 XYZ。
- 这些功能只读原始证据并写派生结果，不得改 RGB、Depth、源网格或 W0。区域三角面、禁区或绑定变化必须使旧路径失效。
- 坐标、测距、覆盖和路线仅供研发可视化；所有相关 JSON 固定 `execution_ready=false`、`physical_accuracy_verified=false`，不得声称临床精度、接触安全或直接驱动设备。
- 完整 schema、固定 RGB 射线规则、预算、区域 ZIP 和算法绑定见 [`doc/技术资料/标注测量与规划约定.md`](doc/技术资料/标注测量与规划约定.md)。

## 测试约定

- 公开接口类型标注使用 `from __future__ import annotations`；签名采用 `Path`、数组、现代容器与可空联合类型，不确定处使用 `Any`，不增加运行时检查或为标注引入重型模块。2026-09-20 任务 D 后，同口径标注为 965/1368（70.54%），零标注文件为 0；详见 `doc/历史开发记录/公开函数类型标注交付记录_20260920.md`。
- `tests/` 没有 `__init__.py`、conftest 或任何 pytest 配置。必须从根目录 `python -m unittest discover -s tests`（不要 `-t .`、不要进 `tests/` 里跑、不要点分模块名）：测试间互相 import 的两种写法并存（`from tests.test_x import …` / `from tests import test_x` 与裸 `from test_x import …`），只有这种跑法两种都能解析。产品代码一律 `from femto.<pkg>.<module> import …`；`mock.patch` 的字符串目标也必须是完整包路径（如 `"femto.temporal.scene_temporal_fusion.imwrite_unicode"`）。
- 8 个跳过是结构性的（`test_gsplat_guard` 6 个 + `test_depth_confidence_integration` 2 个，缺 torch）；`.venv` 里跳过数 ≠8 说明环境变了。注意 `test_qt_capture` 的 `HAVE_QT` 在 PySide6 **或** `femto.app.qt_capture` 导入失败时都会静默跳过 14 项，其余 Qt 测试文件则直接报错。
- Qt 测试文件各自在 import PySide6 之前 `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`（没有共享 helper，新文件照做），用 `QApplication.instance() or QApplication([])` 复用实例；结束时先对页面 `shutdown()`（停定时器、相机线程和 worker）再 `close()` + `processEvents()`。`test_local_comparison` 与 `test_local_offline` 会真正起子进程。
- 驱动相机循环的测试用 `tests/camera_worker_fixture.bare_camera_app()`（`SimpleNamespace` 宿主 + `CameraWorker` 适配），它每次调用都从 `femto.app.femto_control_app` 重新取依赖，所以 patch 那里的 `OrbbecAlignedCamera`（会写数据集的 `start_scan` 路径再 patch `DATASETS_ROOT`）。Qt `CameraHost` 不走这层：它在函数内从 `femto.hardware.orbbec_rgbd` 导入相机，`DATASETS_ROOT` / `OUTPUTS_ROOT` 是 `femto.app.qt_capture` 全局。`Class.__new__(Class)` + 手工设属性构造"裸实例"很常见（`LocalCapture`、`PairEstimator`、`FixedRGBModel` 等，`test_local_capture` 有 `bare()` helper），改构造函数时要同步。
- 共享 fixture：`tests/test_segmented_capture.make_frame/make_camera`、`tests/test_fixed_rgb_experience.make_model`、`tests/camera_worker_fixture.bare_camera_app`、`femto.pipeline.synthetic_scene_demo.generate_dataset`。fixture 一律显式标 synthetic，只有原始 manifest `camera.device` 精确等于 `Orbbec Femto Bolt` 才能得到 `real_femto`。
- `test_scene_gsplat_product` 等在 `.venv` 跑是靠 torch 延迟导入；不要把 torch 提到模块顶层。

## 当前状态（读最新事实请看 `技术路线.md` 顶部）

**先确认自己在哪个工作树。** `git worktree list` 显示四个：主检出 `E:\测试3D人体关节坐标`（`feature/surface-trajectory`）、两个 detached 验证树（`E:\femto-v43-validation-20260923`、`E:\femto-v44-validation-20260923`）、以及本目录 `E:\测试3D人体关节坐标-ui`（`ui/workstation-redesign`）。本文件描述的是**本工作树**；`datasets/`、`outputs/` 被忽略，每个 worktree 各有一份不同内容，marker 里写的是本工作树绝对路径。下面这些是本目录实测值：

- `datasets/`、`outputs/` 内容变化很快（2026-11-01 实测 6 个场景 / 30 项，`latest_scene.txt` 指向 `scene_20261029_174915_local_210269a4`），引用前自己 `Get-ChildItem` 和读 marker 确认。**不要引用主检出或其他 worktree 的目录名当证据**，也不要凭本文件里的名单判断某个目录"已被清理"。
- 真实端到端 3DGS 仍是 BLOCKED：没有通过门禁的真实 ≥12 视角移动采集，也没有实物尺度端点。局部补采结果普遍不足 12 视角，`demo_summary.nerfstudio=null` 属预期。旧文档引用的 2026-09-05 证据目录已经不存在，别把它们当现存证据。
- 合规：`compliance/current_report.json` 开发基线通过、`release_ready=false`，7 条固定的 `manual_release_gate` 警告（权利人、EULA/LGPL、隐私、医疗器械分类等）。PySide6 已登记在 `THIRD_PARTY_NOTICES.md` / `licenses/pyside6/`，但 `commercial_compliance_check.py` 的 `EXPECTED_PACKAGES` / `REQUIRED_LICENSES` 还没收录它。
- **第三条验证命令已实测通过**（`gsplat runtime OK`）。此前记录的“gsplat 运行时没有 PySide6”是**误判**：`C:\FemtoRuntimes\88d813bb44d9a282\gsplat` 里已装 PySide6 6.11.2，`scene_gsplat_viewer.py` 顶层的 `from PySide6.QtCore import Qt` 因此可以保留，**不要**为了"修"这条命令把它改成函数内导入——那只会让导入检查失去意义，两个查看器开窗仍然需要运行时里有 PySide6。依赖已登记在专用清单和安装脚本里，重建环境时照装即可。
- 大量改动尚未提交（2026-11-01 `git status --porcelain` 共 846 条；以当场统计为准）。**不要批量提交**：逐项核对归属，他人或并行 agent 的改动、数据、未跟踪源码都不进提交。Git 成果归类见 [维护与后续迭代任务清单](doc/项目维护与后续迭代任务清单_20261011.md)。

## 文档地图

四个目录职责固定，别混用：`doc/使用说明/` 写面向操作者的当前用法；`doc/技术资料/` 放**契约**（P0–P5 各阶段的证明与格式约定、架构约定、marker 约定、标注规划约定）；`doc/历史开发记录/` 按日期存每轮交付记录（是归档，不是当前事实）；`doc/变更记录/` 按月追加技术路线的完整日期记录。当前待办与废弃边界只在 [doc/项目维护与后续迭代任务清单_20261011.md](doc/项目维护与后续迭代任务清单_20261011.md)。
