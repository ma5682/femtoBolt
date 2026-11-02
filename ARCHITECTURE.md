# Architecture Walkthrough

分析日期：2026-10-20；2026-10-21 同步设备模拟、SQLite 与第 17 层。范围：当前 UI 工作树（分支 ui/workstation-redesign），含未提交源码。方法：入口、分层、调用链与定向测试核对。本文件描述**当前实现**，不是目标蓝图。

走查对象：开发、管理者、后续 AI agent 三者。主线是医师侧「部位与范围 → 局部采集 → 检查模型 → 标记区域 → 生成路线 → 核对保存」。3DGS、固定视角关键帧和手持多视角 CLI 作为旁路记录。

## 1. Executive Summary

这是一个 Windows 本地 RGB-D 采集、表面建模与体表路线规划工作台。操作者通过 PySide6 连接 Orbbec Femto Bolt，在真实参考图上标记工作区域，生成、核验并保存三维表面路线。结果只供研发可视化：相关 JSON 固定 `execution_ready=false`、`physical_accuracy_verified=false`。

最重要的三个关系：

1. 原始 RGB-D 是不可变证据；位姿、网格、选区、路线和覆盖是派生层，只能写各场景的 derived 目录或 outputs。
2. 同会话阶段检查可以继续补采，但阶段模型不能进入规划；规划要求已发布的最终模型，并且操作者确认同一模型版本。确认不把待确认像素改成已支持。
3. 表面路线不是机械臂轨迹。原目标请求、可规划表面和已证明覆盖集合分别管理；「未确认」不等于已覆盖，也不等于漏扫。

对后续 AI 改动最有帮助的已有护栏：17 层单向依赖、算法版本字符串作为持久契约、保存/回读完整复算，以及采集阶段与最终确认的定向测试。工作台已有采集、规划、回看及设备与模拟四页；工作树的大量未提交并行改动仍须逐项核对归属。

## 2. Repository Purpose

系统类型是**桌面应用 + 本地文件数据管线**，不是网络服务或数据库应用。业务能力是：为未来聚焦超声治疗设备准备患者体表三维采集、重建、测量与空间数据基础。当前阶段明确不实现机械臂控制、ROS、MoveIt、轨迹执行和超声治疗控制。

主要使用者：现场操作者（采集与规划）、研发（几何/覆盖算法、gsplat 表现层）、后续维护与 AI agent。下游系统目前就是本机文件：原始场景、重建结果、区域 ZIP / 便携 project.json、覆盖证书。

硬约束来自 `AGENTS.md` 与 `技术路线.md`：相机移动时禁止单位外参；Graphdeco `gaussian-splatting` 不得进入产品树；正式 3DGS 只走 Apache-2.0 `gsplat`；失败时阻断并写报告，不得输出伪 Pose 或伪完成。

容易混淆的边界：

- Qt 工作台只暴露局部移动补采。固定视角关键帧和手持多视角的后端仍在，但没有正式 Qt 入口。
- `prototype/` 是独立 Mock UI，不导入 `femto/`，不能当作产品功能已实现。
- 人体 Pose / Mask / 部位标签是导航与关联用的派生层，不是工作区域或机器人坐标。

## 3. Audience Guide

### For New Developers

先读 `技术路线.md` 顶部状态，再读 `README.md` 的启动与流程表，然后打开 `femto/app/qt_main.py`。不要从 `femto/planning/` 的 80 多个文件逐个读起。

采集从 `femto/app/qt_capture.py` 的 `CameraHost.send('local_start')` 跟到 `femto/capture/local_capture.py`。规划从 `femto/annotation/fixed_rgb_geometry.py` 跟到 `femto/planning/surface_workspace.py`。测试从仓库根目录 `python -m unittest discover -s tests`，不要进 `tests/` 里跑，也不要用点分模块名。

先忽略：`doc/历史开发记录/`（归档不是当前事实）、`prototype/standalone.py`（生成副本）、被 git 忽略的 `datasets/` 与 `outputs/`（每个 worktree 各有一份）、旧测试总数快照。

### For Managers and Business Stakeholders

价值：把「看见患者体表」变成可保存、可回放、可核验的表面路线方案，为后续设备接入留坐标与证据边界。当前交付是研发预览，不是可执行治疗方案，也不是已通过实物尺度验收的产品。

所有权：源码在本私有 Git 仓库；真实数据在本机被忽略的目录中。合规开发基线可通过，但 `release_ready=false`（权利人、EULA/LGPL、隐私、医疗器械分类等人工门禁未关）。

主要风险：现场同会话补采到最终确认/保存的闭环仍不完整；真实多视角 3DGS 仍 BLOCKED；工作树未提交改动很多，交接时不要把「目录里有文件」当成已入库基线。支持负担主要是本机环境隔离（采集 `.venv` 与 gsplat 运行时分开）和长耗时几何复算，而不是服务器运维。

### For AI Coding Agents

先读 `AGENTS.md`、`技术路线.md`、`CLAUDE.md`、本文件和 `ARCHITECTURE_DIAGRAMS.md`。改采集、坐标、筛帧、融合、TSDF、Mesh、格式或依赖时必须同步 `技术路线.md` 并追加 `doc/变更记录/` 当月文件。

开始改之前看 `git status` 与并行进程；不要批量提交。只允许上层 import 下层，函数内延迟 import 也算。不要用放宽预算、删除未知范围或关闭验证来「修好」失败。算法版本字符串是持久契约：旧证明按原版本回放，不按「已有新版本」删除。

## 4. System Context

操作者通过工作台连接 Femto Bolt SDK。相机适配器是 `femto/hardware/orbbec_rgbd.py` 的 `OrbbecAlignedCamera`。帧、标定和时间戳写入本工作树 datasets；派生几何、规划与证明写入 derived 或 outputs。历史对话框和打开项目入口只读回放本地模型或便携方案。

当前没有远端 API。本地 `history.sqlite3` 管理历史索引，`execution.sqlite3` 独立记录软件模拟任务与事件；Qt 后台桥接使用有界本地操作队列。外部依赖是本机 SDK、Open3D、VTK、PySide6，以及可选的独立 gsplat/CUDA 运行时。`scan_exchange` 有任务/观测封装，不等于设备执行。

部位选择来自 `femto/dataset/body_site.py`：第一阶段支持肩、胸、腹、腰、髋、上臂、肘、前臂、腕、手背、手掌、大腿、膝、小腿；头颈与足部不可新建。部位是任务意图，不是测量坐标。

详见 `ARCHITECTURE_DIAGRAMS.md` 第 1 节。

## 5. Main Entry Points

| Entry Point | Type | Purpose | Evidence | Confidence |
| --- | --- | --- | --- | --- |
| `start_femto_app.cmd` | UI launcher | 启动正式采集与规划工作台；先做 import 检查 | `start_femto_app.cmd`、`femto/app/qt_main.py` | High |
| `start_region_planner.cmd` | UI launcher | 同一工作台，`--ui-preview`，不自动连相机 | `start_region_planner.cmd` | High |
| `python -m femto.app.qt_main` | UI | 实际窗口与页面生命周期 | `femto/app/qt_main.py` 的 `WorkstationWindow` | High |
| `tools/simulate_devices.py` | CLI | 同一业务链的显式模拟、故障、只读回看与导出 | `femto/execution/application.py`、`doc/使用说明/设备管理与模拟联调.md` | High |
| `femto/app/femto_control_app.py` | compatibility | 旧入口转发到 Qt；再导出扫描会话与 gsplat 符号 | 模块说明与 import | High |
| `femto/pipeline/local_offline.py` | worker | 无相机、有时间/内存上限的离线重建子进程 | `TIME_BUDGET_S`、`MEMORY_BUDGET_MB`、`publish_result` | High |
| `start_real_scene_demo.cmd` | CLI launcher | 通用物品/场景实拍与重建，不是人体主流程 | `doc/使用说明/启动入口说明.md` | High |
| `python -m femto.gsplat.gsplat_backend` | CLI | 独立环境的 3DGS 状态/训练/查看 | `femto/gsplat/gsplat_backend.py` | High |
| `start_direct_gsplat_train.cmd` | CLI launcher | 正式 gsplat 训练 | `doc/使用说明/启动入口说明.md` | High |
| `start_direct_gsplat_viewer.cmd` | CLI launcher | 正式 gsplat 查看 | `doc/使用说明/启动入口说明.md` | High |
| `prototype/main.py` | prototype | 独立 Mock；`--replay` 只读 | `prototype/README.md` | High |
| `tests/` via unittest discover | test | 软件回归；从仓库根运行 | `CLAUDE.md` 测试约定 | High |

`--project` 启动时 `WorkstationWindow` 会打开 `ReviewPage`，不是规划编辑页。无相机时用 `--ui-preview`。

## 6. Major Components

分层由 `tools/check_layering.py` 的 `ORDER` 固定：`femto` → `depth` → `dataset` → `geometry` → `hardware` → `reports` → `annotation` → `tracking` → `temporal` → `local` → `planning` → `viewer` → `pipeline` → `gsplat` → `capture` → `execution` → `app`。

| Component | Responsibility | Public Interface / Entry | Important Files | Depends On | Boundary Notes |
| --- | --- | --- | --- | --- | --- |
| app | 四页生命周期、流程导航、相机与模拟后台宿主 | `WorkstationWindow`、`CapturePage`、`PlanningPage`、`ReviewPage`、`ExecutionPage` | `femto/app/qt_main.py`、`femto/app/qt_execution.py`、原采集/规划/回看页 | execution、capture、planning、viewer | UI 不是数值证明来源；按钮可用不代表真实执行资格 |
| execution | 冻结模拟方案、前置检查、显式状态、反馈协调、资源与事件库 | `create_coordinator`、`Coordinator`、`ExecutionStore` | `femto/execution/` | hardware、planning、geometry、dataset | 不导入 Qt、不控制真实设备；未知停止保留资源 |
| hardware | 唯一相机 SDK 封装、设备消息与版本化软件模拟 | `OrbbecAlignedCamera`、`DevicePort`、`SimulatedDevice` | `femto/hardware/orbbec_rgbd.py`、`configuration.py`、`interface_contract.py`、`simulation/` | SDK、geometry、dataset | 不理解规划任务；真实未实现适配器不回退模拟 |
| capture / tracking | 原始提交、在线跟踪、合格视角、阶段快照、扫描框 | `LocalCapture`、`freeze_stage`、`CaptureCheckpoints` | `femto/capture/local_capture.py`、`femto/capture/local_checkpoint.py`、`femto/capture/scan_roi.py` | tracking、pipeline、hardware | 同会话保留 W0；失败帧不伪装有效位姿 |
| local / pipeline | 精配准、重建、发布、离线资源边界 | `LocalOfflineJob`、`publish_result`、`reconstruct_scene` | `femto/local/local_processing.py`、`femto/pipeline/local_offline.py`、`femto/pipeline/scene_reconstruction.py` | geometry、reports | 点云与 TSDF 并行消费同一合格 RGB-D，不是「PLY 再喂给 TSDF」 |
| annotation / dataset | 参考投影、选区意图、部位标签、最终确认 | `FixedRGBModel`、`checked_new_selection`、`confirm_for_planning` | `femto/annotation/fixed_rgb_geometry.py`、`femto/annotation/selection_intent.py`、`femto/dataset/body_site.py`、`femto/dataset/acquisition_completion.py` | geometry、depth | 扫描框 ≠ 工作区域；确认 ≠ 质量改写 |
| planning | 原目标、有效域、路线、覆盖、补线、便携保存 | `surface_workspace.generate`、`Certificate` | `femto/planning/surface_workspace.py`、`femto/planning/target_surface.py`、`femto/planning/coverage_workflow.py`、`femto/planning/region_project.py` | annotation、geometry | 版本化回放；输入变化使旧证明失效 |
| viewer / gsplat | 几何显示与可选 Gaussian 表现 | VTK 视口、gsplat CLI | `femto/app/qt_vtk_viewport.py`、`femto/gsplat/gsplat_backend.py` | VTK；独立 ML runtime | Gaussian 不替代测量几何 |
| shared infrastructure | 路径、JSON、摘要、资源缓冲 | `REPO_ROOT`、工作区专属 gsplat 路径 | `femto/paths.py`、`femto/io_utils.py`、`femto/resource_policy.py` | 标准库等 | 资源适配只调节可丢弃缓冲，不放宽证明预算 |
| prototype | Mock 交互原型 | `prototype/main.py` | `prototype/README.md` | 不导入 femto | 视觉合同不能证明采集或规划已实现 |

`femto/planning/` 约 87 个模块，按算法阶段而不是按页面切分：P0/P1 台账、P2 名义覆盖、P3 补充、P4 严格路径、P5 变宽。P6 跨会话继承未开发。

## 7. Data and Control Flow

### 医师控制流（工作台）

`qt_workflow.STEPS` 是采集与规划的六步导航，由 `WorkflowNavigation` 决定哪些步骤可点。`WorkstationWindow` 实际堆四页：

1. `CapturePage`：部位与范围、局部采集、检查模型。
2. `PlanningPage`：标记区域、生成路线、核对保存。
3. `ReviewPage`：已保存方案只读回看；`--project` 与「打开项目」进入此页。
4. `ExecutionPage`：顶部「设备与模拟」进入独立软件模拟、配置/标定详情与模拟记录回看，复用前三页的切换/退出检查链。

采集页经 `CameraHost.command_queue` 发 `local_start`。`CameraWorker` 在相机线程持有 SDK，构造 `LocalCapture`。原始写入回执与帧号、时间戳核对后，才能形成可信派生记录。输入队列有界；预览、接纳视角与原始保存不是同一集合。

`local_check` 由 `CaptureCheckpoints` 在已提交边界冻结原始帧前缀。阶段作业走 `LocalOfflineJob`（600 s / 4096 MiB），阶段模型不发布业务 latest，也不进入规划。最多 8 次阶段是资源上限，不是完整度判据。操作者可在同会话继续采集后再检查。

结束采集重新做最终处理、写 `acquisition_completion` 并 `publish_result`。`confirm_for_planning` 只把同一绑定的最终记录标为已确认。规划加载在 `require_confirmed=True` 时拒绝未确认与阶段模型。历史只读允许查看绑定通过的未确认最终模型，主按钮回到采集检查页。

重新采集同一任务建立独立会话，不继承旧扫描坐标或覆盖。`rescan_region` 是新会话，不是把新帧融进旧网格。

### 重建数据流

`scene_reconstruction` 对合格 RGB-D 做去畸变、深度准备、反投影和位姿变换。同一帧同时供点云累积和 TSDF 积分；局部链再筛观测支持面。因此不能把「融合 PLY → TSDF」画成唯一输入关系。

W0 是会话首个有效彩色相机位姿。网格已在 W0 RDF 米；查看器与标注不再乘相机变换。原点声明必须与 local_geometry.json、重建报告、源 manifest、跟踪报告和首帧 Pose 一致，否则固定 RGB 选区拒绝加载。

### 规划与证明数据流

`FixedRGBModel` 绑定源网格、去畸变 RGB、深度、K 与参考来源。医师涂抹形成 `selection_intent`。`surface_workspace.generate` 捕获完整修订，建质量证据、`TargetLedger` 和有效域，再生成并约束 `SurfaceTrajectory`。没有合格面时阻断并保留原选区。

覆盖核验消费原目标、扫描边种子与作用模型，输出 `Certificate`。P3 补充只有「原确认集合无损 + 新扫描见证增益达标」才接纳。预算退出和数值未确认不等于观测缺失，不应全部送去补采。

保存走 `region_project`：登记资产、验证再发布。独立打开核验来源、几何与证明并完整复算。核验、补线、保存都有文件 I/O，后台任务用取消、超时、代次和模型摘要拦截迟到结果。

### 核心契约

| 数据 | 实际形态 | 关键边界 |
| --- | --- | --- |
| 实时帧 | `AlignedFrame` | 完整对齐 RGB-D；内参在 camera 对象 |
| 原始帧集 | manifest、frames.jsonl、图像与标定 | 原始移动帧可无有效 pose；估计值只写派生层 |
| 扫描框 | 会话 derived 下的 scan_roi.json | 首帧 RGB 像素矩形，不是治疗区域 |
| 原目标 | `TargetLedger` | `pixel_status` 保留缺失/未知；质量不删原请求 |
| 路线 | `SurfaceTrajectory` | 点 `xyz_m`；role 区分 scan/connector；块间不规划转场 |
| 覆盖 | `Certificate` | root/depth/code 是身份，status 才是判定 |
| 修订 | `planning_revision` 字典 | 输入摘要、target_id、保存版本、请求代次各有职责 |
| 确认 | 结果目录中的 acquisition_completion.json | confirmed_for_planning 不改质量字段 |

副作用边界：相机 SDK 与原始写入在相机线程；重建在受限子进程；规划计算在冻结输入的后台任务；Qt 控件和 VTK actor 只在 GUI 线程更新。显示抽样不得替换计算输入。

## 8. Runtime, Configuration, and Deployment

正式采集环境是本工作树 `.venv`。产品 gsplat 路径由 `femto/paths.py` 按工作树标识生成 `C:/FemtoRuntimes/<workspace_id>/gsplat`（非 Windows 回退 `.venv-gsplat`）。`.venv-3dgs` 是已死对照，禁止产品分发。不要把注释或旧文档里的环境名当成当前常量。

无相机预览：

```powershell
.venv\Scripts\python.exe -m femto.app.qt_main --ui-preview
```

打开已有结果或方案：

```powershell
.venv\Scripts\python.exe -m femto.app.qt_main --ui-preview --project "实际的模型目录或方案.zip"
```

安装入口：`setup_environment.cmd`、`setup_direct_gsplat_environment.cmd`。启动清单见 `doc/使用说明/启动入口说明.md`。检查入口是 `检查工作区环境.cmd`。仓库内未见 GitHub/GitLab CI 工作流；验证靠本机六项命令。

运维信号：启动检查日志、工作台 stdout/stderr 日志、publication.json、处理报告、可取消后台任务。outputs 下的 latest marker 只是发现入口，不是来源证明。带 `publication_required` 的结果必须有有效 `published` 凭证才能恢复。

依赖以 `requirements.txt`、锁文件、`THIRD_PARTY_NOTICES.md` 和独立 gsplat 清单为准。新增第三方依赖必须先登记，否则合规检查失败。不要运行 `commercial_compliance_check.py` 主程序做例行检查，它会覆盖 `compliance/current_report.json`；例行只用受限 import 干扫描。

本次走查未启动相机、未训练、未安装依赖、未读取被忽略的凭据目录。

## 9. Testing and Verification

本次实际运行（2026-10-20）：

- `tools/check_layering.py`：`upward edges: 0`
- `tests/test_capture_checkpoints.py`：7 项通过
- `tests/test_acquisition_completion.py`：8 项通过

这些检查验证阶段快照和最终确认等软件边界，不是实物精度验收。未重跑全量 unittest，不沿用文档里的历史总数（842 / 881 / 1520 / 1547 / 1566 / 1581 都是当时快照）。

提交前六项命令以 `CLAUDE.md` 为准，从仓库根执行，不得跳过或放宽。单文件使用 `unittest discover -s tests -p "test_x.py"`。Qt 测试文件在 import PySide6 前设置 `QT_QPA_PLATFORM=offscreen`。`prototype/` 的两套测试必须分两个进程。

最快定向路径：分层检查 + 与改动对应的 tests 文件。改采集门禁至少跑 checkpoint 与 completion；改路线/覆盖至少跑对应 workflow / certificate / revision 测试。

架构文档验证：

```powershell
python "C:\Users\55053\.codex\skills\codex-repo-architecture-walkthrough\scripts\verify_architecture_docs.py"
```

缺口：仓库没有可见 CI，全量回归耗时长且含几何复算；Qt 原生视窗与实机采集仍需手工/现场验收。8 个结构性 skip 与 torch 缺失有关，`.venv` 跳过数变化说明环境变了。

## 10. How to Navigate the Codebase

1. 读 `技术路线.md` 顶部，确认本工作树状态；README / CLAUDE 的概述可能滞后。
2. 读 `femto/app/qt_main.py`，确认四页、`can_leave()` / `shutdown()`、异步来源安装。
3. 读 `femto/app/qt_workflow.py` 与 `femto/app/qt_workflow_navigation.py`，把采集/规划六步导航、只读回看与独立设备模拟入口映射起来。
4. 采集：`femto/app/qt_capture.py` → `femto/app/camera_worker.py` → `femto/capture/local_capture.py` → `femto/capture/local_checkpoint.py` → `femto/pipeline/local_offline.py` → `femto/capture/scan_feedback.py` → `femto/dataset/acquisition_completion.py`。
5. 规划：`femto/annotation/fixed_rgb_geometry.py` → `femto/annotation/selection_intent.py` → `femto/planning/surface_workspace.py` → `femto/planning/target_surface.py` → `femto/planning/coverage_workflow.py` → `femto/planning/region_project.py`。
6. 模拟：`femto/app/qt_execution.py` → `femto/execution/application.py` → `coordinator` / `history` → `femto/hardware/simulation`；方案核验共用 `femto/planning/saved_source.py`。
7. 用对应 `tests/` 文件看失败、取消、缺件和旧格式边界。
8. 需要算法细节时再打开 `doc/技术资料/` 下 P0–P5 契约，而不是历史开发记录。

`prototype/` 与 `doc/历史开发记录/` 最后读。不要用其他 worktree 的 `datasets/` 路径当本树证据。

## 11. Safe Change Guide for Humans and AI Agents

AI agents 先检查 Git 状态与文件更新时间。本工作树在走查时已有大量未提交改动；它们不是本次架构分析产生的。本次只更新 `ARCHITECTURE.md` 与 `ARCHITECTURE_DIAGRAMS.md`。

改前必读：`AGENTS.md`、`技术路线.md`、`CLAUDE.md`、对应契约、本文件。

不要随手改：

- 原始场景目录中的 rgb、depth、pose、calibration、frames.jsonl、manifest.json
- 被忽略的凭据、`git/`、锁文件、`licenses/`
- `prototype/standalone.py`（由 `prototype/build_standalone.py` 生成）
- 历史证明资产、算法版本字符串、固定预算常量
- Graphdeco 相关导入；不得为「修」gsplat 导入检查而把查看器的顶层 PySide6 import 改成延迟导入

常见局部变更与范围：

- 采集 UI / 阶段检查：`femto/app/qt_capture.py`、`femto/capture/local_checkpoint.py`、`tests/test_capture_checkpoints.py`
- 最终确认 / 规划准入：`femto/dataset/acquisition_completion.py`、`tests/test_acquisition_completion.py`
- 部位范围：`femto/dataset/body_site.py`、`tests/test_body_site_boundary.py`
- 路线生成：`femto/planning/surface_workspace.py`、`femto/planning/planning_revision.py`
- 覆盖 / 补线：`femto/planning/coverage_workflow.py`、`femto/planning/coverage_refinement.py`
- 保存回读：`femto/planning/region_project.py`

扩大范围必须向人说明的情况：改证明语义或预算、改坐标/W0、改 marker 文件、改分层方向、把 CLI 能力接到 Qt、引入网络或第二套三维 UI。

验证：至少分层检查 + 相关 unittest。完整提交跑六项。不要把 UI 预览或 Mock 截图当成几何正确。

必须保留的术语：W0、RDF、米、原始证据、阶段模型、最终确认、未确认、原目标、有效表面域、Certificate、execution_ready。

## 12. Assumptions and Items Needing Human Validation

| Item | Assumption or Question | Why It Matters | Confidence |
| --- | --- | --- | --- |
| 实物重复性与尺度 | 本次未重新测量 | 软件一致不代表尺寸误差界 | Low |
| 同会话现场闭环 | 源码与 10/13–10/14 记录存在；完整「多轮阶段检查 → 最终确认 → 选区保存」未在本次实拍复验 | 阶段结果不能冒充产品验收 | Medium |
| CLAUDE「两页」表述 | 源码已有 `ReviewPage`；文档未同步 | agent 若只信 CLAUDE 会漏掉回看页与 `--project` 行为 | High（不一致已证实） |
| 正式发布装配 | 只读了路径与入口，未在干净机器重装 | 运行环境必须按工作树验证 | Medium |
| 部署形态 | 按单机研发工作台分析 | 若改多人服务，持久化、权限和数据隔离都要重做 | Medium |
| 未提交改动归属 | 大量 M/?? 文件可能来自并行 agent | 架构文档不能把未入库实验当成已发布基线 | Medium |

### Validation notes

- 原走查发现的两页概述已同步：当前 `CLAUDE.md` 和 `qt_main` 均记录采集、规划、回看与设备模拟四页；`--project` 仍走 `open_review`。
- 采集/规划流程继续受阶段检查与最终确认门禁约束；设备模拟是独立软件业务，模拟就绪不赋予原方案执行资格。
- 点云与 TSDF 由同一合格 RGB-D 生成，不是串行「先 PLY 再 TSDF」。
- 质量检查结果控制显示和人工决策，不把网格对象替换成统计摘要。
- 硬件层已有软件模拟适配器，真实机械臂 provider 尚未落地；`scan_exchange` 不是执行接口。
- 旧产品示意图若把所有 pending 画成补采/补线，与 `scan_feedback` / `coverage_tasks` 的分类不符。

## 13. Suggested Architecture Improvements

| Category | Recommendation | Evidence | Senior Developer Rationale | Manager Rationale | Effort | Risk | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Documentation/context | 后续增减页面继续同步四页生命周期、独立设备模拟入口及 --project 打开回看 | `qt_main`、`CLAUDE.md` 和本轮模拟文档已同步 | 减少 agent 按过期上下文改错生命周期 | 降低交接返工 | S | Low | 持续维护 |
| Verification/guardrail | 保持阶段模型拒绝规划和最终确认绑定的回归，改 Qt 导航时必跑 | 本次 7+8 项定向测试通过 | 防止 UI 改动绕过后端门禁 | 保护已验证业务边界 | S | Low | P1 |
| Runtime/operations | 补齐同会话现场：阶段检查、最终确认、选区、保存 | `技术路线.md` 现场进度仍待验收 | 验证真实负载与操作路径 | 用完整闭环判断交付，而不是模块完成百分比 | M | Low | P1 |
| Contract/schema | 给 command/event 摘要补字段索引或静态类型，复用现有校验器 | `CameraHost.send` 与 dict 事件 | 更早发现字段漂移，不必平行再做一套 schema | 需求变更更可预测 | M | Medium | P2 |
| Verification/guardrail | 邻接索引等性能试验保持显式版本，多样本比较后再决定是否默认 | `femto/planning/coverage_indexed.py` 仍为试验 | 避免单样本变快被当成全局正确 | 避免未验证的性能承诺 | M | Medium | P2 |
| Boundary/modularity | `femto/app/femto_control_app.py` 继续收缩为兼容转发，避免新 Qt 页再依赖它 | 模块仍顶层再导出 gsplat 与扫描会话 | 降低采集环境误拉训练依赖的风险 | 减少环境故障的支持成本 | M | Medium | P2 |

## 14. Glossary

| Term | Meaning | Evidence | Confidence |
| --- | --- | --- | --- |
| W0 / `T_W0_Ct` | 会话参考彩色相机坐标；当前相机到 W0 的变换 | `femto/capture/local_capture.py` | High |
| RDF | 磁盘相机坐标：X 右、Y 下、Z 前 | `技术路线.md` | High |
| checkpoint / 阶段模型 | 同会话已提交前缀的重建；不发布、不规划 | `femto/capture/local_checkpoint.py` | High |
| `confirmed_for_planning` | 同一最终模型已人工确认；不改质量判定 | `femto/dataset/acquisition_completion.py` | High |
| Scan ROI | 首帧 RGB 上的采集引导框 | `femto/capture/scan_roi.py` | High |
| TargetLedger | 保留完整选择意图及定位状态的原目标 | `femto/planning/target_surface.py` | High |
| Certificate | 原目标、轨迹、种子和作用模型绑定的覆盖证明 | `femto/planning/coverage_certificate.py` | High |
| pending / unconfirmed | 尚未证明；原因可能是观测、几何、算法或预算 | `femto/planning/coverage_tasks.py` | High |
| execution_ready | 固定为 false；不是设备可执行 | `femto/planning/surface_workspace.py` | High |
| P0–P5 | 台账、名义覆盖、补充、严格路径、变宽 | `doc/技术资料/` 契约 | High |
| P6 | 跨会话继承；未开发 | `README.md` | High |
| SPZ / 3DGS | Gaussian 视觉表现支线 | `femto/gsplat/gsplat_backend.py` | High |

## 15. Appendix: Evidence Map

| Claim | Evidence | Confidence |
| --- | --- | --- |
| 四页生命周期与 `--project` 打开回看 | `femto/app/qt_main.py`：`CapturePage` / `PlanningPage` / `ReviewPage` / `ExecutionPage`、`open_review`；`tests/test_qt_execution.py` | High |
| 六步导航独立于页面质量 | `femto/app/qt_workflow.py`：`STEPS`、按钮 state | High |
| 唯一 SDK 适配 | `femto/hardware/orbbec_rgbd.py`：`OrbbecAlignedCamera` | High |
| 相机线程命令入口 | `femto/app/qt_capture.py`：`CameraHost.send`；`femto/app/camera_worker.py`：`_cmd_local_start` | High |
| 阶段前缀与资源上限 | `femto/capture/local_checkpoint.py`：`freeze_stage`、`MAX_STAGE_ROUNDS = 8` | High |
| 离线重建预算与发布 | `femto/pipeline/local_offline.py`：`TIME_BUDGET_S`、`publish_result` | High |
| 最终确认与规划准入 | `femto/dataset/acquisition_completion.py`：`read_completion`、`confirm_for_planning` | High |
| 部位新建范围 | `femto/dataset/body_site.py`：`SUPPORTED_REGIONS`、`checked_new_selection` | High |
| 路线生成顺序 | `femto/planning/surface_workspace.py`：`generate` | High |
| 单向分层 | `tools/check_layering.py`：`ORDER`；本次 `upward edges: 0` | High |
| 工作区专属 gsplat 路径 | `femto/paths.py`：`ASCII_GSPLAT_RUNTIME_ROOT` | High |
| prototype 隔离 | `prototype/README.md` | High |
| 定向测试 | `tests/test_capture_checkpoints.py`、`tests/test_acquisition_completion.py` | High |

配套图见 `ARCHITECTURE_DIAGRAMS.md`。
