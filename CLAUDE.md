# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库是未来聚焦超声治疗设备的"真实场景 / 患者体表三维采集、重建、显示、测量与空间数据基础"。硬件基线：Orbbec Femto Bolt RGB-D、Windows、NVIDIA CUDA GPU、Python。

Femto Bolt → RGB-D 原始数据 → 标定 / Pose / 坐标 → Point Cloud / TSDF / Mesh → 标注 / 测量 / 区域规划（研发演示）→ direct gsplat 3DGS → 查看器。

本仓库不是 Gaussian Splatting Demo。3DGS 只是表现层之一，治疗区域、测量、路径规划都建立在 PointCloud / Mesh / Geometry 层之上。当前阶段明确不实现机械臂控制、ROS、MoveIt、轨迹执行和超声治疗控制，但坐标、单位、Transform、Scene、ROI、Device、Dataset 的设计必须为未来接入留边界。

`技术路线.md` 是唯一技术基线（Source of Truth）；README、本文件和源码与其冲突时以它为准。它的顶部只保留当前状态和稳定约束，变更记录按月归档在 [`doc/变更记录/`](doc/变更记录/技术路线_变更记录_2026-09.md)。

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
- 备份与差异以 Git 为准，不再新建 `outputs/*_backup_*`；遗留备份中的源码不是 live code，搜索时忽略。
- 每轮交付记录写入 `doc/历史开发记录/`，使用说明写入 `doc/使用说明/`；当前状态和稳定约束同步到根目录 `技术路线.md`，完整日期记录追加到 `doc/变更记录/` 对应月份文件。
- 报告口径：软件/合成/历史回放验证与实物验收分开写；开发合规通过 ≠ release_ready。

## 环境

- 三套环境严格隔离：`.venv`（采集/几何/UI/完整测试）、`C:\FemtoGsplatRuntime`（产品 gsplat/torch/SPZ）、`.venv-3dgs`（已死且禁止产品分发的旧对照）。不得合并或跨环境补依赖。
- 产品 gsplat 必须走 ASCII 联接、CUDA 12.8 和 VS2022 Build Tools；不得联网下载 LPIPS，版本、许可证、来源和 SHA-256 必须同步锁定。
- `.venv` 的 `pyorbbecsdk2 --no-deps` 会让 `pip check` 名义缺少 `opencv-python`，这是预期，不能擅自“修复”。
- 完整环境、权重、SPZ 构建和安装约定见 [`开发环境与架构约定.md`](doc/技术资料/开发环境与架构约定.md)。

## 常用验证命令

每次提交前从仓库根目录执行全部六项；不得跳过、改写或放宽：

```powershell
.venv\Scripts\python.exe tools\check_layering.py
.venv\Scripts\python.exe -X utf8 -c "import importlib,pathlib;ms=[p.with_suffix('').as_posix().replace('/','.') for p in pathlib.Path('femto').rglob('*.py')];[importlib.import_module(m) for m in ms];print(len(ms),'modules OK')"
C:\FemtoGsplatRuntime\Scripts\python.exe -X utf8 -c "import femto.gsplat.scene_gsplat_trainer,femto.gsplat.scene_gsplat_viewer,femto.gsplat.gsplat_backend,femto.gsplat.scene_single_gs,femto.gsplat.scene_single_gs_viewer,femto.gsplat.scene_gsplat_acceptance;print('gsplat runtime OK')"
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests
C:\FemtoGsplatRuntime\Scripts\python.exe -m unittest discover -s tests -p "test_gsplat_guard.py"
.venv\Scripts\python.exe -X utf8 -c "from pathlib import Path;from commercial_compliance_check import imported_modules,BLOCKED_IMPORT_PREFIXES;bad=[(f,[m for m in imported_modules(f) if m.startswith(BLOCKED_IMPORT_PREFIXES)]) for f in list(Path('femto').rglob('*.py'))+list(Path('tools').glob('*.py'))];print([b for b in bad if b[1]] or 'clean')"
```

- `.venv` 全量测试固定 8 个环境跳过；单文件使用 `unittest discover -s tests -p "test_x.py"`，不要用点分模块名。Tk 测试必须在交互桌面运行，期间不要操作鼠标键盘。
- **不要随手运行 `commercial_compliance_check.py` 主程序**：它会覆盖 `compliance/current_report.json`。这里只允许第六条的受限 import 干扫描。
## 架构

源码位于 `femto/` 包；入口从仓库根以 `python -m femto.<pkg>.<module>` 运行，数据树路径来自 `femto.paths`。

- 分层由低到高固定为：`femto` → `depth` → `dataset` → `geometry` → `hardware` → `reports` → `annotation` → `tracking` → `temporal` → `local` → `planning` → `viewer` → `pipeline` → `gsplat` → `capture` → `app`。
- **只允许上层 import 下层，函数内延迟 import 也算**。新增 import 前必须运行 `tools/check_layering.py` 并保持 `upward edges: 0`；下层需要上层功能时应下移纯函数，不能用延迟 import 绕过。
- `femto.gsplat.*`、viewer 和 single-GS 只能在函数内 `import torch`；`femto.io_utils` 不能顶层 import `cv2`。
- 16 层逐包职责、模块清单和 `tools/` 约定见 [`doc/技术资料/开发环境与架构约定.md`](doc/技术资料/开发环境与架构约定.md)。

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
