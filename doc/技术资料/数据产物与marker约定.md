# 数据产物与 marker 约定

> 本文保存从 `CLAUDE.md` 迁出的完整目录、产物和 marker 细节；硬性摘要仍保留在 `CLAUDE.md`。

## 目录产物与 marker 文件

这些文本 marker 是各入口之间的唯一耦合点，改输出结构时要一起改：

- `datasets/scene_<ts>/`（高级页/CLI 采集）、`datasets/scene_<ts>_local_<8hex>/`（局部补采）、`datasets/temporal_capture_<ts>/`（时域 burst，`raw/frame_NNN_*.png`，无 `pose/`，校验器会报缺文件）。`datasets/latest_scene.txt` 由 `finalize_scan` 写到 **`session.parent`**（曾经误写全局索引），每次结束含失败会话都会更新。
- `datasets/scene_*/derived/`：`posed_rgbd_odometry_*`、`fixed_keyframe_*`、`local_posed/`、`processing_<8hex>/`、`live_tracking.jsonl`、`local_reference.json`、`local_capture_report.json`、`local_failure.json`、`live_preview_report.json`、`capture_quality.json`、`temporal_fusion_<ts>/`（独立时域）、`temporal_fusion/<ts>/{keyframe,models}/`（一键关键帧），以及 `latest_keyframe_dataset.txt` / `latest_fixed_dataset.txt` / `latest_posed_dataset.txt`。
- `outputs/complete_demo_*/`：`demo_summary.json`（`data_kind`、`fixed_camera`、`frame_count`、`nerfstudio`、`publication_*`、`local_measured`、`nerfstudio_skipped_reason`）、`publication.json`、`demo_report.html`、`result_quality.html/json`、`field_acceptance_spec.json`、`processing_report.json/html`、`reconstruction/`（`scene_tsdf_mesh_m.ply/.glb/.obj`、`texture_multiview.png` + `mesh_triangle_uv.npy`、`scene_fused_points_m.ply`、`initial_observed_points_m.ply`、`local_geometry.json`、`reconstruction_report.json`、`gaussian/`、`derived/depth_quality/`）、`nerfstudio/` 或 `nerfstudio_strict_<ns>/`。`reconstruction/` 下还会累积永不覆盖的 `annotations/<stamp>_<uuid8>/`（其内可含 `trajectory_<12hex>/` 三维表面轨迹与 `derived_surface_candidates_<12hex>/` 阶段 C 候选诊断，后者不进入规划、不被历史清理扫描）、`measurement_validation/`、`region_paths/path_*/`、`region_projects/*.region.zip`、`region_project_history/`。
- `outputs/latest_complete_demo.txt`（→ `…/reconstruction`）、`outputs/latest_moving_reconstruction.txt` 与 `latest_keyframe_reconstruction.txt`（→ complete_demo 根，仅 `real_femto` 且发布凭证有效）、`outputs/latest_direct_gsplat.txt`、`outputs/latest_single_gs.txt`、`outputs/local_history_deleted.json`（历史窗口软删除索引，只隐藏不删文件）。
- `outputs/direct_gsplat_*/`（`training_report.json`、`acceptance_report.json`、`checkpoints/latest.pt`、`model/{trained_gaussians.ply,.spz,model_manifest.json}`、`viewer_stdout.log`、`orbit_render/`）、`outputs/single_gs_*/`、`outputs/local_comparisons/comparison_*/`、`outputs/region_workspaces/<uuid>/`（打开 `.region.zip` 的工作副本）、`outputs/planning_drafts/`。
- `photos/femto_photo_*/`（`femto-rgbd-photo-v1`，含 `derived/depth_quality/`）、`photos/latest_photo.txt`；`outputs/patient_top_*/`、`outputs/latest_session.txt`（人体分支）。
- `compliance/current_report.json`（只有合规脚本写，无代码读）、`Log/`、`%LOCALAPPDATA%\FemtoSceneDemo\ui_preferences.json`（语言偏好）。
- 训练器 `--input` 既可指向 `nerfstudio/` 也可指向 complete demo 根目录（通过 `demo_summary.nerfstudio` 解析）；不传则取 `latest_complete_demo.txt`。

`reconstruction_sources.mode_of` 分类规则：`keyframe` = `real_femto` + `fixed_camera True` + `frame_count 1` + 单帧来源可解析；`moving` = `fixed_camera False` + `frame_count>1` + （`nerfstudio/transforms.json` 存在 或 局部 <12 视角跳过标记）；带 `publication_required` 的结果必须 `published`。启动时 `restore()` 扫描 `outputs/*/demo_summary.json` 重建，不信任目录名。


### Qt 任务与版本派生层（2026-09-28）

- 已完成采集的 `outputs/complete_demo_*/task_context.json` 是可删除的界面任务标签，包装 schema 为 `femto-capture-task-link-v1`，绑定绝对 `capture_result`；其 `task` 使用 `femto-ui-task-v1`，字段为 `task_id/name/body_region/created_at`。不写入原始 RGB-D，不作为患者身份或坐标真实性证明；文件迁移导致绑定失效时忽略任务标签并提示，不据此关联其他采集。
- 便携 `femto-region-project-v1` 增加可选 `task_context`、`parent_revision_id` 字段，旧工程无这些字段仍可读。`project_id` 标记同一工程，`revision_id` 标记每次新保存；GUI 新版本使用新文件路径，不覆盖原保存文件。原生轨迹格式及其几何/执行门禁不变，任务与版本由随附的 `planning.region.zip` 保存。
- GUI 保存成功后同时登记 `outputs/saved_projects/region_project_history/`，索引含任务标签、版本和父版本，供“全部已保存方案版本”读取；原采集目录的索引继续保留；汇总同时扫描 `outputs/complete_demo_*/reconstruction` 和 `outputs/region_workspaces/*` 下的旧索引，按工程路径去重，同目录便携方案与原生轨迹仅展示一个方案入口。索引是派生数据，读取工程仍执行资产指纹、来源与路线核验。
- 只读回看不恢复编辑草稿；关联采集通过完整参考绑定核验，不能用目录相似或最近一次采集替代。

### 设备配置、安装记录与模拟任务（2026-10-21）

以下位置由 `femto.paths` 管理，均不写入原始 RGB-D 数据集，不更新采集、模型或方案的 latest marker：

| 路径 | 写入者与生命周期 | 追溯边界 |
| --- | --- | --- |
| `outputs/hardware_configuration/<device_id>/<revision>.json` | 静态 `DeviceConfiguration`；同一修订拒绝覆盖，读取不连接 | `femto-device-configuration-v1`；连接、就绪、能力和来源由实际适配器反馈，配置不能赋予 |
| `outputs/hardware_configuration/simulation_scenarios/<scenario_id>/<revision>.json` | 可保存的 `Scenario`；显式软件来源及故障、能力、时钟预算 | `femto-simulation-scenario-v1`；场景预算不等于厂家协议、控制周期或物理参数 |
| `outputs/installation_calibrations/<id>/<revision>.json` | `TransformRecord` 独立版本；不修改 `datasets/*/calibration` | `femto-installation-transform-v1`；合成与来源核验但实测未合格区分；结构合法不证明物理精度 |
| `outputs/history_metadata/history.sqlite3` | 原 `HistoryStore` 管理历史集合与隐藏状态，schema v1 保持 | 文件索引可重建，隐藏状态须备份；不承载设备事件 |
| `outputs/history_metadata/execution.sqlite3` | 首次写模拟任务才创建；`ExecutionStore` 追加任务/事件，独立 schema v1 | 冻结路线、源路径/修订/SHA、配置/连接代次与事件；读取或启动空页面不创建，回看不重发 |
| `outputs/simulation_resources/<key_sha256>.lock` | 独立模拟资源和任务所有者进程锁；未知任务保留占用，显式关闭软件实例后释放 | 锁文件中的 owner/PID 是诊断，实际互斥由操作系统持锁保证；不占用真实相机或复用元数据发布锁 |
| 备份目录中的两库快照与 `manifest.json` | `tools/history_metadata.py --backup` 分别使用 SQLite 在线备份 | 各库一致，跨库无原子时刻；恢复须保留清单时间窗及两库，不能混用旧 WAL/SHM，完整图像/模型资产独立保留 |
| 操作者选择的模拟导出 JSON | 显式导出完整冻结快照及全部有序事件，原子写入且拒绝覆盖 | `source=simulation`、只读，`execution_ready=false` / `physical_accuracy_verified=false`；不能作为采集或覆盖证明 |

实采方案只提供核验后的原路线引用，运行来源仍为 simulation。原生任务绑定 `trajectory_files.json` 登记输出、区域清单及来源引用；同目录未登记草稿不影响冻结任务。原治疗头 `values_status`、无工具姿态、断点和未知覆盖结论不改写。双库恢复见[SQLite历史记录与备份](../使用说明/SQLite历史记录与备份.md)，任务/状态及坐标规则见[模拟联调与坐标准备契约](模拟联调与坐标准备契约_20261021.md)。
