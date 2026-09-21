# 数据产物与 marker 约定

> 本文保存从 `CLAUDE.md` 迁出的完整目录、产物和 marker 细节；硬性摘要仍保留在 `CLAUDE.md`。

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
