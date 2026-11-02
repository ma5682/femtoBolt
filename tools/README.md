# tools/ — 研发实验、安装辅助与分层检查

工作台端到端离线测量使用 `benchmark_workstation.py`；完成后用
`audit_workstation_benchmark.py` 检查重复输出。操作与指标边界见
[固定数据性能测量](../doc/使用说明/工作台固定数据性能测量.md)。

这里主要是安装辅助、只读环境检查和历史实验复现脚本；不作为 `femto/` 业务模块依赖。`check_workspace_runtime.py` 有根目录检查入口和专项测试，实验脚本则用于复现历史交付记录中的对照实验。

| 脚本 | 用途 | 记录 |
|---|---|---|
| `history_metadata.py` | 检查 SQLite 历史索引与一致性备份，仅处理元数据 | [SQLite历史记录与备份](../doc/使用说明/SQLite历史记录与备份.md) |
| `simulate_devices.py` | 共享业务状态机的无 Qt 模拟运行、故障注入、独立合成间隙、模拟事件导出与只读回放；不控制真实设备 | [设备管理与模拟联调](../doc/使用说明/设备管理与模拟联调.md) |
| `verify_exact_target_partition.py` | 真实保存方案经精确像素分区重算、覆盖、原生/便携保存及独立回读，分阶段保留失败与成功证据 | [精确分区核验记录](../doc/历史开发记录/精确像素分区核验_实现与验证_20261015.md) |
| `verify_paged_product_flow.py` | 原真实工程经目标续算、完整覆盖、原生/便携保存及独立回读；各操作保持120秒预算，失败阶段单独留档 | [三项业务完善阶段记录](../doc/历史开发记录/三项业务完善_实施进度_20261015.md) |
| `diagnose_gap_observations.py` | 将阶段排查中的参考点投向已接纳原始深度，按记录位姿核对10 mm一致性，保留逐帧残差和来源摘要；不作为独立位姿精度判定 | [三项业务完善阶段记录](../doc/历史开发记录/三项业务完善_实施进度_20261015.md) |
| `audit_reconstruction_gaps.py` | 原采集范围内“有参考深度但无网格”的同参数分阶段重建排查；输出到独立新目录，不发布模型 | [三项业务完善阶段记录](../doc/历史开发记录/三项业务完善_实施进度_20261015.md) |
| `verify_target_paging.py` | 对已有原目标工程验证单次续算、累计分块保存和独立回读；不是覆盖核验或正式方案导出 | [三项业务完善阶段记录](../doc/历史开发记录/三项业务完善_实施进度_20261015.md) |
| `check_workspace_runtime.py` | 只读检查解释器、源码与核心依赖归属，支持 JSON 和仅采集环境 | 根目录 `检查工作区环境.cmd` |
| `check_nominal_coverage.py` | B1 名义表面覆盖独立计算及原输入复算，使用数值案例或已核验的 P1 便携方案；结果仅进入新的 outputs 子目录 | [P2／B1 契约](../doc/技术资料/P2_B1_名义覆盖验证契约.md) |
| `benchmark_surface_coverage.py` | 128／2,048 面、同输入 P2 优先复证、P3 v5 与完整资产回放；支持固定引导及旧 P2／v4 同起点对照，分别记录原始候选和接纳结果、全部指标及失败诊断 | [重复验收与诊断](../doc/使用说明/覆盖性能重复验收与失败诊断.md) |
| `check_body_surface_replay.py` | 自动环境预检、仅枚举设备、现有模型/方案只读回放及可选原生 VTK 显示核验；不启动新采集 | [自动预检与历史回放验收](../doc/使用说明/自动预检与历史回放验收.md) |
| `diagnose_route_continuity.py` | 同源重算原连续候选与质量约束结果，按扫描/连接边诊断断线；可隔离实验保留真实面交点，不改产品算法 | [蛇形连贯性同源诊断](../doc/历史开发记录/蛇形连贯性退化_同源复现诊断_20260930.md) |
| `benchmark_depth_quality.py` | 单帧深度补洞的内存内重复计时，不含文件 IO | `doc/技术资料/Depth补洞修改说明.md` |
| `diagnose_local_span.py` | 局部会话的只读 SIFT + 原始深度 + PnP 交叉检查，不进入生产跟踪链 | `doc/历史开发记录/固定RGB选区与位姿排查交付记录_20260915.md` |
| `scene_gsplat_experiments.py` | 顺序验证集调参 + 一次 test 评估的 gsplat 超参对照（需要 `.venv-gsplat`） | `技术路线.md` 3DGS 章节 |
| `validate_temporal_keyframe.py` | 时域关键帧独立批次 Chamfer / 深度差 / 耗时对比 | `doc/技术资料/Temporal关键帧第一阶段说明.md` |
| `verify_temporal_fusion.py` | 五类合成材质模式的时域融合对比图与 1080p 性能统计 | `doc/使用说明/Temporal融合使用说明.md` |
| `download_model.py` | 下载并校验 MediaPipe Pose Landmarker 模型到 `models/`（`setup_environment.cmd` 调用） | — |
| `download_verified_file.py` | 分块校验下载大文件（`setup_direct_gsplat_environment.cmd` 调用） | — |
| `patch_gsplat_windows.py` | 对已安装 gsplat 1.5.3 做 MSVC 编译标志补丁（`setup_direct_gsplat_environment.cmd` 调用） | `SOURCE_PROVENANCE.md` |

运行方式：从仓库根目录执行，脚本开头会把仓库根目录加入 `sys.path` 以便 import `femto` 包。

```powershell
.venv\Scripts\python.exe tools\verify_temporal_fusion.py
.venv\Scripts\python.exe tools\diagnose_local_span.py <scene_dir>
```

输出仍写到根目录 `outputs/`，不会写进 `tools/`。这些脚本的结果只是软件 / 合成 / 历史回放证据，不是实物验收。

已完成的包结构与分层迁移脚本于 2026-09-28 移除，历史过程见 Git 和 `doc/历史开发记录/包结构迁移交付记录_20260920.md`。
