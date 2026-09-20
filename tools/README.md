# tools/ — 研发实验与一次性诊断脚本

这里的脚本不是产品链路的一部分：没有 `.cmd` 入口，没有被任何根目录模块 import，也没有单元测试覆盖。它们保留下来是为了复现历史交付记录里的对照实验。

| 脚本 | 用途 | 记录 |
|---|---|---|
| `benchmark_depth_quality.py` | 单帧深度补洞的内存内重复计时，不含文件 IO | `doc/技术资料/Depth补洞修改说明.md` |
| `diagnose_local_span.py` | 局部会话的只读 SIFT + 原始深度 + PnP 交叉检查，不进入生产跟踪链 | `doc/历史开发记录/固定RGB选区与位姿排查交付记录_20260915.md` |
| `scene_gsplat_experiments.py` | 顺序验证集调参 + 一次 test 评估的 gsplat 超参对照（需要 `.venv-gsplat`） | `技术路线.md` 3DGS 章节 |
| `validate_temporal_keyframe.py` | 时域关键帧独立批次 Chamfer / 深度差 / 耗时对比 | `doc/技术资料/Temporal关键帧第一阶段说明.md` |
| `verify_temporal_fusion.py` | 五类合成材质模式的时域融合对比图与 1080p 性能统计 | `doc/使用说明/Temporal融合使用说明.md` |

运行方式：从仓库根目录执行，脚本开头会把仓库根目录加入 `sys.path`。

```powershell
.venv\Scripts\python.exe tools\verify_temporal_fusion.py
.venv\Scripts\python.exe tools\diagnose_local_span.py <scene_dir>
```

输出仍写到根目录 `outputs/`，不会写进 `tools/`。这些脚本的结果只是软件 / 合成 / 历史回放证据，不是实物验收。
