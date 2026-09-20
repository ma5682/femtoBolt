# Temporal Confidence Fusion 第一阶段（2026-09-10）

状态：统一深度中间层、confidence 传播、固定视角关键帧建模已接入。TSDF integrate 权重修改按最新要求延后；3DGS 训练仍由原有按钮手动启动。

## 入口与调用链

- `start_femto_app.cmd` → `femto_control_app.main()` → `FemtoControlApp`。
- 实时预览继续使用 `_camera_worker()` 的 SDK 读取和现有预览更新；不在预览逐帧执行融合。
- 一键入口 `toggle_scan()` → `temporal_start` 队列 → 同一个相机线程收集 `TemporalBurst` → `_dispatch_temporal()` → 后台 `process_keyframe_burst()`。
- `TemporalBurst.save()` 保存原始 RGB、uint16 毫米 Depth、时间戳和标定。
- `build_temporal_keyframe()` → 每帧 `process_depth_quality()`（包含原有保守补洞和误差检查）→ `assess_temporal_scene()` → `fuse_quality_frames()` → `TemporalDepthLayer`。
- 中间层 → `build_confident_depth_cloud(..., temporal_layer=layer)` → `depth_cloud()` K6 反投影 → XYZRGB + 连续 confidence。
- 中间层派生关键帧 → `complete_demo.run_demo()` → `reconstruct_scene()` 固定视角像素相邻网格和 `scene_to_nerfstudio.convert()` 输入准备。
- 原有单视角训练按钮 → `scene_single_gs.prepare()` → `seed_arrays()` → 原有训练；连续 confidence 用于种子和损失，不重新赋成 1。

## 修改文件和函数

| 文件 | 变化 |
|---|---|
| `temporal_depth_layer.py`（新增） | `TemporalDepthLayer.validate()`、`load_temporal_layer()`，统一中间层及来源校验 |
| `temporal_keyframe.py`（新增） | `assess_temporal_scene()`、`fuse_quality_frames()`、`build_temporal_keyframe()`、`load_temporal_confidence()`、`prepared_depth()` |
| `app_temporal.py` | `TemporalBurst` 支持 1 帧；新增 `process_keyframe_burst()`，后台保存、融合、几何/训练输入准备 |
| `femto_control_app.py` | 一键质量选择默认 10 帧，支持 1/5/10/15；阶段进度、取消与失败状态恢复；原预览读取和绘制保留 |
| `depth_confidence_cloud.py` | `build_confident_depth_cloud()` 接收经过校验的 temporal layer；PLY 中写连续 confidence，修正来源注释 |
| `scene_reconstruction.py` | 读取融合输入及 confidence，保留原有清理和像素网格结构；单独输出原始参考点云、融合点云和 TSDF 输入包 |
| `scene_to_nerfstudio.py` | `convert()` 传播融合 confidence，按原 K6 最近邻重映射深度/confidence，训练输入结构不变 |
| `scene_gsplat_data.py` | `load_frame_confidence()` 接受连续 0..1，不量化 |
| `scene_single_gs.py` | `prepare()` 不对融合 Depth 重算质量；`seed_arrays()` 排除零 confidence；训练损失保留连续权重，opacity 上限尊重来源 confidence |
| `complete_demo.py` | 修正报告中“未做多帧合并”的过时描述，训练触发行为未改变 |
| `validate_temporal_keyframe.py`（新增，2026-09-20 起位于 `tools/`） | `compare_batches()`、`run()`，独立批次无 ICP 的 Chamfer、深度/边缘差异、点数、无效率和耗时对比 |
| `tests/test_temporal_keyframe.py`（新增） | 来源上限、连续传播、中间层、Depth 移动检测、GS 种子、TSDF 阶段隔离、一键 10 帧与取消测试 |
| `技术路线.md` | 同步本阶段边界和验证记录 |

## 数据和 confidence

`Raw Depth → Depth Quality → Hole Filling → Confidence → Temporal Fusion → 最终 Depth → K6 → XYZRGB → 固定视角建模/GS 输入/TSDF 输入准备`。

新增结果位于 `源采集目录/derived/temporal_fusion/时间戳/`：

```text
per_frame/                      每帧原始副本、质量、补洞、误差检查和 confidence
fused_depth_mm.png              uint16，毫米，0 未知
fused_confidence.npy            float32 权威连续值
fused_confidence.png            0..255 量化预览，不用于融合
final_confidence.npy            相同连续值，审计兼容名称
source_count.npy                深度内点支持数
original_support_count.npy      原始观测内点数
filled_support_count.npy        补洞内点数
selected_original.npy           所选深度是否有原始观测支持
temporal_variance.npy           全部有效样本加权方差，mm²
temporal_stability.npy
depth_quality_confidence.npy
scene_motion_mask.png
motion_report.json
temporal_report.json
fused_pointcloud.ply            原 XYZRGB 类型/单位 + float confidence
keyframe/                      原始参考 RGB-D 和融合输入、标定、固定视角局部 Pose
models/                        一键产生的几何和 GS 输入；不启动训练
```

`final_confidence = depth_quality_confidence × temporal_stability`。
quality 部分是所选来源 confidence 与内点平均 confidence 的较小值；stability 为支持比例 × 内点权重占比 × exp(-MAD/15)。只选实际候选深度的 weighted median，不求深度均值。补洞来源上限 0.3；没有支持或移动可疑区域为 0。

时域内点差 ≤30 mm、内点权重占比 ≥80%、MAD ≤15 mm、支持 ≥60% 且至少 3 帧。存在真实观测时须有足够真实支持，补洞不能投票压倒相矛盾的真实深度。1 帧不声称具备时域稳定性证据。

RGB 使用现有光流和局部差分检查。连续原始有效 Depth 的 >50 mm 跳变若反复出现，会输出 `scene_motion_detected` 并将对应像素权重降为 0。全局移动或静止证据不足时保留诊断层，禁止生成可信几何。此标记是异常提示，不是对材料或真实运动的确定识别。

所有原始 RGB-D、原有 PLY 不覆盖；源文件 SHA256 写入报告。`temporal_observed_mm.png` 是时域候选中有真实观测支持的深度，并非独立真值。PLY 的自定义 confidence 在 legacy Open3D 普通读取后可能丢失，下游应读取权威深度中间层。

## TSDF 阶段边界

本轮已撤回尝试中的浮点 TSDF 实现，没有修改旧 TSDF integrate 的 10:3 原始/补洞权重逻辑。已有非 temporal 多帧 TSDF 路线保持可用。

固定视角路径额外生成 `reconstruction/derived/tsdf_input/000000.npz`：米制 rectified depth、连续 confidence、RGB、内参和 T_CW，供第二阶段消费。

**融合数据直接进入旧 TSDF integrate 暂不启用**，会明确提示第二阶段尚未接入，避免丢失连续权重或错误等权融合。本阶段完整一键固定视角路线保留原有 0.25–2.21 m 条件；其他范围若原流程选择 TSDF，则需等第二阶段接入融合权重。

## 验证结果

最终回归命令：`.venv\Scripts\python.exe -m unittest discover -s tests -q`。214 项中 206 项通过、8 项跳过。没有启动硬件采集或 3DGS 训练。

真实保存序列 `temporal_capture_20260909_175452_474904` 只有 5 帧，没有重复帧伪造 10 帧实测。

| 保存数据回放 | 无效率 | 点数（清理前） | 有效点平均 confidence |
|---|---:|---:|---:|
| 1 帧 | 68.7359% | 648292 | 1.0000 |
| 5 帧 | 68.8056% | 646847 | 0.9700 |

5 帧完整链路输出清理后 646846 点、1262739 个三角面，原始证据哈希一致。无效率略增来自不稳定区域拒绝，不代表覆盖提高。

192×256 合成噪声/缺失/反光式跳变场景，每种模式取 3 个不重复批次：

| 模式 | 平均无效率 | 平均点数 | 相邻批次 Chamfer 均值 mm | 边缘共同有效深度差均值 mm |
|---|---:|---:|---:|---:|
| 1 帧 | 0.1729% | 49067 | 3.6774 | 6.0273 |
| 5 帧 | 1.0451% | 48638 | 2.1447 | 1.9619 |
| 10 帧 | 0.2964% | 49006 | 1.6349 | 1.5045 |

共同有效区域指标必须和覆盖率一起解读：拒绝不稳定区域也会降低差异。数据不证明玻璃/透明物体真实表面的精度提高。合成数据不能替代普通桌面、黑色物体、手机玻璃、透明物体、设备外壳的实拍验证。

详细结果：`datasets/temporal_capture_20260909_175452_474904/derived/temporal_fusion/phase1_validation_v2/comparison.json`。初版低分辨率合成场景因 RGB 纹理证据不足被全部拒绝，保留在 `phase1_validation/`；v2 提高了图像分辨率以具备足够静态特征，没有放宽运动门限。

真实回放纯处理约 0.49 秒/单帧批次、2.41 秒/5帧批次，不含采集和模型导出。数组内存约 21.75/108.76 MiB，非完整进程峰值。实时硬件 FPS、10 帧真实重复采集、TSDF 权重实测留待后续验证。本阶段不自动进入第二阶段。
