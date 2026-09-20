# Depth质量检测与保守补洞：本轮交付

## 新增第十一至十五项（最新，覆盖下方历史指标）

本次新增 `depth_error_check.py`、`depth_confidence_cloud.py`、`depth_debug_viewer.py`、`start_depth_debug.cmd`、`benchmark_depth_quality.py`（2026-09-20 起位于 `tools/`）、`tests/test_depth_error_debug.py`；局部修改 `depth_quality.py`、`scene_segmented_capture.py`、`femto_control_app.py`、`scene_reconstruction.py`、`depth_quality_replay.py` 和技术路线。原有过滤未删除。

误差检查比较原始有效像素与修复后同坐标Depth，任何变化拒绝输出；另对补点检查与原始有效8邻域的最大差，超过50mm撤回整个孔洞，Depth和confidence归零。所有输出新增MAE/max/histogram报告。缺失像素无真值，因此单独记录邻域一致性指标，不将0值当真值、不声称补点真实误差。

派生K6点云在生成前再次复核，逐点携带confidence，PLY增加 `property float confidence`。真实=1、修复=0.3、未知不生成点。TSDF保留10:3积分及原过滤，3DGS保留confidence加权。普通Open3D PLY重读会丢弃confidence，不能用它替代融合的Depth+confidence输入。

最新实拍quality回放：**0.744985%→0.742091%，修复60像素**；原始有效点MAE/max均0mm；补点最大邻域偏差均值11.6mm、最大38mm。fast只分析孤立单像素孔，此帧0修复。源证据哈希不变。结果位于：

`datasets/scene_20260909_103220/derived/fixed_keyframe_20260909_103230/derived/depth_quality_replay_20260909_172112_593738/`

双击项目根目录 `start_depth_debug.cmd`，选择上述目录即可打开四模式查看器；绿=真实、黄=修复、红=未知，1:1与放大可检查小孔，鼠标悬停读取原始/修复毫米值和confidence。或者在项目根目录运行：

```powershell
.venv/Scripts/python.exe -m femto.viewer.depth_debug_viewer datasets/scene_20260909_103220/derived/fixed_keyframe_20260909_103230/derived/depth_quality_replay_20260909_172112_593738
.venv/Scripts/python.exe -m femto.pipeline.depth_quality_replay datasets/scene_20260909_103220/derived/fixed_keyframe_20260909_103230 --mode fast
.venv/Scripts/python.exe -m femto.pipeline.depth_quality_replay datasets/scene_20260909_103220/derived/fixed_keyframe_20260909_103230 --mode quality
```

quality完整CC及RANSAC复核；fast仅检测8个原始邻居均有效的孤立孔，保留所有真实性门禁。10次预热后内存处理计时中位：fast161.7ms、quality202.4ms，不含磁盘/XYZ；均不适合直接同步接入15FPS循环。当前采集循环不运行本模块，实拍FPS尚未复测。

191项主环境测试：183通过、8跳过；独立PyTorch2项通过。测试含真实Open3D TSDF积分、PLY逐点confidence、50/51mm门禁、四模式颜色、Tk切换和缩放。五类材料实物验收仍未完成。

实际项目为 `E:\测试3D人体关节坐标`。审查时项目已有Depth质量模块和TSDF/3DGS confidence接入；本轮是在现有实现上加固，未重建采集架构。

## 修改文件与函数

| 文件 | 本轮变更 |
| --- | --- |
| depth_quality.py | 新增 `DepthQualityConfig.__post_init__` 校验；改进 `assess_and_fill_depth` 局部连通域分析、极差门禁、分类报告 |
| femto_control_app.py | 修改 `save_rgbd_photo`，XYZ之前评估质量，独立保存派生PLY与质量文件 |
| scene_to_nerfstudio.py | 修改 `convert`，新导出的补洞Depth/Confidence移入derived，更新transforms引用 |
| depth_quality_replay.py | 新增 `replay_depth_quality`，离线回放、K6派生点云与源文件SHA-256核验 |
| tests/test_depth_quality.py | 新增6项回归，覆盖边界量程、全局门禁分类、深度阶跃、配置、多连通域坐标、局部策略 |
| tests/test_depth_quality_pipeline.py | 新增拍照保存和真实Open3D TSDF积分管线测试 |
| tests/test_scene_to_nerfstudio.py | 更新并验证derived输入路径 |
| 技术路线.md | 同步当前参数、输出、测试结果，消除旧TSDF空间采样说明冲突 |

原有公共函数 `invalid_depth_mask`、`assess_and_fill_depth`、`save_depth_quality`、`process_depth_quality` 保持调用兼容。原有 `confidence_acceptance_mask` 保留兼容，但实际TSDF不使用它。

## 流程与真实性

Femto Bolt → pyorbbecsdk2 → Depth/Color Align → uint16毫米Depth留档 → 无效像素统计与Mask → 8连通域 → 3%/10%全局门禁 → 小孔面积/宽高/图像边界检查 → 原始8邻域支持≥60%、std<30mm、周边极差≤60mm → Edge-aware加权中值，估计不可用时才尝试局部中值/RANSAC后备 → confidence → derived落盘 → 原有K6/米制几何管线。

每个像素至少5/8原始邻居有效，整块必须全部通过；不使用已补点递推，不做全图模糊，不强求填满。默认自动孔洞面积≤64像素、局部模式≤24像素、宽高≤12像素；邻域门禁可拒绝面积达标的孔洞。透明/反光只是深度异常的疑似原因，分类不是材料识别。

原始有效Depth confidence=1，补点=0.3，未知=0；超量程值仅在派生Depth中归零。原RGB、原depth_mm.png和原PLY不替换。当前原始点云、单帧原始Mesh均不加入补点。修复PLY是单独的低可信显示产物，不能将其新增点当作真实测量精度。

TSDF沿用现有10次真实/3次修复独立积分，保持相对1:0.3贡献；无补点帧也保持10次，因此增加运行成本。多帧3DGS种子来自原始深度，训练Depth损失按confidence加权，验证仅用原始深度。单帧GS允许修复种子，opacity上限0.3，并使用confidence加权训练。第三方训练器只有读取confidence字段才具备该加权语义。

## 实测与测试

实拍源：`datasets/scene_20260909_103220/derived/fixed_keyframe_20260909_103230`。

| 指标 | 本轮原始对齐像素网格回放 |
| --- | --- |
| 总像素 | 2,073,600 |
| 修复前无效 | 15,448（0.744985%） |
| 修复像素 | 64 |
| 修复后无效 | 15,384（0.741898%） |
| repair_rate | 64/15448 = 0.414293%（原无效像素作分母） |
| 拒绝大/触边区域 | 10 |
| 拒绝深度不连续区域 | 10 |
| 拒绝低支持区域 | 29 |

结果目录：源数据集下 `derived/depth_quality_replay_20260909_154836_711987/`，内含 `depth_quality.json`、`quality_report.json`、原始Depth副本、invalid_mask、filled_depth、confidence.npy/.png、派生PLY及 `replay_verification.json`。RGB、Depth和标定等源文件哈希不变，原始有效深度逐值不变。该统计在K6图像重映射之前，不能与历史去畸变网格的无效率直接比较。

182项主环境回归：174通过、8环境跳过；独立PyTorch环境2项confidence测试通过。五类场景为合成模式测试，另有上述未标注材料的实拍回放；五类实物验收尚未完成。透明物体前后背景深度连续时，仅凭此算法无法证明新增几何真实。

离线回放命令（项目根目录）：

```powershell
.venv/Scripts/python.exe -m femto.pipeline.depth_quality_replay datasets/scene_20260909_103220/derived/fixed_keyframe_20260909_103230
```

每次运行生成新的derived目录，不开启相机、不训练、不修改当前模型或照片指针。
