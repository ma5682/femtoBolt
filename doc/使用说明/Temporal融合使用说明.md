# 固定场景 Confidence-based Temporal Fusion

这是独立、可选的采集/回放模式，现有单帧、TSDF、Mesh、3DGS入口不变。固定相机和固定场景缺一不可；不做ICP、相机Pose、多视角拼接，不生成单位世界Pose。

## 操作

### 从原有主程序使用（2026-09-09新增）

关闭旧版主窗口后重新运行 `start_femto_app.cmd`。主界面顶部“拍照”旁新增 **时域融合** 按钮；打开面板后选择5/10/15帧及fast/quality，点击“开始采集并融合”。无需关闭主程序相机，不要同时运行独立采集脚本。

主界面的相机线程是唯一SDK所有者。`temporal_start` 命令创建有界内存短序列，复用后续实时AlignedFrame，后台 `FemtoTemporal` 线程写raw/calibration/metadata并调用既有融合模块。新模块 `app_temporal.TemporalBurst` 持有帧拷贝和脱离SDK的标定快照，后台不访问相机对象；UI不做反投影和融合。采集中可“停止采集”，保留partial raw而不融合；超时、设备停止也交由证据线程保存已取帧。主界面同时禁用新的拍照、扫描、训练提交，实时RGB-D及原始点云预览仍继续。退出处理中的应用仍沿用原有退出确认，已启动的非daemon证据线程可完成保存。

面板提供“查看融合点云”“前后对照”“质量报告”“结果目录”。运动拒绝和未完成任务只允许查看已有报告/目录，不使用上一次成功模型冒充本次结果。点云用既有Open3D查看器独立进程显示；此操作只显示几何，不把丢失自定义confidence的重读点云送入融合。关闭面板不会取消正在进行的任务。

首次打开会回读最近的 `temporal_capture_*/derived/temporal_fusion_*/temporal_report.json`；兼容此前独立脚本输出，包括用户实拍 `temporal_capture_20260909_175452_474904`，不改旧重建/3DGS标记。新入口不会自动把时域PLY当作单帧模型或3DGS训练输入。

代码：修改 `femto_control_app.py` 与 `ui_i18n.py`；新增 `app_temporal.py`、`app_temporal_panel.py`、`tests/test_temporal_gui.py`。`start_femto_app.cmd` 本来就调用femto_control_app.py，因此脚本本身无需改变。7项GUI专项覆盖单SDK、预览继续、部分帧保存、后台失败、参数/按钮状态、结果查看、失败后重新开面板及窗口控件边界；整套回归208项，200通过、8环境跳过。日志 `outputs/temporal_gui_regression.log`。本轮使用模拟相机验证集成，不抢占用户已连接硬件，主界面实机FPS尚未复测。

已有固定数据集回放（项目根目录执行）：

```powershell
.venv/Scripts/python.exe -m femto.temporal.scene_temporal_fusion datasets/scene_20260909_103220 --mode fast --frames 5
.venv/Scripts/python.exe -m femto.temporal.scene_temporal_fusion datasets/scene_20260909_103220 --mode quality --frames 10
.venv/Scripts/python.exe -m femto.temporal.scene_temporal_fusion datasets/scene_20260909_103220 --mode quality --frames 15
```

`--start 20` 表示从第20个记录开始，要求真实不同且时间戳递增的帧。数据集必须已经声明fixed_camera=true，程序不把移动数据改标为固定。

新采集属于研发命令行功能：固定摄像头与场景，关闭占用相机的其他采集程序后执行下列命令，默认5帧；根目录的旧快捷 CMD 已移除。

```powershell
.venv/Scripts/python.exe -m femto.temporal.scene_temporal_fusion --capture --mode fast
.venv/Scripts/python.exe -m femto.temporal.scene_temporal_fusion --capture --mode quality
.venv/Scripts/python.exe -m femto.temporal.scene_temporal_fusion --capture --frames 10
```

fast默认5帧，quality默认15帧，均支持显式5/10/15帧。新采集沿用wide_detail、1920×1080 RGB、15FPS、250–2210mm、K6标定。取帧循环只做有限数量的帧拷贝；关闭相机后才编码PNG和进行质量分析。不会自动停止现有GUI相机，也不会覆盖当前模型指针。SDK中途异常/设备断开仍可能打断采集；常规读超时已获得的帧按incomplete_capture保存，禁止融合不完整序列。

## 调用关系与文件

| 文件/函数 | 职责 |
| --- | --- |
| orbbec_rgbd.OrbbecAlignedCamera.read | 原有采集、对齐、毫米Depth读取，未修改 |
| scene_temporal_fusion.capture_temporal_frames | 新增有限帧采集；raw PNG、timestamps、metadata、calibration |
| scene_temporal_fusion.build_temporal_dataset | 新增回放编排与源文件SHA-256核验；图片读取复用 `io_utils.imread_unicode` |
| scene_temporal_fusion.audit_temporal_motion | 复用assess_rgb_motion_pair；每相邻帧及首帧对各帧检查 |
| depth_quality.process_depth_quality | 原有每帧质量/补洞/confidence，结果仅作独立诊断 |
| temporal_depth_fusion.confidence_weighted_median / fuse_temporal_depth | 新增分块真实观测融合和稳定性统计 |
| scene_segmented_capture.depth_cloud | 原有K6反投影；新增显式temporal_confidence=True连续权重分支，旧默认仍仅接受0/.3/1 |
| depth_confidence_cloud.write_confident_point_cloud | 新增显式连续confidence的PLY属性输出，旧默认不变 |
| tests/test_temporal_depth_fusion.py | 十项测试，含五类场景模式、多帧、运动、SDK模拟采集顺序 |
| tools/verify_temporal_fusion.py | 生成五类合成模式比较图、JSON、1080p重复性能统计（2026-09-20 起移入 `tools/`，从仓库根目录运行） |

先审查调用链，再设计，再局部修改，最后回放验证。没有删除现有过滤、替换旧时域中值或修改3DGS格式。

## 真实性规则

1. 主要时域证据只来自原始有效Depth：真实权重1，无效权重0。每帧补洞结果保留在per_frame，但不会当成新增测量；因此没有“15个同样补点升级为高可信”的通路。
2. 8连通域、50mm空间误差门禁等原有每帧流程保留。时域估计允许不同于参考帧，但选取真实观测，不将偶数样本的两个中间值平均。空间补洞的“原有效值必须零变化”不用于否定独立派生时域估计。
3. 每像素至少3帧且不少于N的60%真实一致观测；距中值≤30mm定义inlier，inlier/有效观测≥80%、MAD≤15mm。少量飞点剔除，两层冲突/弱支持保持未知。1帧有效不会因为方差0而变高可信。
4. 最终confidence = inlier数/N × inlier数/有效观测数 × exp(-MAD/15mm)。不声称是概率或传感器精度。观测方差、MAD、真实支持数、一致支持数均保存。
5. 背景光流必须可靠、appears_fixed且中位位移≤1原图像素；任何帧对失败拒绝整个序列。局部RGB任一通道变化>35的像素扩张1像素后排除，可能保守排除曝光变化；RGB运动提示不等于材料识别或绝对静止证明。
6. 对齐网格/帧内参必须一致，RGB-D时间差超过5ms拒绝。参考RGB固定选择序列第1帧并记录路径，排除运动像素后再赋色，不混合不同时刻RGB。
7. 原始资料不变，派生点云为K6彩色相机RDF米制。没有创建世界Pose。长期稳定的透明/反光错误仍可能通过，五类实物验收未完成。

## 输出

新采集：`datasets/temporal_capture_<时间>/raw/frame_001_rgb.png`、`frame_001_depth_mm.png`、`manifest.json`、`metadata.json`、`frames.jsonl`、`calibration/`。计算产生的confidence属于派生层，不放raw。

每次回放：源目录下新建 `derived/temporal_fusion_<时间>/`：

- `per_frame/000/` 等：原有Depth Quality、Hole Filling、Confidence及Error Check产物。
- `fused_depth_mm.png`：uint16毫米，未知0。
- `temporal_confidence.npy`：float32精确权重；同名PNG是0–255预览，不能代替训练/融合精度。
- `fused_pointcloud.ply`：XYZRGB+每点float confidence。
- `real_support.npy`、`inlier_support.npy`、`variance_mm2.npy`、`mad_mm.npy`。
- `temporal_debug.png`：绿为confidence≥.8，黄为较低可信真实时域观测，红为未知；黄色不是空间补点。
- `comparison.png`：左首帧，右融合，红色覆盖未知区域；仅RGB+有效性预览，不是新视角重建图。
- `motion_report.json`、`motion_mask.png`、`temporal_report.json`。

运动拒绝时只保存失败报告和motion mask，不输出融合PLY。报告rate统一0–1，original_invalid_rate为选中首帧，另列均值和逐帧值；outlier observations是被剔除的采样次数，outlier pixels是至少有一次离群观测的唯一像素位置。

## TSDF与3DGS边界

按本轮确认的分阶段方案，本次交付到Depth/confidence/点云验证。连续confidence **不自动送进旧TSDF/3DGS**，避免旧10:3规则或0/.3/1读取器错误处理。旧流程完全保留。后续接入需要明确浮点权重融合接口和时域来源语义，禁止把本PLY经legacy Open3D丢失confidence后等权融合。

## 验证结果

201项主回归：193通过、8环境跳过；日志 `outputs/temporal_fusion_tests.log`。新增测试还覆盖重复时间戳拒绝、位移失败不输出模型、PLY持续confidence、SDK模拟读帧在编码之前完成。未在本轮连接硬件重新拍摄，也未测持续采集FPS。

合成白桌：单帧已知表面MAE约1.71mm；5/10/15帧约1.08/0.83/0.65mm，点数48598→49152；黑色物体随机缺失模式47227→49129→49152；手机玻璃交替深度区域拒绝，透明大孔保持未知。仅为预设深度模式，不冒充材料传感器物理模型。confidence可能随样本增多略降，因为更充分反映噪声；不是强行随帧数提高。

1080p纯融合最新平均5帧1.22秒、10帧2.39秒、15帧3.72秒（不同运行有波动）；不含逐帧质量/运动分析/XYZ/磁盘编码。输入Depth数组约20.7/41.5/62.2MB，完整RGB-D输入更多；本次验证进程峰值工作集238600192字节，约227.5MiB，含库加载，并非算法独占内存。采用48行分块控制中间数组，无实时帧循环内计算。最新结果：`outputs/temporal_validation_20260909_174821/validation.json`，同目录有五类合成模式四列对照图。最终收紧1px门禁后10项Temporal专项再次通过：`outputs/temporal_final_tests.log`。

实拍回放：`scene_20260909_103220`起始5/10/15帧全部拒绝，邻帧背景位移约5–9px；`scene_20260907_101251`起始5帧也拒绝。即使manifest声明fixed_camera也不能绕过检查。源哈希复核不变。故当前没有通过静止门禁的实拍改善数据，不能给出真实材料融合提升结论。
