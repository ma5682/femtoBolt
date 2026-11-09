# 源码来源与独立实现规则

## 项目自研代码

本目录中的统一操作页面、实时预览、单帧 RGB-D 照片、采集控制、帧元数据读取、数据集格式、相机标定落盘、数据校验、深度反投影、筛帧、点云、顺序 RGB-D 里程计适配、全场景 TSDF、网格修复、合成真值测试、查看器、Nerfstudio 转换、RGB-D 初始点 PLY 写入和 Gaussian 初始化 PLY 写入器，按项目需求独立编写。算法可以依据公开论文、标准数学公式和官方 API 文档实现，但不得从许可证不兼容的仓库复制源码、测试、注释或资源。

项目权利人尚未在仓库中登记。对外发布前必须由实际公司/个人确认著作权归属、员工或承包方权利转让，以及产品最终许可条款。

2026-11-04：`native/cloud_predicate.c`、`femto/geometry/cloud_radius_kernel.py` 和 `tools/build_cloud_kernel.py` 按当前项目的八子盒包络与完整半径 NumPy 判据独立编写。扩展使用 CPython buffer API 与标准 C 算术，不复制或链接新的第三方算法实现，不使用 NumPy C ABI。构建使用本机已有 MSVC 14.44.35207、CPython 3.12.10 头文件及 setuptools 84.0.0；运行环境 NumPy/Open3D 版本保持。源码、构建脚本、编译器、实际命令和 `.pyd` SHA-256 随每次构建记录在 `outputs/cloud_kernel_build_*/build.json`；本轮验证产物的具体记录见[点云本地内核提速](doc/技术资料/点云本地内核提速_20261104.md)。生成二进制不入 Git，不在应用运行时自动编译或下载。

## 有来源标记的允许适配

2026-11-02：`geometry/organized_surface.py`、`local/organized_registration.py` 和相应对照/测试根据 PCL 官方文档中的积分图平均梯度、投影匹配、法向方向采样及有序边缘标签概念独立实现。反投影、积分求和、方向分桶、小位姿最小二乘及合成场景使用本项目 NumPy/OpenCV/Open3D 公共接口；没有复制 PCL 源码或链接 PCL。文档方法链接与实验入口见 `doc/技术资料/有序RGBD算法对照.md`；原始 RGB-D 不改写，实验报告与模型发布隔离。

2026-09-08 新增 `live_point_cloud.py`：按用户指定参考本地 `E:\测试\05_point_cloud.py` 的 Open3D SceneWidget、相机原点取景、主要深度层目标及显示颜色交互方式。当前项目独立实现父子进程数据传递、关闭流程、K6 射线缓存与米制几何；没有运行时导入参考工程或复制其 SDK 相机采集、BAG、导出与历史合并流程。显示调用已登记 Open3D 0.19.0 的公共 [SceneWidget API](https://www.open3d.org/docs/release/python_api/open3d.visualization.gui.SceneWidget.html) 与 [Window API](https://www.open3d.org/docs/release/python_api/open3d.visualization.gui.Window.html)，渲染资源复用现有运行时缓存，中文字体使用本机 Windows 已有字体。

| 文件/范围 | 来源 | 许可证 | 本项目修改 |
|---|---|---|---|
| `orbbec_rgbd.py` 的颜色帧格式解码 | Orbbec `pyorbbecsdk/examples/utils.py` | Apache-2.0 | 改为严格 buffer shape、统一 BGR、异常抛出并接入本项目 `AlignedFrame` |
| `pose_geometry.py` 的 33 点名称/连接关系 | MediaPipe Pose 公共 API/文档 | Apache-2.0（代码） | 转为本项目 NumPy 几何、深度邻域和区域标签结构 |
| `patch_gsplat_windows.py` 对已安装 `gsplat/cuda/_backend.py` 的补丁 | nerfstudio-project/gsplat 1.5.3 | Apache-2.0 | 仅把 Windows 主机编译器的 GCC 专用 `-Wno-attributes` 换为 MSVC `/O2`（FAST_COMPILE 时 `/Od`），nvcc 标志保持 `-O3`/`-O0`；不改算法或 CUDA 内核；补丁在源码与审阅版本不一致时拒绝执行 |

## 禁止进入产品源码

- `graphdeco-inria/gaussian-splatting`：只允许阅读论文、理解算法和进行黑盒结果比较；未经 INRIA 商业授权，不复制、不链接、不检出到产品仓库。
- 未登记许可证的模型、数据集、Web renderer、纹理、图标和示例资产。
- 带 GPL/AGPL 强传染条款且未经过发布架构和法务评审的组件。

## 3DGS 自研方式

2026-09-08：`single_frame_mesh.py` 与 `single_frame_quality.py` 的相邻像素三角化、边界门禁、UV、原始文件校验和质量统计为项目独立实现。仅使用已登记的 NumPy/OpenCV/Open3D/Pillow/trimesh 公共 API，无新增依赖。纹理来自用户本机同一张 Femto RGB-D 的 RGB 文件，血缘保存在模型报告与 source_evidence.json；不使用外部纹理或补全模型。

1. 由本项目定义 Femto RGB-D、内参、`T_WC` 到训练数据的转换。
2. 训练与栅格化只通过 Apache-2.0 `gsplat` 公共 API；适配、损失组合、深度约束、导出和坐标转换由本项目独立实现。
3. 每个 3DGS 模块提交时记录所依据的论文、API 版本、输入数据血缘和许可证。
4. Graphdeco 结果只能作为验收对照，不能作为产品运行依赖或源码模板。
5. `scene_gaussian.py` 只生成标准属性的未训练初始化 PLY；`scene_gsplat_trainer.py` / `scene_gsplat_data.py` / `scene_gsplat_export.py` / `scene_gsplat_metrics.py` 只调用 `gsplat` 公共 API，损失组合、米制深度约束、验收口径、检查点、PLY/SPZ 导出和坐标转换均为本项目独立实现。SSIM 用公开公式与 PyTorch 公共算子自写；LPIPS 直接调用 torchmetrics 公共接口，不复制其内嵌的 richzhang 代码。


## 2026-09-04 真实数据闭环适配

`gsplat_backend.py`、`scene_gsplat_acceptance.py`、`scene_gsplat_viewer.py`、`scene_scale_validation.py` 复用现有实现并作最小修复。来源审计、进程/界面控制、几何测距和报告逻辑均为项目独立代码；Gaussian 渲染仍只调用 gsplat 1.5.3 公共 `rasterization`。没有复制 Graphdeco 源码；本阶段新增的 Niantic `spz==1.1.0`（官方 tag v3.0.0）及 zlib/zstd 构建依赖已登记到 `THIRD_PARTY_NOTICES.md` 和 `licenses/`，并保留实际编译文件中的 Adobe MIT、Zstandard 旧解码器 BSD-2 与 libdivsufsort-lite MIT notice；没有新增未登记权重。已有开发合规检查不代表客户发布许可已完成。

SPZ Python 扩展通过官方源码的 nanobind CMake 路径构建。实际构建输入固定为 nanobind `v3.0.1` / commit `db4827f06f6f1680e5d4004c95fc8d69299dba8b`，本机 wheel SHA-256 为 `4b49491fe8bf483a5e0342125b8fe56237e490ba642c7e296170b8cd256a7764`；该 wheel 含 nanobind 上游固定的 `tsl::robin-map` `v1.4.0` / commit `4ec1bf19c6a96125ea22062f38c2cf5b958e448e` 头文件。zstd 1.5.6 静态构建另覆盖内嵌 libdivsufsort-lite 与旧解码器 notice；运行时 zlib 只接受固定源码构建的 1.3.1 DLL，不再使用 torchvision 1.2.13 候选。上述非独立 distribution 由源码归档/构建产物 SHA-256、`spz-build-constraints.txt` 和 `commercial_compliance_check.py` 的 `spz_static_components` 单独审计；对应许可证快照均在 `licenses/`。
