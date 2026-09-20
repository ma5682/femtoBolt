# 第三方依赖与许可证记录

本清单用于工程筛选和发布追踪，不替代正式法律意见。新增、复制、修改或分发第三方代码前，必须在这里登记固定版本、许可证、修改内容和交付方式。

当前发行义务和阻断项以 [`COMMERCIAL_READINESS.md`](./COMMERCIAL_READINESS.md) 为准；许可证原文快照保存在 [`licenses/`](./licenses/)。

## 当前运行依赖

| 组件 | 当前版本 | 许可证/条件 | 商用状态 |
|---|---:|---|---|
| `pyorbbecsdk2` | 2.1.2 | Python wrapper Apache-2.0；Orbbec Extension 仅可随真实 Orbbec 硬件使用，分发需附 EULA，并遵守其三方 LGPL 等条款 | 有条件允许 |
| MediaPipe Tasks | 1.0.1 | Apache-2.0 | 允许；需附许可证并核对遥测/隐私告知 |
| Pose Landmarker Full bundle | float16/1，SHA-256 `5134a3...11b1` | 官方 BlazePose GHUM 模型卡标注 Apache-2.0 | 允许；随模型附许可证和模型卡 |
| Open3D | 0.19.0 | MIT | 允许；保留版权和许可证 |
| Trimesh | 5.1.0 | MIT | 允许；当前只使用内置网格处理，不调用未知外部工具 |
| NumPy | 2.2.6 | BSD-3-Clause 及 wheel 内兼容依赖 | 允许；分发完整 wheel notices |
| `opencv-contrib-python` | 5.0.0.93 | wrapper MIT、OpenCV Apache-2.0、wheel 含 FFmpeg/LGPL 等三方组件 | 有条件允许；分发完整第三方许可证并满足 LGPL |
| Pillow | 12.3.0 | MIT-CMU；wheel 许可证目录同时列出其编解码三方组件 | 允许；分发完整 wheel LICENSE 并保留版权声明 |
| Niantic `spz` Python binding | 1.1.0（由源码 tag v3.0.0 构建） | MIT；官方源码 commit `5bf2945de1a003cee07133b1e495fe9c6ffdc7e7`，源码归档 SHA-256 `7f0f08ab144ecc940475b4edbfbb967d504aa2c78a1fbed17fd857bef8d95665`；实际编译的 `load-spz.cc` 另含 Adobe Inc. 2025 MIT 版权 | 允许；保留 [`licenses/spz/LICENSE`](licenses/spz/LICENSE) 与 [`licenses/spz/ADOBE_NOTICE.txt`](licenses/spz/ADOBE_NOTICE.txt)，SPZ 仅为 PLY 压缩交付格式 |

## 可进入商用实现

### nerfstudio-project/gsplat

- 来源：<https://github.com/nerfstudio-project/gsplat>
- 用途：3D Gaussian Splatting 训练、CUDA 栅格化和 Python 后端。
- 许可证：Apache License 2.0。
- 当前状态：`gsplat==1.5.3` 已安装在独立产品运行时 `.venv-gsplat`（Python 3.10、CUDA 12.8、RTX 5060 `sm_120` JIT 编译通过），由 `scene_gsplat_trainer.py` 只通过 `rasterization` / `DefaultStrategy` 公共 API 调用；2026-09-03 完成 800 步 GPU 验收。未复制上游源码进入项目。
- 本项目修改：`patch_gsplat_windows.py` 对已安装的 `gsplat/cuda/_backend.py` 做可复现的 MSVC 编译标志补丁（Windows 上把 GCC 专用 `-Wno-attributes` 换成 `/O2` / `/Od`，nvcc 标志不变），不改任何算法或 CUDA 内核源码；补丁只接受与审阅过的 1.5.3 源码完全匹配的文件。
- 发布要求：随分发物保留 Apache-2.0 LICENSE、版权声明及仓库中的 NOTICE（若存在）；发布说明中记录上述编译标志修改。
- 注意：`gsplat/rendering.py` 含一个可选的 Graphdeco `diff_gaussian_rasterization` 兼容包装函数，本项目不调用；该包**未**安装在 `.venv-gsplat`，`commercial_compliance_check.py` 会阻断其出现。

### 直接 3DGS 产品运行时（`.venv-gsplat`）

版本由 `requirements-gsplat-lock.txt`、官方 cu128 wheel 与 `commercial_compliance_check.EXPECTED_GSPLAT_PACKAGES` 钉死；快照在 `licenses/` 对应目录。

| 组件 | 当前版本 | 许可证/条件 | 商用状态 |
|---|---:|---|---|
| PyTorch (`torch`) | 2.7.1+cu128 | BSD-3-Clause；wheel 内 LICENSE/NOTICE 列出捆绑三方组件（含 NVIDIA CUDA 运行库的再分发条款） | 有条件允许；分发完整 LICENSE 与 NOTICE，并核对 CUDA 运行库再分发条件 |
| torchvision | 0.22.1+cu128 | BSD-3-Clause（代码） | 允许；仅用于 LPIPS 的 AlexNet 特征主干 |
| torchmetrics | 1.9.0 | Apache-2.0；LPIPS 模块内嵌 richzhang/PerceptualSimilarity 的 BSD-2 代码与线性头权重（`lpips_models/alex.pth`） | 允许；同时保留 Apache-2.0 与 BSD-2 声明（见 `licenses/torchmetrics/`、`licenses/lpips/`） |
| lightning-utilities | 0.15.3 | Apache-2.0（torchmetrics 传递依赖） | 允许 |
| imageio | 2.37.0 | BSD-2-Clause | 允许 |
| tqdm | 4.67.1 | MPL-2.0 + MIT 双重声明 | 允许；保留 LICENCE 文本，不修改其源码 |
| NumPy / Pillow | 2.2.6 / 11.3.0 | 同主环境条款 | 允许 |
| zlib（SPZ 构建/运行时） | 1.3.1 | zlib License；源码归档 SHA-256 `9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23`；当前部署 DLL SHA-256 `2f08f389724c8a05c672eeb3dec1ac1ff7f589d133d4011d807e489bee522baa`，运行时 `zlibVersion()` 已核为 1.3.1 | 允许；随 SPZ 运行库保留 [`licenses/zlib/LICENSE.txt`](licenses/zlib/LICENSE.txt) |
| Zstandard（SPZ 构建，静态链接） | 1.5.6 | BSD-3-Clause；源码归档 SHA-256 `8c29e06cf42aacc1eafc4077ae2ec6c6fcb96a626157e0593d5e82a34fd403c1`；实际构建还覆盖 Yann Collet 旧解码器 BSD-2、Yann Collet/Meta xxhash 与 Tino Reichardt threading 文件级版权 | 允许；保留 [`licenses/zstd/LICENSE`](licenses/zstd/LICENSE) 与 [`licenses/zstd/LEGACY_BSD2_NOTICES.txt`](licenses/zstd/LEGACY_BSD2_NOTICES.txt) |
| libdivsufsort-lite（Zstandard 1.5.6 内嵌构建源码） | Zstandard 1.5.6 归档内快照 | MIT；Copyright 2003-2008 Yuta Mori；由上述 Zstandard 归档 SHA-256 覆盖 | 允许；保留 [`licenses/divsufsort/LICENSE`](licenses/divsufsort/LICENSE) |
| nanobind（SPZ Python binding 构建，静态编入 `.pyd`） | 3.0.1 | BSD-3-Clause；官方 tag `v3.0.1`、commit `db4827f06f6f1680e5d4004c95fc8d69299dba8b`；固定 wheel SHA-256 `4b49491fe8bf483a5e0342125b8fe56237e490ba642c7e296170b8cd256a7764` | 允许；`spz-build-constraints.txt` 钉死版本，保留 [`licenses/nanobind/LICENSE`](licenses/nanobind/LICENSE) |
| `tsl::robin-map`（nanobind vendored 头文件，静态编入 `.pyd`） | 1.4.0 | MIT；官方 tag `v1.4.0`、commit `4ec1bf19c6a96125ea22062f38c2cf5b958e448e`；由上述固定 nanobind wheel 的 SHA-256 覆盖 | 允许；保留 [`licenses/robin-map/LICENSE`](licenses/robin-map/LICENSE) |

### LPIPS 评测权重（仅评估，不进入模型输出）

- torchmetrics 内嵌的 LPIPS 线性头权重：来源 <https://github.com/richzhang/PerceptualSimilarity>，BSD-2-Clause，快照 `licenses/lpips/LICENSE`。
- AlexNet 特征主干 `alexnet-owt-7be5be79.pth`：来源 torchvision 官方 `AlexNet_Weights.IMAGENET1K_V1`，URL <https://download.pytorch.org/models/alexnet-owt-7be5be79.pth>，SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`。由 `install_lpips_weights.cmd` 离线校验后装入 torch hub 缓存，训练器不隐式联网下载。
- 用途边界：只用于训练后留出视角的 LPIPS 验收指标，不参与训练损失，也不写入任何 Gaussian PLY/SPZ 产物。
- 待复核：该权重在 ImageNet 上训练，torchvision 声明预训练权重可能受训练数据条款约束。若最终产品在客户环境运行 LPIPS 验收（而非仅在研发侧），发布前需法务确认该权重的商用分发条件；否则把 LPIPS 保留为研发侧验收项。

### nerfstudio-project/nerfstudio

- 来源：<https://github.com/nerfstudio-project/nerfstudio>
- 用途：Splatfacto 数据管线和效果对照候选。
- 许可证：Apache License 2.0。
- 当前状态：训练控制层已按公共 CLI 接入，独立运行时尚未安装完成。该 PyPI 包声明的强制依赖包含 GPL-3.0 的 PyMeshLab，因此当前只作为内部开发/效果验证训练工具；未经完整传递依赖审计与法务确认，不随闭源客户端、设备镜像或客户安装包分发。
- 产品目标：把正式训练下沉为本项目直接调用 Apache-2.0 `gsplat` 公共 API 的独立适配器，按最小依赖锁定和发布；训练模型输出另行核对数据权利与合同条件。

### cnr-isti-vclab/PyMeshLab（Nerfstudio 传递依赖）

- 来源：<https://github.com/cnr-isti-vclab/PyMeshLab>
- 许可证：GPL-3.0。
- 当前状态：项目业务代码没有直接导入；Nerfstudio 1.1.5 的 PyPI 元数据将它列为 Windows 强制依赖。内部使用与对外分发的条件不同，当前发布门禁禁止把整个 Nerfstudio 环境打入闭源产品。

## 只允许研究参考，不进入商用代码

### graphdeco-inria/gaussian-splatting

- 来源：<https://github.com/graphdeco-inria/gaussian-splatting>
- 用途：阅读论文、理解公开算法、核对输入输出和比较结果。
- 许可证边界：官方免费许可限非商业研究与评估；商业/专业用途需事先取得 INRIA 明示同意。
- 项目规则：不复制其源码片段，不链接、不构建、不随安装包或容器分发。若未来取得商业授权，必须在此记录授权范围和版本后才能调整规则。

## 后续引入与发布复核

- Niantic SPZ 已完成本机引入验证：`setup_spz_environment.cmd` 从固定官方 MIT tag v3.0.0（commit `5bf2945de1a003cee07133b1e495fe9c6ffdc7e7`）的本机校验归档构建，实际 Python distribution 为 `spz==1.1.0`。Windows 安装脚本固定 `/utf-8` 编译选项、zlib 1.3.1、zstd 1.5.6 和 `nanobind==3.0.1`；nanobind/`tsl::robin-map`、Zstandard 旧解码器与其 libdivsufsort-lite 构建组成，以及 `load-spz.cc` 的 Adobe MIT 版权均已单独登记。脚本使用 `--no-deps` 安装最终 wheel，避免改动锁定的 NumPy 2.2.6；不再复制 torchvision 的 zlib 1.2.13，而是只部署并核验固定源码构建的 zlib 1.3.1。2026-09-05 最终构建日志为 `outputs/spz_setup_20260905_0345.log`：wheel SHA-256 `9cd40ecdeceadc8fa68d23c280196bbb5b56fac443b71bbd6bd0bb913f4951e5`，安装扩展 SHA-256 `0b9dfdfbc8d53f882d861b12e7ad014efefd8d932ae62a6c3eab8cad41e7a59e`，运行时 zlib DLL SHA-256 `2f08f389724c8a05c672eeb3dec1ac1ff7f589d133d4011d807e489bee522baa`。随后既有训练 PLY 再次转换，官方 `load_spz` 回读 12,781 个 Gaussian，与 PLY 一致，且 `training_report.json` / `model_manifest.json` 均有非空 `model_spz` 与 `spz_error=null`（证据见 `outputs/direct_gsplat_20260904_140102/model/`、`training_report.json` 和 `outputs/spz_export_verify_20260905_0350.log`）。当前精确安装产物已锁定，但 PEP 517 的 `scikit-build-core>=0.10` / `python_devtools` 构建工具尚未做完整离线 wheel 锁定，脚本也尚未在每次构建前从归档重建干净源码树；因此不能把它称为发布级可复现构建，仍需在正式发布前补齐。后续仍需按发布目标完成法律/再分发复核。
- Web 3DGS renderer：选型时优先 Apache-2.0、MIT 或 BSD；同时核对训练 PLY、压缩格式和 WASM 依赖的许可证。
