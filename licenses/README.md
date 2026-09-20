# 许可证快照

本目录保存截至 2026-09-03 审计时使用的官方许可证和模型卡快照，供开发与未来安装包组装。它不是最终产品的自动合规证明。

- `orbbec/`：Orbbec SDK v2 总许可证和 Extension EULA，包含三方 MIT/BSD/LGPL/libyuv 条款。
- `pyorbbecsdk2/`：Python wrapper 的 Apache-2.0 许可证。
- `mediapipe/`：MediaPipe Apache-2.0 许可证和 BlazePose GHUM 3D 模型卡。
- `open3d/`、`trimesh/`、`numpy/`、`opencv-contrib-python/`、`pillow/`：采集/几何环境运行依赖许可证；OpenCV 与 Pillow wheel 的第三方文本必须完整分发。
- `gsplat/`：直接 3DGS 产品后端 `gsplat==1.5.3` 的 Apache-2.0 许可证；已安装在 `.venv-gsplat`。
- `torch/`（LICENSE + NOTICE）、`torchvision/`、`torchmetrics/`、`lightning-utilities/`、`imageio/`、`tqdm/`：`.venv-gsplat` 产品训练运行时依赖，直接取自已安装 wheel 的 dist-info。
- `lpips/`：richzhang/PerceptualSimilarity 的 BSD-2-Clause 许可证，覆盖 torchmetrics 内嵌的 LPIPS 代码与线性头权重。AlexNet 主干权重的来源、哈希与复核事项见 `THIRD_PARTY_NOTICES.md`。
- `pymeshlab/`：GPL-3.0 快照，仅用于记录内部 Nerfstudio 对照环境的传递依赖；不进入产品运行时。

最终发布包应由锁定环境自动生成 SBOM 和 notices，并由法务根据实际交付方式复核 LGPL、模型、专利、商标和硬件 EULA 条件。
