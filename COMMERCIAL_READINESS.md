# 商用发布门禁

审计更新：2026-09-03。本文是工程合规清单，不替代律师、医疗器械法规或数据保护专业意见。

## 当前结论

**可以用于商用产品的内部研发和原型验证；当前版本尚不能直接打包销售或交付客户。**

主要开源组件允许商用，Graphdeco 受限代码已经排除。发布仍受 Orbbec Extension EULA、SDK/OpenCV 中 LGPL 组件的再分发义务、项目权利人登记、真人数据隐私，以及产品用途可能触发的医疗器械监管约束。

## 组件判断

| 范围 | 判断 | 发布条件 |
|---|---|---|
| 本项目独立代码 | 可做闭源商用 | 确认公司/个人权利人、贡献者权利转让和产品 EULA |
| Orbbec Python wrapper | 可商用 | 保留 Apache-2.0 许可证和修改说明 |
| Orbbec Extension Library | 有条件可商用 | 必须与真实 Orbbec 硬件结合；不可修改/反编译；随产品附 Orbbec EULA；不得单独收费或单独分发 |
| Orbbec SDK 三方库 | 有条件可商用 | 保留 MIT/BSD/LGPL/libyuv 等文本；对 LGPL 动态库提供相应源码/书面要约和可替换、可重链接方案 |
| MediaPipe runtime + Pose 模型 | 可商用 | 随分发物附 Apache-2.0 和模型卡；核实 Tasks 遥测行为并完成用户告知/同意策略 |
| Open3D / Trimesh | 可商用 | 保留 MIT 版权与许可证；新增 Trimesh 外部后端时重新审计 |
| NumPy / OpenCV wheels | 有条件可商用 | 分发 wheel 自带的完整 LICENSE/第三方 notices；处理 FFmpeg 等 LGPL 条件 |
| Pillow | 可商用 | MIT-CMU；保留版权/许可文本并分发 wheel 内完整三方许可证目录 |
| `gsplat` | 可作为商用 3DGS 基线 | Apache-2.0；`gsplat==1.5.3` 已安装在独立产品运行时 `.venv-gsplat` 并完成 GPU 验收；发布时保留 LICENSE 与 `patch_gsplat_windows.py` 编译标志修改说明；不安装 Graphdeco/PyMeshLab/Nerfstudio（门禁自动检查） |
| PyTorch / torchvision / torchmetrics 运行时 | 有条件可商用 | BSD-3 / BSD-3 / Apache-2.0（含 BSD-2 LPIPS 代码）；分发 `licenses/torch/` 的 LICENSE 与 NOTICE，核对 cu128 wheel 捆绑的 NVIDIA 运行库再分发条款；版本由门禁钉死 |
| LPIPS AlexNet 评测权重 | 研发侧验收可用；客户侧分发待复核 | torchvision 官方 ImageNet 权重，仅用于训练后 LPIPS 指标，不进入模型产物；离线按 SHA-256 装载。若产品在客户环境执行 LPIPS 验收，发布前需法务确认 ImageNet 衍生权重的商用条件 |
| Nerfstudio / Splatfacto | 仅限当前内部训练验证 | 主项目 Apache-2.0，但 PyPI 强制传递依赖包含 GPL-3.0 PyMeshLab；当前禁止把完整环境打入闭源客户端/设备，正式产品改为直接 `gsplat` 最小适配并重新审计 |
| Graphdeco Gaussian Splatting | 不进入商用产品 | 只有取得 INRIA 书面商业授权后才能改变此结论 |

## 发布前必须完成

- [ ] 登记项目著作权人和商业主体；补齐员工/承包开发成果归属。
- [ ] 固定实际发布依赖并生成 SBOM、哈希和漏洞报告；不能直接把当前开发环境全部打包。
- [ ] 安装包内放入 `licenses/`、`THIRD_PARTY_NOTICES.md` 和各组件要求的 NOTICE。
- [ ] 把 Orbbec Extension EULA 纳入安装许可或客户合同，并限定与真实 Orbbec 硬件配套使用。
- [ ] 为 Orbbec/OpenCV 中的 LGPL 动态库制定源码提供、替换和重链接方案，经法务确认。
- [ ] 对 MediaPipe Tasks 做离线网络抓包；若发送使用/性能指标，完成明确告知、合法基础和网络策略。
- [ ] 对真人 RGB、脸部、人体表面、骨架和健康场景数据完成个人信息保护影响评估、单独同意、最小化、加密、权限、删除和留存机制。
- [ ] 明确产品宣传用途。若用于诊断、治疗规划、患者定位或机械臂控制，先完成医疗器械分类、风险管理、软件生命周期、临床评价和注册路径；当前研发标签不得直接驱动治疗。
- [ ] 完成商标、专利自由实施和最终法律审查。

## 可先发布的低风险范围

在完成许可证打包、Orbbec EULA和隐私制度后，可优先把产品定义为“真实场景 RGB-D 采集、三维重建和可视化工具”，不宣称诊断、疗效、治疗靶点或自动控制功能。人体 Pose/区域只作为研发可视化标注，并明确精度限制。

首个商用构建建议把 MediaPipe/Pose 设为可选模块。核心全场景采集与重建不依赖人体模型；这样可以先交付 RGB-D、Point Cloud、Mesh 和 3DGS 功能，再单独完成语义模块的模型、遥测、隐私和用途验证。MediaPipe 模型卡明确不适合提供米制精确深度，因此本项目的尺度只能来自 Femto Depth 和经过验证的相机位姿，不能来自模型的 `world_landmarks`。

## 许可证材料

当前审计快照位于 `licenses/`。版本升级、改变安装包内容、加入新模型或 Web renderer 后必须重新审计，不能沿用旧结论。
