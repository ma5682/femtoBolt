# UI 工作区隔离

当前 UI 工作区为 `E:\测试3D人体关节坐标-ui`，工作区 ID 为 `88d813bb44d9a282`。ID 从规范化后的仓库绝对路径计算，与切换分支无关。

| 内容 | 当前 UI 工作区位置 |
|---|---|
| 采集、Qt、VTK、Open3D、SDK 环境 | `E:\测试3D人体关节坐标-ui\.venv` |
| 产品 3DGS、Torch、SPZ 环境 | `C:\FemtoRuntimes\88d813bb44d9a282\gsplat` |
| 语言设置、原生 I/O、模型、Torch/HF 缓存 | `%LOCALAPPDATA%\FemtoWorkspaces\88d813bb44d9a282` |
| SPZ 构建输入缓存 | `C:\FemtoRuntimes\88d813bb44d9a282\spz-deps` |
| 原始采集、重建结果、日志与模型 | 本工作区的 `datasets/outputs/Log/models` |

两个环境均是独立文件副本，没有指向主目录环境的目录联接或硬链接。复制时新建虚拟环境、保留已验证的依赖版本及现有 gsplat/SPZ 修正，再重新生成 console/gui 启动器；没有直接沿用主目录的 Python 启动器。每个环境中的 `isolation_snapshot.json` 记录复制来源、包版本、文件数量及源文件未变化校验。

## 日常开发

- 双击本目录 `start_femto_app.cmd`。采集启动器和原型启动器只使用本工作区 `.venv`，环境缺失会提示安装，不借用主目录环境。
- 安装采集依赖使用 `.venv\Scripts\python.exe -m pip ...`，不要用不确定归属的裸 `pip`。
- 3DGS 启动脚本通过 `_resolve_gsplat_runtime.cmd` 查询专属路径，同时配置独立 ML 缓存；直接运行产品训练/查看 CLI 也会配置本工作区缓存。
- 重建环境分别使用 `setup_environment.cmd` 和 `setup_direct_gsplat_environment.cmd`。SPZ 重建使用专属 `spz-deps` 目录中的源码，固定归档和哈希不变；现有可用 SPZ 二进制已随专属环境复制。完整源树按 `setup_spz_environment.cmd` 的提示准备，不使用主目录的构建输出。
- LPIPS 权重已校验并独立复制。首次调用需要 JIT 的 CUDA 运算可能重新编译到独立缓存，系统 CUDA、编译器和显卡驱动仍共用。
- 移动工作区会改变 ID，需要按新路径重新配置专属环境；不要修改 ID 来借用其他分支环境。

检查当前解释器及 3DGS 路径：

```powershell
.venv\Scripts\python.exe -c "import sys; from femto.gsplat.scene_gsplat_trainer import runtime_python; print(sys.prefix); print(runtime_python())"
```

## Git 与历史数据边界

保留现有 Git worktree，不改分支、不提交或覆盖未提交修改。源码文件、工作区索引及运行环境各自独立；Git 对象库、远端配置等仓库元数据仍共用。这不是把 UI 转换成另一个完全独立的 Git 仓库，主目录仍持有共享 Git 元数据，不能把它当作可随意删除的环境目录。

没有复制或删除采集证据和历史方案，没有批量重写存档中的绝对来源与模型血缘。显式导入的外部工程仍保留真实来源；当前采集 marker 已确认指向 UI 自己的输出目录。

## 本次验证

独立采集环境全量测试 915 项完成（8 项预期跳过）；隔离/backend/product 专项 23 项通过；独立产品环境 guard 6 项通过。169 模块导入、分层检查和受限依赖扫描通过。采集包、Qt/VTK、Torch/gsplat/SPZ 的加载路径均验证属于各自专属环境；CUDA 可用，LPIPS 从独立缓存离线加载。没有执行完整 3DGS 训练或重建 SPZ 编译链。


## 一键环境检查与启动加固

双击根目录 `检查工作区环境.cmd` 可检查两个运行环境。命令行传 `--json` 获取结构化结果，传 `--capture-only` 仅检查采集环境。此操作不启动相机、工作台或训练任务，不安装依赖，也不改写报告/数据文件。输出包含实际解释器前缀、源码路径、核心依赖版本与加载位置；缺失、加载失败、超时、外部 site-packages 或环境联接会明确报告失败，并返回非零退出码。版本列用于核对，检查通过只表示归属及核心模块导入通过。

启动器和 3DGS 子进程清除继承的 `PYTHONHOME`、`PYTHONPATH`，禁用用户级 site-packages；采集及产品环境解析器同时验证真实解释器前缀，拒绝借用其他路径的联接环境。安装采集依赖前也验证环境归属。批处理统一使用 Windows CRLF 换行，避免中文行被错误解析。


状态查询也使用与正式子进程相同的隔离环境，并核对解释器及 Torch/gsplat 的实际加载位置；LPIPS 查询固定检查当前工作区权重，不受调用方 `TORCH_HOME` 干扰。一键诊断优先用系统 Python 3.12 运行纯标准库检查器，因此本地 `.venv` 损坏时仍有机会给出诊断；业务启动器仍只使用专属虚拟环境。异常结构的探测输出会报告失败，导入结束后再次采集搜索路径，检查依赖导入是否混入外部 site-packages。
