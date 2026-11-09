# GEMINI.md — Femto Bolt 体表采集与三维规划工作台

本文件为 Gemini CLI 提供本仓库的工程上下文、架构规范、构建/测试命令及核心开发约束。

---

## 1. 项目概述 (Project Overview)

### 项目定位
本仓库是面向未来聚焦超声治疗设备的**真实场景 / 患者体表三维采集、重建、显示、测量与空间数据基础**桌面工作台（研发预览阶段）。
核心数据流：
`Orbbec Femto Bolt → 完整场景原始 RGB-D → 标定 / Pose / 坐标 (W0/RDF) → Point Cloud / TSDF / 单帧邻接建面 → 标注 / 测量 / 区域规划 (P0–P5) → Direct gsplat 3DGS (可选视觉表现层) → 查看器 / 历史回放 / 设备软件模拟`

- **当前定位边界**：本仓库**不是**单纯的 Gaussian Splatting Demo，3DGS 仅作为表现层之一，所有区域标记、测量和路线规划均建立在 PointCloud / Mesh / Geometry 几何层之上。
- **安全与执行边界**：当前处于研发预览阶段，**不实现真实机械臂控制、ROS、MoveIt、轨迹执行或超声能量输出**。所有相关导出 JSON 固定标记 `execution_ready=false`、`physical_accuracy_verified=false`。
- **唯一技术基线 (Source of Truth)**：根目录下的 [`技术路线.md`](技术路线.md) 是当前实现的唯一权威技术基线。若 `README.md`、其他文档或注释与其冲突，一律以 `技术路线.md` 顶部当前状态和稳定约束为准。

### 核心技术栈
- **硬件与平台**：Orbbec Femto Bolt RGB-D 相机 (`pyorbbecsdk2`)、Windows、NVIDIA CUDA GPU、Python
- **桌面 GUI 与可视化**：PySide6 (Qt 6.11.2) + VTK (9.5.2)
- **三维几何与视觉**：Open3D (0.19.0)、Trimesh (5.1.0)、OpenCV Contrib、NumPy、Pillow、MediaPipe
- **3DGS 渲染与训练**：Apache-2.0 `gsplat` + PyTorch + SPZ（运行在隔离的工作区专属 ASCII 环境中）
- **数据存储**：本地文件系统（不可变 `datasets/` + 派生 `derived/` & `outputs/`）与本地 SQLite（`history.sqlite3` 历史索引、`execution.sqlite3` 模拟联调事件库）

### 系统架构与 17 层单向依赖
源码位于 `femto/` 包下，由 `tools/check_layering.py` 强制执行严格的 **17 层由低到高单向依赖**：

```text
femto -> depth -> dataset -> geometry -> hardware -> reports -> annotation
      -> tracking -> temporal -> local -> planning -> viewer -> pipeline
      -> gsplat -> capture -> execution -> app
```

- **单向导入铁律**：只允许上层包导入下层包（`femto.paths` 例外），**函数内延迟 import 同样受检**。任何代码修改后运行 `tools/check_layering.py` 必须保持 `upward edges: 0`。
- **主要模块职责**：
  - `femto.app`：唯一 GUI 入口（`qt_main.WorkstationWindow`），包含采集 (`CapturePage`)、规划 (`PlanningPage`)、只读回看 (`ReviewPage`)、设备与模拟 (`ExecutionPage`) 四页生命周期。
  - `femto.execution`：软件设备模拟任务协调、前置检查、状态机、资源占用与 `execution.sqlite3` 事件库（不导入 Qt，不驱动真实硬件）。
  - `femto.capture` / `femto.tracking` / `femto.local` / `femto.pipeline`：相机线程 `CameraWorker`、实时 RGB-D 里程计跟踪 (`LiveTracker`)、阶段快照 (`CaptureCheckpoints`)、局部重建与受限子进程离线作业 (`LocalOfflineJob`)。
  - `femto.annotation` / `femto.planning`：固定参考图投影 (`FixedRGBModel`)、涂抹选区意图、P0/P1 原目标台账 (`TargetLedger`)、表面蛇形路线生成 (`SurfaceTrajectory`)、P2 名义覆盖核验 (`Certificate`)、P3 固定模型补充规划、P4 严格路径、P5 名义变宽模型及便携工程 (`.region.zip`) 保存回读。
  - `femto.gsplat`：隔离在专用运行时的多视角与单帧 Gaussian Splatting 训练、评估与查看器。
  - `prototype/`：独立的 PySide6 Mock 交互原型，与 `femto/` 互不导入，不作为产品功能已实现的依据。

---

## 2. 构建、运行与验证命令 (Building, Running, and Testing)

### 环境隔离约定
项目严格隔离三套环境，**严禁跨环境混装或合并依赖**：
1. **`.venv`**：工作区独立的采集 / 几何 / PySide6 UI / 完整单元测试环境（注：`pyorbbecsdk2 --no-deps` 会使 `pip check` 提示缺 `opencv-python`，属预期行为，切勿擅自修复）。
2. **`C:\FemtoRuntimes\<workspace_id>\gsplat`**（如 `C:\FemtoRuntimes\88d813bb44d9a282\gsplat`）：由 `femto.paths.ASCII_GSPLAT_RUNTIME_ROOT` 解析的工作区专属 ASCII 路径环境，用于 `gsplat` / `torch` / `SPZ`。
3. **`.venv-3dgs`**：已废弃且禁止产品分发的旧对照环境。

环境初始化与检查：
```powershell
# 首次安装采集与 UI 环境
.\setup_environment.cmd

# 首次安装 direct gsplat 训练/渲染环境
.\setup_direct_gsplat_environment.cmd

# 只读检查当前工作区环境归属与依赖状态
.\检查工作区环境.cmd
```

### 启动应用与常用入口
```powershell
# 启动正式人体局部采集、规划和历史回看工作台（含相机）
.\start_femto_app.cmd

# 无相机 UI 预览模式启动工作台
.venv\Scripts\python.exe -m femto.app.qt_main --ui-preview

# 直接打开已有模型目录或便携方案 (.zip / .json) 进入只读回看
.venv\Scripts\python.exe -m femto.app.qt_main --ui-preview --project "实际的模型目录或方案.zip"

# 通用物品/场景实拍与重建 CLI
.\start_real_scene_demo.cmd

# Direct gsplat 状态检查与训练
.venv\Scripts\python.exe -m femto.gsplat.gsplat_backend status --input "实际完整重建结果目录"
.venv\Scripts\python.exe -m femto.gsplat.gsplat_backend train --input "实际完整重建结果目录" --require-real --profile low_vram
```

### 必跑六项验证命令 (Pre-Commit Verification Suite)
每次提交或完成代码修改后，**必须从仓库根目录依次执行以下全部六项检查**，不得跳过、改写或放宽：

```powershell
# 1. 17 层架构单向依赖门禁（必须保持 upward edges: 0）
.venv\Scripts\python.exe tools\check_layering.py

# 2. 全模块导入完整性检查（验证 .venv 下无顶层 torch 等非法导入）
.venv\Scripts\python.exe -X utf8 -c "import importlib,pathlib;ms=[p.with_suffix('').as_posix().replace('/','.') for p in pathlib.Path('femto').rglob('*.py')];[importlib.import_module(m) for m in ms];print(len(ms),'modules OK')"

# 3. gsplat 专属运行时核心模块导入检查
C:\FemtoRuntimes\88d813bb44d9a282\gsplat\Scripts\python.exe -X utf8 -c "import femto.gsplat.scene_gsplat_trainer,femto.gsplat.scene_gsplat_viewer,femto.gsplat.gsplat_backend,femto.gsplat.scene_single_gs,femto.gsplat.scene_single_gs_viewer,femto.gsplat.scene_gsplat_acceptance;print('gsplat runtime OK')"

# 4. 全量单元测试套件（在 .venv 中固定存在 8 个因缺 torch 产生的结构性 skip）
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests

# 5. gsplat 运行时专项守卫测试
C:\FemtoRuntimes\88d813bb44d9a282\gsplat\Scripts\python.exe -m unittest discover -s tests -p "test_gsplat_guard.py"

# 6. 商业合规禁用依赖 AST 干扫描（必须输出 clean）
.venv\Scripts\python.exe -X utf8 -c "from pathlib import Path;from commercial_compliance_check import imported_modules,BLOCKED_IMPORT_PREFIXES;bad=[(f,[m for m in imported_modules(f) if m.startswith(BLOCKED_IMPORT_PREFIXES)]) for f in list(Path('femto').rglob('*.py'))+list(Path('tools').glob('*.py'))];print([b for b in bad if b[1]] or 'clean')"
```

### 定向测试运行方式
- **运行单个测试文件**（必须从仓库根目录使用 `-s tests -p`，不要进入 `tests/` 目录或使用点分模块名）：
  ```powershell
  .venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_<name>.py"
  ```
- **运行 Prototype 原型测试**（两套测试因 `QCoreApplication` 与 `QApplication` 冲突必须分两个独立进程执行）：
  ```powershell
  .venv\Scripts\python.exe prototype/tests/test_workflow.py
  .venv\Scripts\python.exe prototype/tests/test_visual_contract.py
  ```
- **注意**：不要随手运行 `commercial_compliance_check.py` 主程序，它会覆盖 `compliance/current_report.json`；日常检查仅运行上方第 6 条干扫描命令。

---

## 3. 开发约定与硬性规则 (Development Conventions & Mandates)

### 文档与基线同步规则
1. **技术基线同步**：修改采集、对齐、Mask、关键点、坐标、筛帧、融合、TSDF、Mesh、规划算法、数据格式或依赖时，**必须同步更新根目录 [`技术路线.md`](技术路线.md) 的参数与状态，并将完整变更记录追加到 [`doc/变更记录/`](doc/变更记录/) 对应月份文件**。
2. **文档目录职责**：
   - `doc/使用说明/`：面向操作者的当前功能用法。
   - `doc/技术资料/`：架构约定、marker 约定、P0–P5 各阶段契约与算法证明规范。
   - `doc/历史开发记录/`：按日期归档的每轮交付记录（仅作历史参考，不覆盖当前基线）。
   - `doc/变更记录/`：按月追加的完整技术路线变更流水。

### 数据、坐标与物理安全铁律
1. **原始采集证据不可变**：`datasets/scene_*/{rgb,depth,pose,calibration,frames.jsonl,manifest.json}` 是不可变原始证据层。任何估计位姿、融合结果、点云、Mesh、选区、路线、覆盖证明只能写入场景下的 `derived/` 或全局 `outputs/`，严禁覆盖或裁剪原始 RGB-D。
2. **坐标系与单位显式化**：
   - 磁盘相机坐标统一采用 **RDF**（`+X` 右、`+Y` 下、`+Z` 前）。
   - 深度图 PNG 为 `uint16` **毫米 (mm)**（0 表示未知）；点云、Mesh、关节、标注、表面轨迹一律使用 **米 (m)**（UI 界面展示为 mm 仅作格式化，不代表物理精度上界）。
   - **W0 参考系**：局部采集会话首个有效彩色相机位姿定义为 `W0`（首帧 Pose 精确为单位矩阵 `I`）。`T_AB` 将 B 系坐标变换到 A 系（`p_W = T_WC · p_C`）。所有三维数据必须明确标注坐标系、单位、变换方向和来源，禁止裸 XYZ。
   - **移动相机禁止单位外参**：`T_WC = I` 仅对物理固定机位成立。移动采集中位姿估计不合格时必须**阻断并写质量报告**，严禁输出伪 Pose、单位矩阵或沿用旧位姿。
3. **规划与覆盖判定语义**：
   - **“未确认 (unconfirmed/pending)” ≠ “已覆盖”，也 ≠ “漏扫”**：缺面、无深度、深度冲突、未定位范围一律保持待确认状态，**严禁用补洞或跨块连线伪装完成**。
   - **算法版本字符串与预算契约**：路线与覆盖算法版本字符串（如 `quality-constrained-surface-v3`、`interleaved-partition-anchor-ball-coverage-v5` 等）是持久化回放契约。旧版本代码用于按版本回放历史证明，**不得因有新版本而删除**。
   - **严禁跨请求缓存接受结论**：只允许在单次操作内复用按当前可用内存配置的只读几何缓冲，不得跨请求复用覆盖资格或放宽时间/单元预算（如工作台核心 30s、操作 120s、目标建账 45s 等契约上限）。

### 编码、导入与平台规范
1. **延迟导入与依赖隔离**：
   - `femto.gsplat.*`、viewer 和 `scene_single_gs` **只能在函数内部延迟 `import torch`**（因为 `.venv` 不安装 `torch`）。
   - `femto.io_utils` 禁止在模块顶层导入 `cv2`。
   - 仅 `femto/app/qt_*`（及 `scene_gsplat_viewer.py`）在顶层导入 `PySide6`；新增 Qt 单元测试必须在导入 `PySide6` 前设置 `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`，并在测试结束时调用页面的 `shutdown()` 与 `close()`。
2. **合规与第三方许可**：
   - **严禁引入 Graphdeco `gaussian-splatting` 源码**或导入 `diff_gaussian_rasterization`、`simple_knn`、`pymeshlab`。正式 3DGS 仅使用 Apache-2.0 `gsplat` 公共 API。
   - 严禁运行时隐式联网下载权重或模型（如 LPIPS 权重必须离线锁定 SHA-256）。
   - 新增任何第三方依赖必须先登记 `THIRD_PARTY_NOTICES.md` 与 `licenses/`。
3. **Windows 与中文路径安全**：
   - 图像读写统一使用 `femto.io_utils.imread_unicode` / `imwrite_unicode` 或 PIL。
   - Open3D 原生文件读写必须经过 ASCII 临时目录安全包装。
   - JSON 读取统一使用 `utf-8-sig`，并严格保持既有写出格式与 SHA 校验语义。
4. **类型标注与模块导入**：
   - 公开接口使用 `from __future__ import annotations` 及现代类型标注（`Path`、`np.ndarray`、`X | None`）。
   - 产品代码一律使用完整绝对包路径导入（`from femto.<pkg>.<module> import ...`），单元测试中 `unittest.mock.patch` 的目标字符串也必须使用完整包路径。

### 多 Worktree 与并行协作注意事项
- 本仓库存在多个同级 Git Worktree（主检出 `E:\测试3D人体关节坐标` 及本 UI 工作树 `E:\测试3D人体关节坐标-ui` 等），且可能有其他 AI Agent 会话并行编辑。
- 开始修改或提交前，务必先检查 `git status` 和文件修改时间，确认当前工作树路径，不要跨 worktree 引用被忽略的 `datasets/` 或 `outputs/`，也不要批量提交未核实归属的改动或私有凭据（`git/` 目录）。
