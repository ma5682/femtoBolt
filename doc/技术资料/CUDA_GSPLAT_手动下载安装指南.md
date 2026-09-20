# Femto / gsplat Windows 手动下载安装指南

本文用于为 `E:\测试3D人体关节坐标` 安装可编译 `gsplat` CUDA 扩展的独立训练环境。目标机器为 Windows 11 x64、NVIDIA GeForce RTX 5060 Laptop GPU（8 GB），项目固定使用 CUDA Toolkit 12.8 与 Python 3.10。

> NVIDIA 驱动中显示的“CUDA Version”只是驱动兼容上限，不代表系统已经安装 CUDA Toolkit 或 `nvcc.exe`。

## 1. 当前电脑状态

| 项目 | 当前状态 | 说明 |
|---|---|---|
| NVIDIA GPU / 驱动 | 已具备 | RTX 5060 Laptop GPU，驱动 591.91 |
| Python 3.10 x64 | 已安装 | `C:\Users\55053\AppData\Local\Programs\Python\Python310\python.exe` |
| Python 3.12 / Femto 环境 | 保留 | 不要删除，不影响相机采集 |
| Visual Studio 2022 C++ Build Tools | 待手动安装 | 需要提供 MSVC `cl.exe` |
| CUDA Toolkit 12.8 | 待手动安装 | 需要提供 `nvcc.exe`、CUDA 头文件和开发库 |
| PyTorch / gsplat 独立虚拟环境 | 前置项完成后再安装 | 不装入现有 Femto `.venv` |

已下载并验证数字签名的安装文件：

```text
C:\Users\55053\Downloads\femto-build-prereqs\python-3.10.11-amd64.exe
C:\Users\55053\Downloads\femto-build-prereqs\vs_BuildTools.exe
```

Python 3.10 已经安装；通常只需手动运行现有的 `vs_BuildTools.exe`，再下载 CUDA 12.8。

## 2. 官方下载地址

请只使用下表中的官方地址，不要从软件站、网盘或驱动工具下载。

| 软件 | 官方地址 | 预计下载量 |
|---|---|---:|
| Python 3.10.11 x64 | [python-3.10.11-amd64.exe](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) | 约 28 MB |
| Visual Studio 2022 Build Tools | [vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe) | 引导器约 4 MB，组件数 GB |
| CUDA 12.8 Windows 11 下载页 | [NVIDIA CUDA 12.8 Archive](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network) | 可选网络或离线安装器 |
| CUDA 12.8 网络安装器 | [cuda_12.8.0_windows_network.exe](https://developer.download.nvidia.com/compute/cuda/12.8.0/network_installers/cuda_12.8.0_windows_network.exe) | 引导器约 13.6 MB，按所选组件下载 |
| CUDA 12.8 完整离线安装器 | [cuda_12.8.0_571.96_windows.exe](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe) | 约 3.2 GB |

网络较慢时优先下载完整离线 CUDA 安装器，它支持浏览器或下载工具断点续传。网络安装器本身较小，但安装过程中断后可能需要重新拉取组件。两种 CUDA 安装器二选一即可。

不要安装 winget 当前提供的 CUDA 13.x。项目的 PyTorch CUDA wheel、`gsplat` 编译参数和测试基线固定为 12.8。

## 3. 安装顺序

必须按以下顺序安装：

```text
Python 3.10
    ↓
Visual Studio 2022 Build Tools（C++）
    ↓
CUDA Toolkit 12.8
    ↓
重启 Windows
    ↓
创建独立 gsplat 虚拟环境
```

先安装 Build Tools，再安装 CUDA，可以让 CUDA 安装器正确发现 Visual Studio 2022。

## 4. 安装 Visual Studio 2022 Build Tools

1. 运行：

   ```text
   C:\Users\55053\Downloads\femto-build-prereqs\vs_BuildTools.exe
   ```

2. Windows 弹出管理员授权时选择“是”。
3. 在“工作负荷”中勾选：

   ```text
   使用 C++ 的桌面开发
   Desktop development with C++
   ```

4. 在右侧安装详细信息中确认至少包含：

   - MSVC v143 - VS 2022 C++ x64/x86 生成工具（最新）；
   - Windows 11 SDK（安装一个最新的 x64 SDK 即可）；
   - 用于 Windows 的 C++ CMake 工具；
   - Visual Studio 的 C++ 核心功能。

5. 不需要完整 Visual Studio IDE，也不需要 .NET、移动开发、游戏开发或 ARM 工具。
6. 建议使用默认安装目录，便于项目脚本通过 `vswhere.exe` 自动发现。
7. 等待安装完成，不要在组件仍在下载时启动 CUDA 安装器。

## 5. 安装 CUDA Toolkit 12.8

1. 运行网络安装器或完整离线安装器，两者只运行一个。
2. Windows 弹出管理员授权时选择“是”。
3. 选择“自定义安装（高级）”。
4. 保留以下 CUDA 12.8 组件：

   - CUDA Development（包含 `nvcc`、头文件和开发库）；
   - CUDA Runtime；
   - CUDA Visual Studio Integration（建议保留，Build Tools 已先安装）；

5. 可以取消以下非训练必需组件以节省空间：

   - Display Driver：当前 591.91 驱动比 CUDA 12.8 安装包附带的 571.96 更新，不要降级；
   - GeForce Experience / NVIDIA App；
   - Nsight Systems、Nsight Compute、Nsight Visual Studio Edition；
   - CUDA Samples、Documentation、Demo Suite。

6. 默认安装目录应为：

   ```text
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
   ```

7. 安装完成后重启 Windows。

## 6. 校验下载文件

在 PowerShell 中执行：

```powershell
Get-AuthenticodeSignature "C:\Users\55053\Downloads\femto-build-prereqs\vs_BuildTools.exe"
Get-AuthenticodeSignature "C:\Users\55053\Downloads\cuda_12.8.0_windows_network.exe"
```

`Status` 应为 `Valid`。CUDA 12.8 官方页面公布的 MD5 为：

```text
网络安装器：1d7e1cf4047f2b8d9a8096e18ebea1c7
完整离线包：5e7fe0f087d65fae4c0278f13af55c1d
```

按实际文件路径校验：

```powershell
Get-FileHash "C:\Users\55053\Downloads\cuda_12.8.0_windows_network.exe" -Algorithm MD5
Get-FileHash "C:\Users\55053\Downloads\cuda_12.8.0_571.96_windows.exe" -Algorithm MD5
```

只需校验你实际下载的那一种 CUDA 安装器。

## 7. 安装后验证

重启后打开新的“命令提示符”并执行：

```bat
py -3.10 --version
nvidia-smi
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" --version
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
cl
```

预期结果：

- Python 输出 `Python 3.10.11`；
- `nvidia-smi` 能看到 RTX 5060 Laptop GPU；
- `nvcc` 输出 `release 12.8`；
- `cl` 输出 Microsoft C/C++ x64 Compiler 19.x，不应提示“不是内部或外部命令”。

再检查关键文件：

```bat
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" echo CUDA_OK
if exist "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" echo VSWHERE_OK
```

`cl.exe` 在普通终端里找不到并不一定表示未安装；先调用上面的 `VsDevCmd.bat`，再执行 `cl`。

## 8. 下一步：项目独立训练环境

验证四项命令都通过后，再回到：

```text
E:\测试3D人体关节坐标
```

后续将创建独立的 `.venv-gsplat` 产品训练环境，并直接安装 PyTorch CUDA 12.8、Apache-2.0 `gsplat` 和评测依赖。它与现有 Femto `.venv` 分开，不影响相机 SDK、Open3D、采集和 TSDF。

现有 `setup_3dgs_environment.cmd` 是 Nerfstudio 内部开发验证环境，含需要单独评估的传递依赖，不应打包进闭源商用客户端。正式产品训练器使用直接 `gsplat` 的最小环境。

## 9. 常见问题

### `nvcc` 找不到

先确认文件是否存在：

```text
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe
```

然后关闭并重新打开终端。CUDA 安装器通常会自动设置 `CUDA_PATH`；不要手动把 CUDA 13.x 路径放在 12.8 前面。

### `cl` 找不到

重新打开“x64 Native Tools Command Prompt for VS 2022”，或先调用：

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
```

### CUDA 安装器提示 Visual Studio Integration 失败

先完成 Build Tools 的“使用 C++ 的桌面开发”，重启后再运行 CUDA 12.8 自定义安装。即使 Visual Studio Integration 单项失败，只要 `nvcc`、CUDA headers/libraries 和 `cl` 都能工作，Python CUDA 扩展仍可编译。

### 是否需要卸载 Python 3.12

不需要。Femto 采集环境继续使用 Python 3.12；3DGS 训练通过 `py -3.10` 创建独立虚拟环境。

## 10. 官方依据

- [NVIDIA CUDA 12.8 Windows 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html)
- [NVIDIA CUDA 12.8 下载归档](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Windows)
- [Microsoft Visual Studio Build Tools 命令行安装说明](https://learn.microsoft.com/visualstudio/install/use-command-line-parameters-to-install-visual-studio)
- [Python 3.10.11 官方发布文件](https://www.python.org/downloads/release/python-31011/)
- [gsplat 官方安装说明](https://docs.gsplat.studio/main/installation.html)

