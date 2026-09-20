# PyTorch CUDA 12.8 / gsplat 手动下载清单

CUDA Toolkit 12.8、MSVC 和 Python 3.10 已经验证通过。命令行下载 PyTorch 只有约 30 KB/s，因此改为浏览器手动下载两个官方 wheel。其余依赖体积较小，下载完成后由项目脚本自动处理。

## 保存目录

先创建以下目录，并把两个文件都保存到这里：

```text
C:\Users\55053\Downloads\femto-gsplat-wheels
```

不要解压 `.whl` 文件。

## 必须下载的文件

| 序号 | 文件 | 官方直链 | 精确大小 | SHA256 |
|---:|---|---|---:|---|
| 1 | `torch-2.7.1+cu128-cp310-cp310-win_amd64.whl` | [下载 torch 2.7.1 CUDA 12.8](https://download-r2.pytorch.org/whl/cu128/torch-2.7.1%2Bcu128-cp310-cp310-win_amd64.whl) | 3,273,064,440 字节（约 3.05 GiB） | `5174f02de8ca14df87c8e333c4c39cf3ce93a323c9d470d690301d110a053b3c` |
| 2 | `torchvision-0.22.1+cu128-cp310-cp310-win_amd64.whl` | [下载 torchvision 0.22.1 CUDA 12.8](https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.1%2Bcu128-cp310-cp310-win_amd64.whl) | 7,638,365 字节（约 7.29 MiB） | `ad48ba3c3ffd48027e3a8de42fcea131a53a524ee9416ca4efb22f9ac6b7328d` |

两个链接都来自 PyTorch 官方 CUDA 12.8 wheel CDN。版本与项目环境固定一致：

```text
Python       3.10 x64
PyTorch      2.7.1+cu128
torchvision  0.22.1+cu128
CUDA Toolkit 12.8
gsplat       1.5.3
```

## 下载后的目录结构

文件名必须保持如下形式；如果浏览器添加了 `(1)`、`.download` 或其他后缀，请等待下载完成并改回准确文件名：

```text
C:\Users\55053\Downloads\femto-gsplat-wheels\
├─ torch-2.7.1+cu128-cp310-cp310-win_amd64.whl
└─ torchvision-0.22.1+cu128-cp310-cp310-win_amd64.whl
```

其中：

- `cp310` 表示 Python 3.10；
- `win_amd64` 表示 64 位 Windows；
- `cu128` 表示 CUDA 12.8。

不要下载 Linux wheel、Python 3.11/3.12 wheel、CPU wheel 或 CUDA 13.x wheel。

## 可选：手工校验

下载完成后可在 PowerShell 执行：

```powershell
Get-FileHash "C:\Users\55053\Downloads\femto-gsplat-wheels\torch-2.7.1+cu128-cp310-cp310-win_amd64.whl" -Algorithm SHA256
Get-FileHash "C:\Users\55053\Downloads\femto-gsplat-wheels\torchvision-0.22.1+cu128-cp310-cp310-win_amd64.whl" -Algorithm SHA256
```

项目安装脚本也会自动计算完整 SHA256。任何一个哈希不匹配都会停止安装，不会执行来源不明或下载损坏的 wheel。

## 下载完成后的处理

下载完两个文件后告诉我“PyTorch 下载完成”。我会继续运行：

```text
E:\测试3D人体关节坐标\setup_direct_gsplat_environment.cmd
```

脚本会优先使用上述本地 wheel，然后安装 Apache-2.0 `gsplat==1.5.3`、图像和 PSNR/SSIM/LPIPS 评测依赖。它不会安装 Nerfstudio、PyMeshLab 或 Graphdeco 代码。

## 官方索引依据

- [PyTorch CUDA 12.8 torch wheel 索引](https://download.pytorch.org/whl/cu128/torch/)
- [PyTorch CUDA 12.8 torchvision wheel 索引](https://download.pytorch.org/whl/cu128/torchvision/)
- [PyTorch 历史版本安装表](https://pytorch.org/get-started/previous-versions/)
- [gsplat 1.5.3 PyPI 页面](https://pypi.org/project/gsplat/1.5.3/)

