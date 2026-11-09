# 可选点云计算内核

这是项目独立编写的 CPython 扩展，只加速既有的八子盒负例判定和完整半径邻域判定。原始点云、法向、空间格、邻域、阈值和临界值回退规则保持。它不包含设备控制或第三方算法源码。

在本 checkout 的 Windows x64 Python 中显式构建：

```powershell
.venv\Scripts\python.exe -X utf8 tools\build_cloud_kernel.py
.venv\Scripts\python.exe -X utf8 -m unittest discover -s native -p "test_cloud_predicate.py"
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -p "test_cloud_kernel.py"
```

构建使用本机已有的 MSVC C/C++ 工具链、当前 Python 的开发头文件和 setuptools，不联网、不安装依赖，也不生成应用 EXE。只支持已验证的 Windows x64 配方；本次验证环境为 CPython 3.12.10、setuptools 84.0.0、MSVC 14.44.35207，运行时 NumPy 2.2.6 / Open3D 0.19.0。

扩展使用 CPython buffer protocol 直接读取当前调用的数组，不依赖 NumPy C API。构建固定 `/O2 /fp:strict`，不启用 fast-math 或乘加融合。`outputs/cloud_kernel_build_*/build.json` 保存源文件、构建脚本、编译器、产物 SHA-256、Python ABI 和实际编译/链接命令；构建完成并重新核对源码后才发布 `.pyd`。被其他进程占用的旧扩展不能原子替换时，命令失败并保留旧文件，不会结束其他进程。

源码和构建脚本入 Git，生成的 `_cloud_predicate*.pyd` 不入 Git。更换 Python ABI 或修改 C 源码后需重新构建；源码与扩展记录的源码 SHA-256 不一致时不启用加速。未构建、加载失败、ABI 不匹配、非默认 NumPy 错误策略、非原生连续对齐数组及可选内核分配失败时，保留原 NumPy / 完整查询路径。应用运行时不会自动编译或下载。

可在启动新进程前设置 `FEMTO_DISABLE_CLOUD_KERNEL=1` 临时停用；清除这个进程环境变量并重新启动即可恢复。该开关不修改数据或模型。

完整邻域每批仍受原 65,536 候选对预检上限约束；正例只在找到原始点的严格见证后提前结束，未确定和负例继续完整检查，半径/高度临界值走旧 KDTree 判据。八子盒查询每批最多 4,096 点，任何子盒不能证明不存在见证时，就将该点留给原见证/完整半径检查。调用间继续检查取消，不保存跨调用的坐标副本或几何结论。

开启细分计时观测时，融合路径使用 `cloud.native_fused_predicate`，NumPy 回退保留原算术子阶段名称。两条路径都记录完整查询返回的邻点数；正例提前结束不减少该计数，也不生成未实际执行的子阶段计时。观测不是几何证据或历史回放输入。

扩展构建与应用 EXE 打包是两个步骤。本次只交付源码与当前 checkout 的本机扩展；已有 EXE 不会因此自动获得该优化。完整数据对照及性能范围见 `doc/技术资料/点云本地内核提速_20261104.md`。
