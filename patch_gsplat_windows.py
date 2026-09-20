"""Apply the minimal gsplat 1.5.3 MSVC JIT flag compatibility patch.

gsplat 1.5.3 passes GCC-only ``-Wno-attributes`` to the host compiler and uses
one optimization flag for both MSVC and nvcc.  On Windows this becomes the
invalid MSVC option ``/Wno-attributes``.  The installed Apache-2.0 module is
patched reproducibly: MSVC receives /O2 (/Od for FAST_COMPILE), while nvcc
keeps -O3 (-O0).  No algorithm or CUDA kernel source is changed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ORIGINAL = '''        opt_level = "-O0" if FAST_COMPILE else "-O3"
        extra_cflags = [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level]
'''

PATCHED = '''        if os.name == "nt":
            host_opt_level = "/Od" if FAST_COMPILE else "/O2"
            extra_cflags = [host_opt_level]
        else:
            host_opt_level = "-O0" if FAST_COMPILE else "-O3"
            extra_cflags = [host_opt_level, "-Wno-attributes"]
        cuda_opt_level = "-O0" if FAST_COMPILE else "-O3"
        extra_cuda_cflags = [cuda_opt_level]
'''


def patch_backend(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    if PATCHED in source:
        return "already_patched"
    if ORIGINAL not in source:
        raise RuntimeError(
            "Installed gsplat backend no longer matches the reviewed 1.5.3 source; "
            "do not apply this compatibility patch to an unknown version"
        )
    path.write_text(source.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    return "patched"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.backend
    if path is None:
        import gsplat

        path = Path(gsplat.__file__).resolve().parent / "cuda" / "_backend.py"
    result = patch_backend(path.resolve())
    print(f"gsplat MSVC compatibility: {result}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
