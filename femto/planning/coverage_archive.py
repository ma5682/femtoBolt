"""Separate versioned proof assets, published after bounded readback and replay.

This format is independent of the product's P1 trajectory/project formats. A
caller must provide the authoritative domain, seeds, footprint and route binding.
Hashes alone never establish coverage or provenance.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable
import zipfile

import numpy as np

from femto.planning import coverage_certificate as coverage
from femto.planning.coverage_domain import CoverageDomain
from femto.planning.planning_io import durable, publish, staging_directory
from femto.planning.surface_trajectory import canonical_json_bytes

MANIFEST = 'coverage_certificate.json'
ASSET = 'coverage_certificate.npz'
MAX_MANIFEST_BYTES = 1024**2
HINT_PREFIX = 'recalculation_hint_'
HINT_ASSETS = {HINT_PREFIX+name for name in (MANIFEST, ASSET)}


def asset_names(metadata: dict, *, prefix: str = '') -> set[str]:
    names = {prefix+MANIFEST, prefix+ASSET}
    if metadata.get('algorithm') in coverage.RECALCULATION_ALGORITHMS:
        names |= {prefix+name for name in HINT_ASSETS}
    return names


def check_hint_inventory(names: set[str], metadata: dict | None, *, prefix: str = '') -> None:
    expected = ({prefix+name for name in HINT_ASSETS}
                if metadata is not None and metadata.get('algorithm') in coverage.RECALCULATION_ALGORITHMS else set())
    present = {name for name in names if name.startswith(prefix+HINT_PREFIX)}
    if present != expected:
        raise ValueError('覆盖复算引导资产缺失、孤立或包含递归历史')


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):raise ValueError('覆盖清单包含重复 JSON 字段')
    return result


def load_assets(folder: str | Path, check: Callable[[], None] = lambda: None, *, prefix: str = '',
                _hint_only: bool = False, _remaining_bytes: int = coverage.MAX_ARRAY_BYTES) -> coverage.Certificate:
    """Bounded structural read only; callers MUST reproduce before accepting claims."""
    folder = Path(folder)
    check()
    path = folder/(prefix+MANIFEST)
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('覆盖清单缺失或超过 1 MiB')
    metadata = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_object)
    if not isinstance(metadata, dict):raise ValueError('覆盖清单必须是对象')
    coverage.validate_metadata(metadata)
    required_bytes = sum(math.prod(row['shape'])*np.dtype(row['dtype']).itemsize for row in metadata['arrays'].values())
    if required_bytes > _remaining_bytes:
        raise ValueError('覆盖证据与复算引导合计超过 256 MiB 预算')
    if not _hint_only:
        check_hint_inventory({p.name for p in folder.iterdir() if p.is_file()}, metadata, prefix=prefix)
    asset = folder/(prefix+ASSET)
    if not asset.is_file() or asset.stat().st_size > coverage.MAX_ARRAY_BYTES:
        raise ValueError('覆盖数组缺失或超过 256 MiB')
    arrays = {}
    # Read through one ZIP handle; never reopen an unchecked path with np.load.
    with zipfile.ZipFile(asset) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(coverage.DTYPES) or set(names) != {key+'.npy' for key in coverage.DTYPES}:
            raise ValueError('覆盖数组包包含缺失、重复或未知条目')
        if sum(item.file_size for item in entries) > coverage.MAX_ARRAY_BYTES:
            raise ValueError('覆盖数组解压大小超过 256 MiB')
        for key in coverage.DTYPES:
            check()
            info = archive.getinfo(key+'.npy')
            if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError('覆盖数组压缩方式不受支持')
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=8192)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=8192)
                else:raise ValueError('覆盖数组 NPY 版本不受支持')
                record = metadata['arrays'][key]
                if (list(shape) != record['shape'] or dtype.str != record['dtype'] or fortran or dtype.hasobject):
                    raise ValueError('覆盖数组头与受界的清单不一致：'+key)
                size = math.prod(shape)*dtype.itemsize
                if info.file_size != stream.tell()+size:raise ValueError('覆盖数组字节长度不一致：'+key)
                payload = stream.read(size+1)
                if len(payload) != size:raise ValueError('覆盖数组截断或多出字节：'+key)
                arrays[key] = np.frombuffer(payload, dtype=dtype).reshape(shape)
    certificate = coverage.Certificate(deepcopy(metadata), arrays)
    coverage._checked_assets(certificate, check)
    if not _hint_only and metadata['algorithm'] in coverage.RECALCULATION_ALGORITHMS:
        from femto.planning.coverage_recalculation import RecalculatedCertificate, require_guide
        hint = load_assets(folder, check, prefix=prefix+HINT_PREFIX, _hint_only=True,
                           _remaining_bytes=_remaining_bytes-required_bytes)
        certificate = RecalculatedCertificate(certificate.metadata, arrays, hint)
        require_guide(certificate, check)
    return certificate


def read(folder: str | Path, domain: CoverageDomain, seeds: list[dict], model: dict, *,
         route_binding: dict | None = None, check: Callable[[], None] = lambda: None) -> coverage.Certificate:
    """Bounded allocation and full semantic reproduction, including saved stopping steps."""
    return coverage.validate(load_assets(folder, check), domain, seeds, model, route_binding=route_binding, check=check)


def save(folder: str | Path, certificate: coverage.Certificate, domain: CoverageDomain,
         seeds: list[dict], model: dict, *, route_binding: dict | None = None,
         check: Callable[[], None] = lambda: None,
         commit: Callable[[Callable[[], Any]], Any] | None = None) -> Path:
    """Create a new derived directory; cancellation/failure never replaces a result."""
    path = Path(folder).resolve()
    if path.exists():raise FileExistsError('覆盖结果目录已存在，必须使用新目录：'+str(path))
    check()
    coverage._checked_assets(certificate, check)
    path.parent.mkdir(parents=True, exist_ok=True)
    with staging_directory(path.parent, '.coverage-stage-') as stage:
        write_checked(stage, certificate, domain, seeds, model, route_binding=route_binding, check=check)
        def install():
            if path.exists():raise FileExistsError('发布期间覆盖结果目录已被创建')
            stage.rename(path)
        publish(install, check, commit)
    return path


def write_checked(folder: str | Path, certificate: coverage.Certificate, domain: CoverageDomain,
                  seeds: list[dict], model: dict, *, route_binding: dict | None = None,
                  check: Callable[[], None] = lambda: None) -> None:
    """Add assets to a caller-owned transaction; never publish or overwrite here."""
    write_assets(folder, certificate, check)
    read(folder, domain, seeds, model, route_binding=route_binding, check=check)
    for name in asset_names(certificate.metadata):durable(Path(folder)/name)


def write_assets(folder: str | Path, certificate: coverage.Certificate, check=lambda: None, *, prefix: str = '',
                 _hint_only: bool = False) -> None:
    """Write to a caller-owned stage; structural checks alone do not authorize publication."""
    folder = Path(folder)
    coverage._checked_assets(certificate, check)
    hint = None
    if not _hint_only and certificate.metadata['algorithm'] in coverage.RECALCULATION_ALGORITHMS:
        from femto.planning.coverage_recalculation import require_guide
        hint = require_guide(certificate, check)
    check()
    with (folder/(prefix+ASSET)).open('xb') as stream:
        np.savez_compressed(stream, **{key:certificate.arrays[key] for key in sorted(coverage.DTYPES)})
    with (folder/(prefix+MANIFEST)).open('xb') as stream:
        stream.write(canonical_json_bytes(certificate.metadata))
    if hint is not None:
        write_assets(folder, hint, check, prefix=prefix+HINT_PREFIX, _hint_only=True)
