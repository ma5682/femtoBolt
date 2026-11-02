"""Bounded independent file hashing; no geometry or proof parallelism."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from femto.io_utils import sha256_file
from femto.checksum_resources import ChecksumResources, checksum_resources

Check = Callable[[], None]
Hasher = Callable[[Path, Check], str]


@dataclass(frozen=True)
class ChecksumBatch:
    hashes: Mapping[Path, str]
    resources: ChecksumResources


class _ChecksumCancelled(RuntimeError):
    pass


def _parallel_hashes(paths: tuple[Path, ...], check: Check, hasher: Hasher, workers: int) -> dict[Path, str]:
    """Keep at most two unfinished tasks; propagate errors in original input order.

Only the caller thread invokes its budget/cancel check. Workers receive a
private stop check, so callbacks and geometry meters never cross threads.
Always join workers before rethrowing the original exception or publishing.
    """
    stopped = threading.Event()
    pending: deque[tuple[Path, Future[str]]] = deque()
    results: dict[Path, str] = {}

    def stop_check() -> None:
        if stopped.is_set():raise _ChecksumCancelled('Checksum batch stopped')

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='FemtoChecksum') as pool:
        remaining = iter(paths)
        def submit_next() -> None:
            path = next(remaining, None)
            if path is not None:
                check()
                pending.append((path, pool.submit(hasher, path, stop_check)))
        try:
            for _ in range(workers):submit_next()
            while pending:
                path, future = pending.popleft()
                while True:
                    check()
                    try:
                        value = future.result(timeout=.025)
                        break
                    except FutureTimeout:
                        # A completed worker can itself raise TimeoutError.
                        # Re-read completed outcomes instead of polling forever.
                        if future.done():value = future.result();break
                check()
                results[path] = value
                submit_next()
            check()
        finally:
            stopped.set()
            for _, future in pending:future.cancel()
    return results


def sha256_files(paths: Sequence[Path], check: Check | None = None, *,
                 hasher: Hasher = sha256_file) -> ChecksumBatch:
    """Freeze one bounded resource decision and return only a complete ordered map."""
    control = check if check is not None else lambda: None
    control()
    inputs = tuple(Path(path) for path in paths)
    if len(set(inputs)) != len(inputs):raise ValueError('Checksum paths must be unique')
    total_bytes = sum(path.stat().st_size for path in inputs)
    selected = checksum_resources(len(inputs), total_bytes)
    if selected.workers == 1:
        values = {path: hasher(path, control) for path in inputs}
    else:
        values = _parallel_hashes(inputs, control, hasher, selected.workers)
    control()
    return ChecksumBatch(MappingProxyType(values), selected)
