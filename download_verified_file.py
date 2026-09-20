"""Download a large immutable file in verified HTTP range chunks.

The downloader exists for the multi-GB official PyTorch Windows wheel, because
the local single-connection route is heavily throttled.  It never installs or
executes the downloaded file.  Chunks are merged only after every Content-Range
has been validated, then SHA-256 must match the caller-provided full digest or
the upstream-published hash prefix.  Prefix mode requires at least 8 hex digits
and is intended for official PyTorch model URLs whose filenames publish 8.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import math
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verified concurrent range downloader")
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", type=int, required=True)
    digest = parser.add_mutually_exclusive_group(required=True)
    digest.add_argument("--sha256")
    digest.add_argument("--sha256-prefix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--retries", type=int, default=12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_range(start: int, end: int, total: int) -> str:
    return f"bytes {start}-{end}/{total}"


def download_verified(
    *,
    url: str,
    total_size: int,
    expected_sha256: str,
    output: Path,
    workers: int,
    chunk_size: int,
    retries: int,
) -> Path:
    if total_size <= 0 or chunk_size <= 0 or workers <= 0:
        raise ValueError("size, chunk size, and worker count must be positive")
    expected_sha256 = expected_sha256.lower().strip()
    if not 8 <= len(expected_sha256) <= 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected SHA-256 must contain 8 to 64 hexadecimal characters")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size == total_size:
        actual = sha256_file(output)
        if actual.startswith(expected_sha256):
            print(f"Already verified: {output}", flush=True)
            return output
        output.unlink()

    chunk_dir = output.with_name(output.name + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    count = math.ceil(total_size / chunk_size)
    ranges: list[tuple[int, int, int, Path]] = []
    for index in range(count):
        start = index * chunk_size
        end = min(total_size - 1, start + chunk_size - 1)
        ranges.append((index, start, end, chunk_dir / f"{index:05d}.part"))

    stop_progress = threading.Event()
    started = time.monotonic()

    def completed_bytes() -> int:
        total = 0
        for _, start, end, path in ranges:
            expected = end - start + 1
            try:
                total += min(path.stat().st_size, expected)
            except FileNotFoundError:
                pass
        return total

    def report_progress() -> None:
        while not stop_progress.wait(15.0):
            done = completed_bytes()
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"Downloaded {done / 1024**3:.2f}/{total_size / 1024**3:.2f} GiB "
                f"({done / total_size * 100:.1f}%), average {done / 1024**2 / elapsed:.2f} MiB/s",
                flush=True,
            )

    def fetch(item: tuple[int, int, int, Path]) -> int:
        index, start, end, path = item
        expected = end - start + 1
        if path.is_file() and path.stat().st_size == expected:
            return expected
        temp = path.with_suffix(".tmp")
        for attempt in range(1, retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={start}-{end}",
                        "User-Agent": "FemtoGsplatVerifiedDownloader/1.0",
                    },
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    content_range = response.headers.get("Content-Range", "").lower()
                    if response.status != 206 or content_range != _expected_range(start, end, total_size):
                        raise RuntimeError(
                            f"chunk {index}: invalid range response {response.status} {content_range!r}"
                        )
                    with temp.open("wb") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                actual_size = temp.stat().st_size
                if actual_size != expected:
                    raise RuntimeError(f"chunk {index}: received {actual_size}, expected {expected}")
                os.replace(temp, path)
                return expected
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                temp.unlink(missing_ok=True)
                if attempt == retries:
                    raise RuntimeError(f"chunk {index} failed after {retries} attempts: {exc}") from exc
                time.sleep(min(2.0 * attempt, 20.0))
        raise AssertionError("unreachable")

    reporter = threading.Thread(target=report_progress, daemon=True)
    reporter.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, item) for item in ranges]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        stop_progress.set()
        reporter.join(timeout=1.0)

    merge_temp = output.with_suffix(output.suffix + ".merging")
    digest = hashlib.sha256()
    with merge_temp.open("wb") as target:
        for _, start, end, path in ranges:
            expected = end - start + 1
            if not path.is_file() or path.stat().st_size != expected:
                raise RuntimeError(f"missing or incomplete chunk: {path}")
            with path.open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    target.write(block)
                    digest.update(block)
    if merge_temp.stat().st_size != total_size:
        raise RuntimeError("merged file size does not match the official size")
    actual_sha256 = digest.hexdigest()
    if not actual_sha256.startswith(expected_sha256):
        merge_temp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch: expected {expected_sha256}, received {actual_sha256}"
        )
    os.replace(merge_temp, output)
    shutil.rmtree(chunk_dir)
    print(f"Verified SHA-256 {actual_sha256}", flush=True)
    print(f"Saved {output}", flush=True)
    return output


def main() -> int:
    args = parse_args()
    download_verified(
        url=args.url,
        total_size=args.size,
        expected_sha256=args.sha256 or args.sha256_prefix,
        output=args.output,
        workers=args.workers,
        chunk_size=args.chunk_mib * 1024 * 1024,
        retries=args.retries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
