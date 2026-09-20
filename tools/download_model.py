"""Download the official MediaPipe Pose Landmarker Full task model."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "models" / "pose_landmarker_full.task"
URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
CHUNK = 512 * 1024


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        size = int(response.headers["Content-Length"])
    if OUTPUT.exists() and OUTPUT.stat().st_size == size:
        print(f"Model already present: {OUTPUT}")
        return

    def fetch(start: int) -> tuple[int, bytes]:
        end = min(size, start + CHUNK) - 1
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"Incomplete model range {start}-{end}")
        return start, data

    starts = list(range(0, size, CHUNK))
    with ThreadPoolExecutor(max_workers=16) as pool:
        parts = dict(pool.map(fetch, starts))
    temporary = OUTPUT.with_suffix(".task.download")
    with temporary.open("wb") as output:
        for start in starts:
            output.write(parts[start])
    temporary.replace(OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    OUTPUT.with_suffix(".task.sha256").write_text(digest + "\n", encoding="ascii")
    print(f"Downloaded {OUTPUT} ({size:,} bytes, SHA-256 {digest})")


if __name__ == "__main__":
    main()
