"""Unicode-safe file IO helpers shared by every module in the repository.

This is the single implementation of the JSON / JSONL / image read-write
primitives that used to be copied into a dozen modules. Two constraints shape it:

* The repository lives under a non-ASCII path, so images never go through
  ``cv2.imread`` / ``cv2.imwrite`` (their Windows path handling is narrow);
  they are decoded from and encoded to bytes instead.
* The product 3DGS runtime (``.venv-gsplat``) has no OpenCV, yet modules that run
  there need the JSON helpers. ``cv2`` is therefore imported lazily inside the
  image functions only.

Conventions (see CLAUDE.md, "Windows / 中文路径注意事项"):

* JSON is always written with ``ensure_ascii=False`` and read with
  ``utf-8-sig`` so BOM-prefixed marker files remain readable.
* JSON writes are atomic (temporary file + ``os.replace``) and create parent
  directories; ``write_json_strict`` additionally rejects NaN / Infinity.
* Image codec failures raise :class:`ImageCodecError`, which subclasses both
  ``ValueError`` and ``RuntimeError`` so existing ``except`` clauses written
  against either of the former per-module helpers keep working.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


__all__ = [
    "ImageCodecError",
    "append_jsonl",
    "imread_unicode",
    "imread_unicode_or_none",
    "imwrite_unicode",
    "load_json",
    "load_json_object",
    "load_jsonl",
    "write_json",
    "write_json_strict",
    "write_jsonl",
]


# ---------------------------------------------------------------------------
# JSON / JSONL
# ---------------------------------------------------------------------------


def load_json(path: str | os.PathLike[str]) -> Any:
    """Parse a UTF-8 JSON file; a leading BOM is tolerated."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_json_object(path: str | os.PathLike[str], *, message: str = "Expected a JSON object") -> dict[str, Any]:
    """Like :func:`load_json` but require the top-level value to be an object."""
    value = load_json(path)
    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def load_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    """Parse one JSON value per non-blank line."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: str | os.PathLike[str], value: Any, *, indent: int | None = 2, allow_nan: bool = True) -> None:
    """Atomically write ``value`` as pretty-printed UTF-8 JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=allow_nan), encoding="utf-8")
    os.replace(temporary, target)


write_json_strict = functools.partial(write_json, allow_nan=False)
write_json_strict.__doc__ = "write_json that rejects NaN / Infinity (``allow_nan=False``)."


def write_jsonl(path: str | os.PathLike[str], values: Iterable[Any]) -> None:
    """Overwrite ``path`` with one compact JSON value per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values),
        encoding="utf-8",
    )


def append_jsonl(path: str | os.PathLike[str], value: Any) -> None:
    """Append one compact JSON value as a new line."""
    with Path(path).open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


# ---------------------------------------------------------------------------
# Images (OpenCV codecs through bytes, never cv2.imread / cv2.imwrite)
# ---------------------------------------------------------------------------


class ImageCodecError(ValueError, RuntimeError):
    """An image could not be decoded or encoded."""


def imread_unicode(path: str | os.PathLike[str], flags: int) -> np.ndarray:
    """Decode an image file through ``np.fromfile`` + ``cv2.imdecode``.

    Raises ``OSError`` if the file cannot be read and :class:`ImageCodecError`
    if the bytes do not decode.
    """
    import cv2

    target = Path(path)
    encoded = np.fromfile(target, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags) if encoded.size else None
    if image is None:
        raise ImageCodecError(f"Cannot decode image: {target}")
    return image


def imread_unicode_or_none(path: str | os.PathLike[str], flags: int) -> np.ndarray | None:
    """Validator variant of :func:`imread_unicode`: unreadable or undecodable → ``None``."""
    import cv2

    try:
        encoded = np.fromfile(Path(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, flags) if encoded.size else None


def imwrite_unicode(path: str | os.PathLike[str], image: np.ndarray, params: list[int] | None = None) -> None:
    """Encode ``image`` by the target suffix and write the bytes, creating parent directories."""
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(target.suffix, image, params or [])
    if not ok:
        raise ImageCodecError(f"Cannot encode image: {target}")
    target.write_bytes(encoded.tobytes())
