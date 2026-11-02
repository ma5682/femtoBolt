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

* JSON defaults to ``ensure_ascii=False`` and is read with ``utf-8-sig`` so
  BOM-prefixed marker files remain readable. Legacy byte-compatible writes may
  explicitly retain their earlier escaping and whitespace settings.
* JSON writes are atomic (temporary file + ``os.replace``) and create parent
  directories; ``write_json_strict`` additionally rejects NaN / Infinity.
* Image codec failures raise :class:`ImageCodecError`, which subclasses both
  ``ValueError`` and ``RuntimeError`` so existing ``except`` clauses written
  against either of the former per-module helpers keep working.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import numpy as np

from femto.resource_policy import preparation_resources


__all__ = [
    "ImageCodecError",
    "JsonSizeLimitError",
    "append_jsonl",
    "imread_unicode",
    "imread_unicode_or_none",
    "imwrite_unicode",
    "load_json",
    "load_json_object",
    "load_jsonl",
    "sha256_file",
    "write_json",
    "write_json_strict",
    "write_jsonl",
]


def sha256_file(path: str | Path, check: Callable[[], None] | None = None) -> str:
    """Hash every byte with an adaptive buffer <=1 MiB and cooperative checks."""
    if check is None:
        check = lambda: None
    check()
    digest = hashlib.sha256()
    source = Path(path)
    resources = preparation_resources(0, file_bytes=source.stat().st_size)
    buffer = bytearray(resources.hash_buffer_bytes)
    view = memoryview(buffer)
    with source.open('rb') as stream:
        while True:
            check()
            size = stream.readinto(buffer)
            if not size:
                break
            digest.update(view[:size])
    check()
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# JSON / JSONL
# ---------------------------------------------------------------------------


class JsonSizeLimitError(ValueError):
    """The encoded JSON would exceed the caller's byte limit."""


def _write_json_checked(target, value, *, indent, allow_nan, ensure_ascii,
                        durable, temporary_prefix, check, max_bytes):
    """Stream bounded UTF-8 chunks into a unique, unpublished temporary file."""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
        raise ValueError('JSON byte limit must be a nonnegative integer')
    check = check or (lambda: None)
    check()
    encoder = json.JSONEncoder(ensure_ascii=ensure_ascii, indent=indent, allow_nan=allow_nan)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent,
                prefix=temporary_prefix or f'.{target.name}-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            pending = bytearray(); size = 0; pieces = []; characters = 0
            def flush_pieces():
                nonlocal size, characters
                if not pieces:return
                # Joining small encoder fragments preserves exact JSON and
                # newline bytes while avoiding one UTF-8 encoding per token.
                raw = ''.join(pieces).replace('\n', os.linesep).encode('utf-8')
                size += len(raw)
                if max_bytes is not None and size > max_bytes:
                    raise JsonSizeLimitError(f'JSON exceeds its {max_bytes} byte limit')
                pending.extend(raw); pieces.clear(); characters = 0
                if len(pending) >= 65536:
                    check(); stream.write(pending); pending.clear()
            for index, piece in enumerate(encoder.iterencode(value)):
                if index % 1024 == 0:check()
                for start in range(0, len(piece), 16384):
                    chunk = piece[start:start+16384]
                    pieces.append(chunk); characters += len(chunk)
                    if characters >= 16384:flush_pieces()
            check()
            flush_pieces()
            check()
            if pending:stream.write(pending)
            if durable:
                check(); stream.flush(); os.fsync(stream.fileno())
        check()
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            try:temporary.unlink(missing_ok=True)
            except OSError:pass  # Preserve the actual task error if cleanup also fails.
        raise


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


def write_json(
    path: str | os.PathLike[str],
    value: Any,
    *,
    indent: int | None = 2,
    allow_nan: bool = True,
    ensure_ascii: bool = False,
    durable: bool = False,
    unique_temporary: bool = False,
    temporary_prefix: str | None = None,
    check: Callable[[], None] | None = None,
    max_bytes: int | None = None,
) -> None:
    """Atomically write UTF-8 JSON, optionally flushing it before publication.

    Supplying ``check`` or ``max_bytes`` streams into a unique temporary file,
    checks each output chunk, and enforces the encoded size before overflow.
    Unspecified options retain the existing writer and its byte format.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if check is not None or max_bytes is not None:
        _write_json_checked(target, value, indent=indent, allow_nan=allow_nan,
            ensure_ascii=ensure_ascii, durable=durable, temporary_prefix=temporary_prefix,
            check=check, max_bytes=max_bytes)
        return
    if durable or unique_temporary:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=target.parent,
                prefix=temporary_prefix or f".{target.name}-",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(
                    value,
                    stream,
                    ensure_ascii=ensure_ascii,
                    indent=indent,
                    allow_nan=allow_nan,
                )
                if durable:
                    stream.flush()
                    os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, allow_nan=allow_nan),
        encoding="utf-8",
    )
    os.replace(temporary, target)


write_json_strict = functools.partial(write_json, allow_nan=False)
write_json_strict.__doc__ = "write_json that rejects NaN / Infinity (``allow_nan=False``)."


def write_jsonl(
    path: str | os.PathLike[str],
    values: Iterable[Any],
    *,
    ensure_ascii: bool = False,
    compact: bool = True,
) -> None:
    """Overwrite ``path`` with one JSON value per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    options = {"separators": (",", ":")} if compact else {}
    target.write_text(
        "".join(json.dumps(value, ensure_ascii=ensure_ascii, **options) + "\n" for value in values),
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
