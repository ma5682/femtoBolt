"""Prefer an ASCII staging directory on the destination's actual volume."""
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from femto.paths import WORKSPACE_ID
from femto.performance_detail import count


def _existing_parent(path):
    path=Path(path)
    while not path.exists():
        parent=path.parent
        if parent==path:raise FileNotFoundError(str(path))
        path=parent
    return path


def _volume_cache(final_path):
    """No change to the output directory; the native library sees ASCII only."""
    if os.name!='nt':return None
    final_path=Path(final_path).resolve()
    directory=Path(final_path.anchor)/'FemtoNativeIO'/WORKSPACE_ID
    if not str(directory).isascii():return None
    directory.mkdir(parents=True,exist_ok=True)
    if directory.stat().st_dev!=_existing_parent(final_path.parent).stat().st_dev:return None
    return directory


def temporary_path(final_path,fallback):
    final_path=Path(final_path)
    token=hashlib.sha256(str(final_path.resolve()).encode('utf-8')).hexdigest()[:16]
    def reserve(directory):
        descriptor,name=tempfile.mkstemp(prefix=token+'_',suffix=final_path.suffix,dir=directory)
        os.close(descriptor)
        path=Path(name)
        path.unlink()
        return path
    try:
        directory=_volume_cache(final_path)
        path=reserve(directory) if directory is not None else None
    except OSError:
        directory=None
        path=None
    if directory is None:
        directory=Path(fallback)
        directory.mkdir(parents=True,exist_ok=True)
        path=reserve(directory)
        count('native_io.staging',fallback_paths=1)
    else:
        count('native_io.staging',same_volume_paths=1)
    # Do not preserve a possibly non-ASCII user filename. Each invocation owns
    # a distinct temporary file, including simultaneous exports of one target.
    return path


def publish(temporary,final_path):
    """Same-volume replacement avoids a second complete read/write of the PLY."""
    try:
        os.replace(temporary,final_path)
    except OSError:
        # Permissions, unavailable staging or another filesystem retain the
        # existing move/copy behaviour and propagate its original I/O failure.
        shutil.move(str(temporary),str(final_path))
        count('native_io.publish',move_fallbacks=1)
    else:
        count('native_io.publish',replacements=1)


def cleanup(temporary):
    try:
        Path(temporary).unlink(missing_ok=True)
    except OSError:
        # Never mask the native writer's failure with a best-effort cleanup.
        count('native_io.cleanup',failed=1)
