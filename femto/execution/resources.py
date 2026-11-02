"""Independent process-level simulation resource ownership, released by the OS."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from femto.paths import execution_resource_root


class ResourceLease:
    def __init__(self, keys, *, owner, outputs=None):
        self.keys = tuple(sorted(set(keys)))
        if not self.keys or any(not key.startswith('simulation:') for key in self.keys):
            raise ValueError('This coordinator leases only explicit simulation resources')
        self.owner = owner
        self.root = execution_resource_root(outputs)
        self._streams = []

    def acquire(self):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            for key in self.keys:
                path = self.root/(hashlib.sha256(key.encode()).hexdigest()+'.lock')
                stream = path.open('a+b')
                if stream.seek(0, os.SEEK_END) == 0:
                    stream.write(b'\0'); stream.flush()
                stream.seek(0)
                try:
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    stream.close()
                    raise RuntimeError('模拟资源已被另一个窗口或进程占用：'+key) from error
                self._streams.append(stream)
                stream.seek(1); stream.truncate()
                stream.write(json.dumps(dict(owner=self.owner, pid=os.getpid(), resource=key), ensure_ascii=False).encode())
                stream.flush()
        except BaseException:
            self.release()
            raise
        return self

    @property
    def held(self):
        return bool(self._streams)

    def release(self):
        for stream in reversed(self._streams):
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
        self._streams.clear()
