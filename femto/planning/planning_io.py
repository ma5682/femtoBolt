"""Cooperative task feedback and a serialized cancellation/publication boundary."""
import os
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class TaskControl:
    def __init__(self, stage='准备', seconds=120):
        self.started=time.monotonic();self.stage=stage;self.seconds=seconds
        self._cancel=threading.Event();self._lock=threading.RLock();self.committed=False

    def set(self):
        with self._lock:
            if not self.committed:self._cancel.set()

    def is_set(self):return self._cancel.is_set()

    def check(self):
        if self.committed:return
        if self.is_set():raise RuntimeError('任务已取消；未发布结果')
        if time.monotonic()-self.started>self.seconds:raise TimeoutError('任务超过120秒预算；未发布结果')

    def progress(self, stage):
        self.check();self.stage=stage

    def publish(self, action):
        with self._lock:
            self.check();self.stage='发布完整结果'
            value=action();self.committed=True
            return value


def publish(action, check, commit=None):
    if commit:return commit(action)
    check();return action()


def windows_file_retry(action, timeout_s=2.0):
    """Retry only transient Windows sharing/access violations from scanners."""
    deadline=time.monotonic()+timeout_s
    delay=.025
    while True:
        try:return action()
        except PermissionError as error:
            if getattr(error,'winerror',None) not in (5,32) or time.monotonic()>=deadline:raise
            time.sleep(delay);delay=min(.2,delay*2)


@contextmanager
def staging_directory(parent, prefix='tmp'):
    """Remove staging with sharing-violation retries without hiding task errors."""
    root=Path(tempfile.mkdtemp(dir=parent,prefix=prefix))
    try:
        yield root
    finally:
        task_failed=sys.exc_info()[0] is not None
        try:windows_file_retry(lambda:shutil.rmtree(root) if root.exists() else None)
        except OSError:
            if not task_failed:raise


def durable(path):
    with open(path,'r+b') as stream:os.fsync(stream.fileno())


def copy_stream(source, target, check):
    while True:
        check();block=source.read(1024*1024)
        if not block:break
        target.write(block)
