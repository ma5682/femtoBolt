"""Shared workstation/exclusive maintenance lease, released by the OS on exit.

This gate coordinates cooperating applications only. Command-line asset writers
must be stopped before cold maintenance; it is not a filesystem-wide lock.
"""
import os
from contextlib import contextmanager
from pathlib import Path


class DataLease:
    def __init__(self, workspace, *, maintenance=False):
        self.workspace=Path(workspace).resolve()
        self.maintenance=maintenance
        self.stream=None
        self.overlapped=None

    def acquire(self):
        if self.stream is not None:
            raise RuntimeError('Data lease is already held')
        stream=(self.workspace/'.data_maintenance.lock').open('a+b')
        try:
            if stream.seek(0,os.SEEK_END)==0:
                stream.write(b'\0');stream.flush()
            if os.name=='nt':
                import ctypes
                from ctypes import wintypes
                import msvcrt
                class Overlapped(ctypes.Structure):
                    _fields_=[('Internal',ctypes.c_size_t),('InternalHigh',ctypes.c_size_t),
                              ('Offset',wintypes.DWORD),('OffsetHigh',wintypes.DWORD),('hEvent',wintypes.HANDLE)]
                self.overlapped=Overlapped()
                kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                kernel.LockFileEx.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.DWORD,
                                           wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(Overlapped)]
                kernel.LockFileEx.restype=wintypes.BOOL
                flags=1|(2 if self.maintenance else 0)
                if not kernel.LockFileEx(msvcrt.get_osfhandle(stream.fileno()),flags,0,1,0,ctypes.byref(self.overlapped)):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl
                fcntl.flock(stream.fileno(),(fcntl.LOCK_EX if self.maintenance else fcntl.LOCK_SH)|fcntl.LOCK_NB)
        except OSError as error:
            stream.close();self.overlapped=None
            message=('请先关闭本工作区的工作台实例，再进行数据维护。' if self.maintenance
                     else '数据正在备份或恢复，请完成维护后再打开工作台。')
            raise RuntimeError(message) from error
        self.stream=stream
        return self

    def release(self):
        # Closing the handle releases the exact shared/exclusive OS lock.
        if self.stream is not None:
            self.stream.close();self.stream=None;self.overlapped=None

    def __enter__(self):return self.acquire()
    def __exit__(self,*_):self.release()


@contextmanager
def maintenance_session(workspace, workstation_lease=None):
    """Temporarily upgrade an idle owning workstation to exclusive maintenance."""
    if workstation_lease is not None:
        if (workstation_lease.workspace!=Path(workspace).resolve()
                or workstation_lease.maintenance or workstation_lease.stream is None):
            raise RuntimeError('工作台数据锁状态无效。')
        workstation_lease.release()
    try:
        with DataLease(workspace,maintenance=True):yield
    finally:
        if workstation_lease is not None:workstation_lease.acquire()
