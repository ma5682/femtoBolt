"""Best-effort Windows hints for one dedicated interactive task thread.

These are scheduler preferences, never affinity, priority or system settings.
The caller must use a new thread that exits after this task: Windows exposes no
documented query for its previous ThreadPowerThrottling policy. Ideal-processor
preferences are restored on exit; the HighQoS preference ends with the thread.
"""
from contextlib import contextmanager
import ctypes
import struct
import sys


class _Processor(ctypes.Structure):
    _fields_ = [('Group', ctypes.c_uint16), ('Number', ctypes.c_uint8), ('Reserved', ctypes.c_uint8)]


class _Throttle(ctypes.Structure):
    _fields_ = [('Version', ctypes.c_uint32), ('ControlMask', ctypes.c_uint32), ('StateMask', ctypes.c_uint32)]


class _GroupAffinity(ctypes.Structure):
    _fields_ = [('Mask', ctypes.c_size_t), ('Group', ctypes.c_uint16), ('Reserved', ctypes.c_uint16 * 3)]


class _WindowsThread:
    def __init__(self):
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self._bind('GetCurrentThread', [], ctypes.c_void_p)
        self._bind('GetCurrentProcess', [], ctypes.c_void_p)
        self.thread = self.api.GetCurrentThread()

    def _bind(self, name, args, result):
        function = getattr(self.api, name)
        function.argtypes, function.restype = args, result
        return function

    def high_qos(self):
        call = self._bind('SetThreadInformation', [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int)
        # Windows SDK: ThreadPowerThrottling=3; Version=1; EXECUTION_SPEED=1.
        # Microsoft's documented HighQoS example turns that mechanism off.
        state = _Throttle(1, 1, 0)
        return bool(call(self.thread, 3, ctypes.byref(state), ctypes.sizeof(state)))

    def topology(self):
        call = self._bind('GetSystemCpuSetInformation',
            [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int)
        size = ctypes.c_uint32()
        call(None, 0, ctypes.byref(size), self.api.GetCurrentProcess(), 0)
        if not 0 < size.value <= 1024 * 1024:return []
        buffer = ctypes.create_string_buffer(size.value)
        if not call(buffer, len(buffer), ctypes.byref(size), self.api.GetCurrentProcess(), 0):return []
        if size.value > len(buffer):return []
        rows = [];offset = 0
        while offset < size.value:
            if size.value - offset < 8:return []
            length, kind = struct.unpack_from('<II', buffer, offset)
            if length < 8 or offset + length > size.value:return []
            if kind == 0:
                if length < 32:return []
                group, logical = struct.unpack_from('<HB', buffer, offset + 12)
                efficiency, flags = struct.unpack_from('<BB', buffer, offset + 18)
                rows.append(dict(group=group, logical=logical, efficiency=efficiency,
                    parked=bool(flags & 1), allocated=bool(flags & 2), own=bool(flags & 4)))
            offset += length
        return rows

    def affinity(self):
        call = self._bind('GetThreadGroupAffinity', [ctypes.c_void_p, ctypes.POINTER(_GroupAffinity)], ctypes.c_int)
        value = _GroupAffinity()
        if not call(self.thread, ctypes.byref(value)):return None
        return value.Group, value.Mask

    def prefer(self, group, number):
        call = self._bind('SetThreadIdealProcessorEx',
            [ctypes.c_void_p, ctypes.POINTER(_Processor), ctypes.POINTER(_Processor)], ctypes.c_int)
        previous = _Processor();desired = _Processor(group, number, 0)
        if not call(self.thread, ctypes.byref(desired), ctypes.byref(previous)):return None
        return previous

    def restore(self, previous):
        return bool(self.api.SetThreadIdealProcessorEx(self.thread, ctypes.byref(previous), None))


@contextmanager
def interactive_task_thread():
    """Hint only this dedicated worker; unsupported APIs cannot fail its job."""
    report = dict(high_qos=False, preferred_processor=None)
    windows = None;previous = None
    if sys.platform == 'win32':
        try:windows = _WindowsThread()
        except Exception:pass
    if windows is not None:
        try:report['high_qos'] = windows.high_qos()
        except Exception:pass
        try:
            rows = windows.topology();affinity = windows.affinity()
            classes = {row['efficiency'] for row in rows}
            if affinity is not None and len(classes) > 1:
                group, mask = affinity;fastest = max(classes)
                allowed = [row for row in rows if row['efficiency'] == fastest and row['group'] == group
                    and mask & (1 << row['logical']) and not row['parked']
                    and (not row['allocated'] or row['own'])]
                if allowed:
                    # Spread independent task threads across eligible cores;
                    # the OS remains free to choose any permitted processor.
                    import threading
                    target = allowed[threading.get_native_id() % len(allowed)]
                    previous = windows.prefer(group, target['logical'])
                    if previous is not None:report['preferred_processor'] = [group, target['logical']]
        except Exception:pass
    try:
        yield report
    finally:
        if windows is not None and previous is not None:
            try:windows.restore(previous)
            except Exception:pass
