"""One outstanding immutable RGB-D write, overlapped with pose estimation.

Admission and preview wait for the receipt. There is no detached save backlog;
the next frame cannot be submitted until the previous frame has committed.
"""
from concurrent.futures import ThreadPoolExecutor
import time


class RawFrameWriter:
    def __init__(self, save):
        self.save = save
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='LocalRawWriter')
        self.pending = None

    def submit(self, scan, frame, index):
        if self.pending is not None:
            raise RuntimeError('Previous raw frame receipt has not been consumed')
        self.pending = self.pool.submit(self._write, scan, frame, index)

    def _write(self, scan, frame, index):
        started = time.perf_counter()
        if scan.frame_count != index:
            raise ValueError('Raw save index mismatch')
        self.save(scan, frame, lossless=True)
        if scan.frame_count != index+1:
            raise ValueError('Raw save did not commit the expected frame')
        return dict(frame_id=index, capture_timestamp_us=int(frame.timestamp_us),
                    color_timestamp_us=int(frame.color_timestamp_us), depth_timestamp_us=int(frame.depth_timestamp_us),
                    write_ms=(time.perf_counter()-started)*1000)

    def commit(self, record):
        if self.pending is None:
            raise RuntimeError('No pending raw frame')
        try:
            receipt = self.pending.result()
        finally:
            self.pending = None
        if any(receipt[k] != record.get(k) for k in ('frame_id', 'capture_timestamp_us',
                                                   'color_timestamp_us', 'depth_timestamp_us')):
            raise ValueError('RGB-D write receipt / pose timestamp mismatch')
        return receipt

    def close(self):
        self.pool.shutdown(wait=True)
