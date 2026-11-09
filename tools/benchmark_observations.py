"""Optional sampled process observations, separate from geometry evidence."""
from pathlib import Path
import hashlib
from importlib.metadata import version,PackageNotFoundError
from importlib import import_module
import json
import platform
import threading
import time
import psutil


class Observation:
    def __enter__(self):
        self.process=psutil.Process();self.stop=threading.Event()
        self.peak=0;self.available=None;self.samples=0
        self.cpu_start=sum(self.process.cpu_times()[:2]);self.started=time.perf_counter()
        self._sample()
        def watch():
            while not self.stop.wait(.05):self._sample()
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
        return self

    def _sample(self):
        self.peak=max(self.peak,self.process.memory_info().rss)
        available=psutil.virtual_memory().available
        self.available=available if self.available is None else min(self.available,available)
        self.samples+=1

    def __exit__(self,*args):
        self.stop.set();self.thread.join();self._sample()
        self.result=dict(sampled_peak_rss_bytes=self.peak,minimum_available_bytes=self.available,
            samples=self.samples,sampling_seconds=.05,hard_memory_limit=False,
            process_cpu_seconds=sum(self.process.cpu_times()[:2])-self.cpu_start,
            wall_seconds=time.perf_counter()-self.started)


def source_identity(code_root,output):
    root=Path(code_root)
    files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted((root/'femto').rglob('*.py'))}
    encoded=json.dumps(files,sort_keys=True,separators=(',',':')).encode('utf-8')
    (Path(output)/'source_manifest.json').write_bytes(encoded)
    packages={}
    for name in ('numpy','open3d','scipy','PySide6','psutil'):
        try:packages[name]=version(name)
        except PackageNotFoundError:
            try:packages[name]=getattr(import_module(name),'__version__','metadata_unavailable')
            except ImportError:packages[name]='unavailable'
    return dict(code_tree_sha256=hashlib.sha256(encoded).hexdigest(),files=len(files),
        python=platform.python_version(),platform=platform.platform(),processor=platform.processor(),
        runtime_packages=packages,
        total_memory_bytes=psutil.virtual_memory().total,available_at_start_bytes=psutil.virtual_memory().available)
