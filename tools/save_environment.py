"""Bounded system and process-load snapshots outside save timing intervals."""
import os
import psutil
from tools.runtime_memory import performance_environment as basic_environment


def performance_environment():
    result = basic_environment()
    processes, errors = [], []
    for process in psutil.process_iter(['pid', 'name', 'cpu_times', 'memory_info']):
        try:
            info = process.info
            if info['pid'] == os.getpid():
                continue
            cpu, memory = info['cpu_times'], info['memory_info']
            processes.append(dict(pid=info['pid'], name=info['name'],
                cpu_seconds=cpu.user+cpu.system if cpu else None,
                rss_bytes=memory.rss if memory else None))
        except psutil.Error as error:
            errors.append(type(error).__name__)
    largest = sorted(processes, key=lambda row: row['rss_bytes'] or 0, reverse=True)[:16]
    busiest = sorted(processes, key=lambda row: row['cpu_seconds'] or 0, reverse=True)[:16]
    result.update(other_processes=list({row['pid']: row for row in largest+busiest}.values()),
        observed_process_count=len(processes), additional_observation_errors=errors[:8],
        process_selection='Union of top 16 RSS and cumulative CPU totals; no command lines')
    return result
