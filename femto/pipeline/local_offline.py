"""Bounded offline local reconstruction and cancellation-aware result publication.

The child has no camera and never publishes a business index. The capture worker
commits only completed, source-bound results while holding its cancellation lock.
"""
from __future__ import annotations
from typing import Any, Callable


import argparse
import copy
import json
import os
from pathlib import Path
import sys
import uuid

from femto.io_utils import load_json_object, write_json_strict
from femto.local.local_processing import empty_report, result_quality, write_html

TIME_BUDGET_S = 600.0
MEMORY_BUDGET_MB = 4096
PUBLICATION_SCHEMA = 'femto-local-publication-v1'


class PublicationError(RuntimeError):
    def __init__(self, reason, details, status='failed'):
        super().__init__(reason)
        self.details = details
        self.status = status


def publish_result(result: str | Path, outputs: str | Path, session_id: str, *, cancelled: Callable[..., Any]=lambda: False) -> Any:
    """Commit indexes after all work; rollback only values written by this commit.

    Caller serializes cancellation against this short commit. Two index files
    cannot be renamed atomically together, so partial/rollback facts are explicit.
    """
    result, outputs = Path(result).resolve(), Path(outputs).resolve()
    def check_cancel():
        if cancelled():
            from femto.pipeline.local_comparison import StageStopped
            raise StageStopped('cancelled','Cancelled before publication commit',{})
    check_cancel()
    summary = load_json_object(result / 'demo_summary.json')
    receipt_path = result / 'publication.json'
    receipt = load_json_object(receipt_path)
    quality = result_quality(result)
    if (result.parent != outputs or summary.get('publication_id') != session_id
            or summary.get('publication_required') is not True
            or receipt.get('schema') != PUBLICATION_SCHEMA
            or receipt.get('status') != 'pending' or receipt.get('session_id') != session_id
            or Path(receipt.get('result', '')).resolve() != result
            or quality.get('read_error') or quality.get('state') != 'complete'
            or quality.get('source_session_id') != session_id
            or quality.get('reconstruction', {}).get('status') != 'complete'):
        raise ValueError('Unfinished or mismatched local result cannot be published')
    required = ['demo_report.html', 'processing_report.html', 'result_quality.html',
                'reconstruction/reconstruction_report.json']
    nerfstudio_skipped = (summary.get('nerfstudio') is None
        and summary.get('local_measured') is True
        and summary.get('reconstruction_source_mode') == 'moving'
        and summary.get('nerfstudio_skipped_reason') == 'local_capture_has_fewer_than_12_views')
    if not nerfstudio_skipped:
        required.append('nerfstudio/transforms.json')
    for name in required:
        if not (result / name).is_file():
            raise ValueError('Missing completed result artifact: ' + name)
    values = {outputs / 'latest_complete_demo.txt': str(result / 'reconstruction').encode('utf-8')}
    if (summary.get('data_kind') == 'real_femto' and summary.get('fixed_camera') is False
            and type(summary.get('frame_count')) is int and summary['frame_count'] >= 3):
        values[outputs / 'latest_moving_reconstruction.txt'] = str(result).encode('utf-8')
    before, temporaries, committed, rolled_back, rollback_errors = {}, {}, [], [], []
    token = uuid.uuid4().hex
    commit_file = result / ('publication.' + token + '.tmp')
    try:
        for path, value in values.items():
            before[path] = path.read_bytes() if path.exists() else None
            temporary = path.with_name(path.name + '.' + token + '.tmp')
            with temporary.open('xb') as stream:
                stream.write(value)
            temporaries[path] = temporary
        published = dict(receipt, status='published', published_markers=[str(p) for p in values])
        with commit_file.open('x', encoding='utf-8') as stream:
            json.dump(published, stream, ensure_ascii=False, indent=2, allow_nan=False)
        for path, temporary in temporaries.items():
            check_cancel()
            # Do not clobber another producer's update made during preparation.
            current = path.read_bytes() if path.exists() else None
            if current != before[path]:
                raise RuntimeError('Latest index changed concurrently: ' + str(path))
            os.replace(temporary, path)
            committed.append(path)
        check_cancel()
        os.replace(commit_file, receipt_path)
        return published
    except Exception as error:
        for path in reversed(committed):
            try:
                if path.read_bytes() != values[path]:
                    rollback_errors.append('Index changed by another producer: ' + str(path))
                    continue
                if before[path] is None:
                    path.unlink()  # Only this exact newly created index, never a directory.
                else:
                    rollback = path.with_name(path.name + '.' + token + '.rollback')
                    with rollback.open('xb') as stream:
                        stream.write(before[path])
                    os.replace(rollback, path)
                rolled_back.append(str(path))
            except Exception as rollback_error:
                rollback_errors.append(str(rollback_error))
        active = []
        for path, value in values.items():
            try:
                if path.is_file() and path.read_bytes() == value:
                    active.append(str(path))
            except OSError:
                rollback_errors.append('Could not inspect index: ' + str(path))
        details = dict(committed_markers=[str(p) for p in committed], rolled_back_markers=rolled_back,
                       remaining_markers=active, rollback_errors=rollback_errors,
                       successful_model_published=bool(active))
        status='failed' if active else getattr(error,'status','failed')
        try:
            write_json_strict(receipt_path, dict(receipt, status=status, error=str(error), **details))
        except Exception as report_error:
            details['publication_report_error'] = str(report_error)
        raise PublicationError(str(error), details, status) from error
    finally:
        for temporary in [*temporaries.values(), commit_file]:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class LocalOfflineJob:
    def __init__(self, dataset, output, outputs, session_id, cancelled, progress=None,
                 *, timeout_s=TIME_BUDGET_S, memory_mb=MEMORY_BUDGET_MB):
        self.dataset, self.output, self.outputs = map(lambda p: Path(p).resolve(), (dataset, output, outputs))
        if self.output == self.dataset or self.dataset in self.output.parents:
            raise ValueError('Offline reports must be outside the posed input')
        self.output.mkdir(parents=True, exist_ok=False)
        self.session_id, self.cancelled, self.progress = session_id, cancelled, progress
        self.timeout_s, self.memory_mb = timeout_s, memory_mb
        self.path = self.output / 'optimization' / 'processing_report.json'
        self.report, self.result, self.stats = empty_report(), None, {}
        self._last_journal = None

    def _refresh(self, stats=None):
        if stats is not None:
            self.stats = stats
        try:
            raw = self.path.read_bytes()
            value = json.loads(raw.decode('utf-8-sig'))
            if value.get('source_session_id') != self.session_id:
                raise ValueError('Offline progress session mismatch')
            self.report = value
            self.report['worker_resources'] = copy.deepcopy(self.stats)
            if raw != self._last_journal and self.progress:
                self.progress(copy.deepcopy(self.report))
            self._last_journal = raw
        except FileNotFoundError:
            pass  # Worker has not yet imported the geometry runtime.

    def _created_result(self):
        path = self.output / 'created_result.json'
        if not path.exists():
            return None
        created = load_json_object(path)
        result = Path(created['result']).resolve()
        if result.parent != self.outputs or created.get('session_id') != self.session_id:
            raise ValueError('Offline result path/session mismatch')
        self._owned_result(result)
        self.result = result
        return result

    def _owned_result(self, result):
        """A child sidecar is insufficient authority to edit an existing result."""
        result=Path(result).resolve()
        receipt=load_json_object(result/'publication.json')
        if (result.parent != self.outputs or receipt.get('schema') != PUBLICATION_SCHEMA
                or receipt.get('session_id') != self.session_id
                or Path(receipt.get('result','')).resolve() != result
                or receipt.get('status') not in ('pending','cancelled','failed')):
            raise ValueError('Offline result publication ownership mismatch')
        return receipt

    def run(self) -> Any:
        from femto.pipeline.local_comparison import StageStopped, supervise
        request = dict(dataset=str(self.dataset), output=str(self.output), outputs=str(self.outputs),
                       session_id=self.session_id)
        write_json_strict(self.output / 'request.json', request)
        try:
            self.stats = supervise([sys.executable, '-X', 'utf8', '-m', 'femto.pipeline.local_offline',
                                   '--worker', str(self.output / 'request.json')],
                self.output / 'worker.log', timeout_s=self.timeout_s, memory_mb=self.memory_mb,
                cancelled=self.cancelled, progress=self._refresh)
            self._refresh(self.stats)
            self._created_result()
            completed = load_json_object(self.output / 'worker_result.json')
            if (completed.get('status') != 'complete' or self.result is None
                    or Path(completed['result']).resolve() != self.result
                    or completed.get('session_id') != self.session_id):
                raise ValueError('Missing or mismatched offline completion receipt')
            if self.cancelled():
                raise StageStopped('cancelled', 'Cancelled before result publication', self.stats)
            summary=load_json_object(self.result/'demo_summary.json')
            existing=result_quality(self.result)
            if (summary.get('publication_required') is not True
                    or summary.get('publication_id') != self.session_id
                    or existing.get('read_error') or existing.get('state') != 'complete'
                    or any(existing.get(key) != self.report.get(key) for key in
                           ('source_session_id','reference_frame_id','actual_dataset','actual_pose_source'))):
                raise ValueError('Completed result does not match its worker journal')
            self.report['worker_resources'] = copy.deepcopy(self.stats)
            self._save_report()
            return self.result, summary
        except Exception as error:
            if isinstance(error, StageStopped):
                self.stats = error.stats
            self.terminal(getattr(error, 'status', 'failed'), str(error))
            raise

    def _save_report(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json_strict(self.path, self.report)
        write_html(self.path.with_suffix('.html'), self.report)
        if self.result is not None:
            self._owned_result(self.result)
            write_json_strict(self.result / 'processing_report.json', self.report)
            write_html(self.result / 'processing_report.html', self.report)

    def terminal(self, status: str, error: Any) -> list[Any]:
        """Reporting failures cannot suppress the caller's terminal event."""
        failures = []
        try:
            self._refresh()
            self._created_result()
        except Exception as report_error:
            failures.append(str(report_error))
        self.report.update(state=status, error=error, source_session_id=self.session_id,
                           worker_resources=copy.deepcopy(self.stats))
        for stage in ('refinement', 'temporal', 'reconstruction'):
            if self.report.get(stage, {}).get('status') == 'running':
                self.report[stage].update(status=status, error=error)
        try:
            self._save_report()
        except Exception as report_error:
            failures.append(str(report_error))
        if self.result is not None:
            try:
                receipt = self._owned_result(self.result)
                if receipt.get('status') == 'pending':
                    write_json_strict(self.result / 'publication.json', dict(receipt, status=status, error=error))
            except Exception as report_error:
                failures.append(str(report_error))
        if failures:
            self.report['reporting_errors'] = failures
        return failures


def _source_paths(dataset):
    """Only immutable evidence; newly created derived reports are deliberately separate."""
    dataset = Path(dataset)
    manifest = load_json_object(dataset / 'manifest.json')
    roots = {dataset, Path(manifest['source_dataset']).resolve()}
    paths = set()
    for root in roots:
        paths.update((root / 'manifest.json', root / 'frames.jsonl'))
        paths.update(p for p in (root / 'calibration').rglob('*') if p.is_file())
        for row in (root / 'frames.jsonl').read_text(encoding='utf-8-sig').splitlines():
            if row.strip():
                frame = json.loads(row)
                paths.update((root / frame[key]).resolve() for key in ('rgb', 'depth', 'pose'))
    return paths


def worker(request_file: Any) -> Any:
    from femto.pipeline.local_comparison import changed_files, fingerprints
    from femto.local.local_processing import LocalProcessing
    request = load_json_object(request_file)
    output = Path(request['output'])
    job = None
    try:
        before = fingerprints(_source_paths(request['dataset']))
        if any(value is None for value in before.values()):
            raise ValueError('Missing immutable source evidence')
        write_json_strict(output / 'source_hashes.json', before)
        job = LocalProcessing(request['dataset'], output / 'optimization')
        if job.report.get('source_session_id') != request['session_id']:
            raise ValueError('Offline source session mismatch')
        selected, _ = job.run()
        job.reconstruction('running')
        from femto.pipeline.complete_demo import run_demo
        result, _ = run_demo(argparse.Namespace(mode='latest', dataset=selected,
            datasets_root=Path(request['dataset']).parent, outputs_root=Path(request['outputs']),
            no_viewer=True, publish_latest=False, no_source_report=True,
            publication_id=request['session_id'], created_result_file=output / 'created_result.json',
            local_processing_report=str(job.path)))
        changed = changed_files(before)
        write_json_strict(output / 'source_integrity.json', dict(source_unchanged=not changed, changed_files=changed))
        if changed:
            raise ValueError('Source evidence changed during offline processing')
        job.report['result'] = str(result)
        job.reconstruction('complete')
        write_json_strict(output / 'worker_result.json', dict(status='complete', session_id=request['session_id'], result=str(result)))
        return 0
    except Exception as error:
        if job is not None:
            try:
                job.reconstruction('failed', str(error))
            except Exception:
                pass
        try:
            write_json_strict(output / 'worker_result.json', dict(status='failed', error=str(error), session_id=request['session_id']))
        except Exception:
            pass
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    sys.exit(worker(parser.parse_args().worker))
