"""Shared metadata publication preserves binding and commit/cancel semantics."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from femto import asset_storage as storage
from femto.dataset import task_context
from femto.planning import region_library
from femto.planning.planning_io import TaskControl


class HistoryMetadataWriterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_create_only_preserves_existing_binding_without_starting_publication(self):
        target = self.root / 'task_context.json'
        target.write_bytes(b'original binding')
        with self.assertRaises(FileExistsError):
            storage.publish_json(target, {'replacement': True}, create_only=True)
        self.assertEqual(target.read_bytes(), b'original binding')
        self.assertEqual(list((self.root / '.publications').glob('*.json')), [])

    def test_commit_wraps_only_final_replace_and_preparation_stays_cancellable(self):
        control = TaskControl()
        phases = []
        inside_commit = False
        receipt = storage._write_receipt
        replace = storage.os.replace

        def write_receipt(path, data):
            phases.append(data['state'])
            self.assertFalse(inside_commit)
            if data['state'] != 'published':
                self.assertFalse(control.committed)
            receipt(path, data)

        def replace_stage(source, target):
            if Path(source).suffix == '.stage':
                phases.append('replace')
                self.assertTrue(inside_commit)
                self.assertFalse(control.committed)
            return replace(source, target)

        def commit(action):
            nonlocal inside_commit
            phases.append('commit_enter')
            inside_commit = True
            try:
                return control.publish(action)
            finally:
                inside_commit = False
                phases.append('commit_exit')

        with patch.object(storage, '_write_receipt', side_effect=write_receipt), \
                patch.object(storage.os, 'replace', side_effect=replace_stage):
            storage.publish_json(self.root / 'index.json', {'version': 1},
                                 check=control.check, commit=commit)
        self.assertEqual(phases, [
            'preparing', 'ready', 'commit_enter', 'replace', 'commit_exit', 'published'])
        self.assertTrue(control.committed)
        control.set()
        control.check()  # Cancellation after the replace cannot undo publication.
        self.assertEqual(json.loads((self.root / 'index.json').read_text()), {'version': 1})

    def test_receipt_failure_after_commit_preserves_committed_state_and_can_recover(self):
        control = TaskControl()
        receipt = storage._write_receipt

        def fail_final_receipt(path, data):
            if data['state'] == 'published':
                raise OSError('final receipt unavailable')
            receipt(path, data)

        with patch.object(storage, '_write_receipt', side_effect=fail_final_receipt):
            with self.assertRaises(storage.PublicationError) as raised:
                storage.publish_json(self.root / 'index.json', {'version': 2},
                                     check=control.check, commit=control.publish)
        self.assertTrue(raised.exception.published)
        self.assertTrue(control.committed)
        self.assertEqual(storage.inspect_publication(raised.exception.journal)['status'],
                         'published_unrecorded')
        control.set()
        control.check()
        self.assertEqual(storage.recover_publication(raised.exception.journal)['status'], 'published')
        self.assertEqual(json.loads((self.root / 'index.json').read_text()), {'version': 2})

    def test_commit_callback_error_after_replace_reports_actual_published_state(self):
        def commit(action):
            action()
            raise OSError('callback failed after replacement')

        with self.assertRaises(storage.PublicationError) as raised:
            storage.publish_json(self.root / 'index.json', {'version': 3}, commit=commit)
        self.assertTrue(raised.exception.published)
        self.assertEqual(json.loads((self.root / 'index.json').read_text()), {'version': 3})
        self.assertEqual(storage.inspect_publication(raised.exception.journal)['status'],
                         'published_unrecorded')
        self.assertEqual(storage.recover_publication(raised.exception.journal)['status'], 'published')

    def test_index_cancel_after_ready_preserves_old_index_and_legacy_commit_argument(self):
        original = dict(schema=region_library.SCHEMA, records={'old': {'note': 'retain'}})
        initial = TaskControl()
        # Existing callers may supply commit as the third positional argument.
        region_library.write_index(self.root, original, initial.publish)
        self.assertTrue(initial.committed)
        index = self.root / 'region_library.json'
        before = index.read_bytes()
        evidence = self.root / 'source.bin'
        evidence.write_bytes(b'original source')
        control = TaskControl()
        receipt = storage._write_receipt

        def cancel_when_ready(path, data):
            receipt(path, data)
            if data['state'] == 'ready':
                control.set()

        with patch.object(storage, '_write_receipt', side_effect=cancel_when_ready):
            with self.assertRaises(storage.PublicationError) as raised:
                region_library.write_index(
                    self.root, dict(schema=region_library.SCHEMA, records={}),
                    control.publish, check=control.check,
                )
        self.assertFalse(control.committed)
        self.assertFalse(raised.exception.published)
        self.assertEqual(index.read_bytes(), before)
        self.assertEqual(evidence.read_bytes(), b'original source')
        self.assertEqual(storage.inspect_publication(raised.exception.journal)['status'],
                         'failed_retained')

    def test_index_check_is_used_during_publication_without_commit_callback(self):
        original = dict(schema=region_library.SCHEMA, records={'old': {'hidden': True}})
        region_library.write_index(self.root, original)
        before = (self.root / 'region_library.json').read_bytes()
        cancelled = False
        receipt = storage._write_receipt

        def write_receipt(path, data):
            nonlocal cancelled
            receipt(path, data)
            if data['state'] == 'ready':
                cancelled = True

        def check():
            if cancelled:
                raise RuntimeError('cancel during index publication')

        with patch.object(storage, '_write_receipt', side_effect=write_receipt):
            with self.assertRaises(storage.PublicationError):
                region_library.write_index(
                    self.root, dict(schema=region_library.SCHEMA, records={}), check=check)
        self.assertEqual((self.root / 'region_library.json').read_bytes(), before)

    def test_concurrent_first_task_saves_return_the_same_single_binding(self):
        first = dict(task_context.new_task(), name='first task')
        second = dict(task_context.new_task(), name='second task')
        barrier = threading.Barrier(2)
        local = threading.local()
        checked = task_context.checked_task

        def synchronize_initial_validation(value):
            result = checked(value)
            # Both callers have observed the absent file before publication.
            # Reading the winner later must not enter this barrier again.
            if not getattr(local, 'validated_initial', False):
                local.validated_initial = True
                barrier.wait(timeout=5)
            return result

        with patch.object(task_context, 'checked_task', side_effect=synchronize_initial_validation):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(task_context.save_task, self.root, task)
                           for task in (first, second)]
                results = [future.result(timeout=10) for future in futures]
        self.assertEqual(results[0], results[1])
        self.assertIn(results[0], (first, second))
        self.assertEqual(task_context.read_task(self.root), results[0])
        journals = list((self.root / '.publications').glob('*.json'))
        self.assertEqual(len(journals), 1)
        self.assertEqual(storage.inspect_publication(journals[0])['status'], 'published')
        before = (self.root / 'task_context.json').read_bytes()
        self.assertEqual(task_context.save_task(self.root, task_context.new_task()), results[0])
        self.assertEqual((self.root / 'task_context.json').read_bytes(), before)

    def test_task_binding_survives_failure_to_write_final_receipt(self):
        task = dict(task_context.new_task(), name='saved binding')
        receipt = storage._write_receipt

        def fail_final_receipt(path, data):
            if data['state'] == 'published':
                raise OSError('receipt write failed')
            receipt(path, data)

        with patch.object(storage, '_write_receipt', side_effect=fail_final_receipt):
            with self.assertRaises(storage.PublicationError) as raised:
                task_context.save_task(self.root, task)
        self.assertTrue(raised.exception.published)
        self.assertEqual(task_context.read_task(self.root), task)
        self.assertEqual(task_context.save_task(self.root, task_context.new_task()), task)
        self.assertEqual(storage.recover_publication(raised.exception.journal)['status'], 'published')


if __name__ == '__main__':
    unittest.main()
