import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from femto import asset_storage as storage
from femto.io_utils import sha256_file, write_json_strict


class AssetStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_location_is_relocatable_and_rejects_escape(self):
        (self.root/'nested').mkdir()
        (self.root/'nested/data.bin').write_bytes(b'evidence')
        assets = storage.LocalAssets({'data': self.root})
        location = assets.describe(self.root/'nested/data.bin')
        self.assertEqual(location['relative_path'], 'nested/data.bin')
        self.assertEqual(assets.locate(location), self.root/'nested/data.bin')
        for name in ('../escape', r'..\escape', r'C:\escape', '/escape', r'\\host\share\file', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                storage.resolve_relative(self.root, name)

    def test_success_and_own_receipt_are_idempotently_recoverable(self):
        target = storage.publish_json(self.root/'index.json', {'name': '测试'})
        journal = next((self.root/'.publications').glob('*.json'))
        before = target.read_bytes()
        self.assertEqual(storage.inspect_publication(journal)['status'], 'published')
        self.assertEqual(storage.recover_publication(journal)['status'], 'published')
        self.assertEqual(target.read_bytes(), before)

    def test_failure_before_replace_preserves_old_metadata(self):
        target = self.root/'index.json'
        target.write_bytes(b'old')
        replace = storage.os.replace
        def fail_stage(source, destination):
            if Path(source).suffix == '.stage':
                raise OSError('simulated disk failure')
            return replace(source, destination)
        with patch.object(storage.os, 'replace', side_effect=fail_stage):
            with self.assertRaises(storage.PublicationError) as context:
                storage.publish_json(target, {'name': 'new'})
        self.assertFalse(context.exception.published)
        self.assertEqual(target.read_bytes(), b'old')
        self.assertEqual(storage.inspect_publication(context.exception.journal)['status'], 'failed_retained')
        with self.assertRaises(ValueError):
            storage.recover_publication(context.exception.journal)

    def test_failure_after_replace_can_finalize_receipt_without_overwrite(self):
        writer = storage._write_receipt
        def fail_receipt(path, data):
            if data['state'] == 'published':
                raise OSError('receipt disk failure')
            writer(path, data)
        with patch.object(storage, '_write_receipt', side_effect=fail_receipt):
            with self.assertRaises(storage.PublicationError) as context:
                storage.publish_json(self.root/'index.json', {'name': 'new'})
        error = context.exception
        self.assertTrue(error.published)
        self.assertEqual(storage.inspect_publication(error.journal)['status'], 'published_unrecorded')
        self.assertEqual(storage.recover_publication(error.journal)['status'], 'published')
        self.assertEqual(json.loads((self.root/'index.json').read_text(encoding='utf-8')), {'name': 'new'})

    def test_crash_ready_receipt_resumes_only_if_target_unchanged(self):
        target = self.root/'index.json'
        target.write_bytes(b'old')
        folder = self.root/'.publications'; folder.mkdir()
        token = 'a'*32
        stage = folder/(token+'.stage'); stage.write_bytes(b'new')
        journal = folder/(token+'.json')
        data = dict(schema='femto-metadata-publication-v1', operation_id=token, state='ready',
                    target=target.name, stage=stage.name, before_sha256=sha256_file(target),
                    sha256=sha256_file(stage))
        write_json_strict(journal, data)
        target.write_bytes(b'external change')
        self.assertEqual(storage.inspect_publication(journal)['status'], 'conflict')
        with self.assertRaises(ValueError):
            storage.recover_publication(journal)
        self.assertEqual(target.read_bytes(), b'external change')
        target.write_bytes(b'old')
        self.assertEqual(storage.recover_publication(journal)['status'], 'published')
        self.assertEqual(target.read_bytes(), b'new')

    def test_cancel_before_replace_cannot_be_resumed_as_success(self):
        def cancel():
            raise RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            storage.publish_json(self.root/'index.json', {}, cancel)
        self.assertFalse((self.root/'index.json').exists())

    def test_cancel_after_ready_receipt_retains_old_target_and_cannot_resume(self):
        target = self.root/'index.json'
        target.write_bytes(b'old')
        cancelled = []
        writer = storage._write_receipt
        def receipt(path, data):
            writer(path, data)
            if data['state'] == 'ready':
                cancelled.append(True)
        def check():
            if cancelled:
                raise RuntimeError('cancelled after preparation')
        with patch.object(storage, '_write_receipt', side_effect=receipt):
            with self.assertRaises(storage.PublicationError) as context:
                storage.publish_json(target, {'payload': 1}, check)
        self.assertEqual(target.read_bytes(), b'old')
        self.assertFalse(context.exception.published)
        self.assertEqual(storage.inspect_publication(context.exception.journal)['status'], 'failed_retained')
        with self.assertRaises(ValueError):
            storage.recover_publication(context.exception.journal)

    def test_recorded_failure_with_identical_old_and_new_bytes_cannot_resume(self):
        payload = {'name': 'unchanged'}
        target = self.root / 'index.json'
        write_json_strict(target, payload)
        original = target.read_bytes()
        cancelled = False
        writer = storage._write_receipt

        def receipt(path, data):
            nonlocal cancelled
            writer(path, data)
            if data['state'] == 'ready':
                cancelled = True

        def check():
            if cancelled:
                raise RuntimeError('cancel identical metadata before replacement')

        with patch.object(storage, '_write_receipt', side_effect=receipt):
            with self.assertRaises(storage.PublicationError) as raised:
                storage.publish_json(target, payload, check)
        error = raised.exception
        self.assertFalse(error.published)
        recorded = json.loads(error.journal.read_text(encoding='utf-8'))
        self.assertEqual(recorded['state'], 'failed')
        self.assertEqual(recorded['sha256'], sha256_file(target))
        journal_before = error.journal.read_bytes()
        self.assertEqual(storage.inspect_publication(error.journal)['status'], 'failed_retained')
        with self.assertRaises(ValueError):
            storage.recover_publication(error.journal)
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(error.journal.read_bytes(), journal_before)

    def _prepared_receipt(self, state, *, identical=False):
        target = self.root / 'index.json'
        target.write_bytes(b'original')
        folder = self.root / '.publications'
        folder.mkdir(exist_ok=True)
        token = 'b' * 32
        stage = folder / (token + '.stage')
        stage.write_bytes(target.read_bytes() if identical else b'new metadata')
        journal = folder / (token + '.json')
        data = dict(schema='femto-metadata-publication-v1', operation_id=token, state=state,
                    target=target.name, stage=stage.name, before_sha256=sha256_file(target),
                    sha256=sha256_file(stage))
        write_json_strict(journal, data)
        return target, stage, journal

    def test_preparing_receipt_with_matching_content_never_authorizes_recovery(self):
        target, stage, journal = self._prepared_receipt('preparing', identical=True)
        before = {path: path.read_bytes() for path in (target, stage, journal)}
        try:
            inspected = storage.inspect_publication(journal)
        except ValueError:
            pass  # Rejecting this inconsistent preparation receipt is also safe.
        else:
            self.assertNotIn(inspected['status'], ('published', 'published_unrecorded', 'recoverable'))
        with self.assertRaises(ValueError):
            storage.recover_publication(journal)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_unknown_or_missing_receipt_state_is_rejected_before_recovery(self):
        for state in ('unknown', None, [], 12, 'missing'):
            with self.subTest(state=state):
                target, stage, journal = self._prepared_receipt(state, identical=True)
                if state == 'missing':
                    data = json.loads(journal.read_text(encoding='utf-8'))
                    del data['state']
                    write_json_strict(journal, data)
                before = {path: path.read_bytes() for path in (target, stage, journal)}
                with self.assertRaises(ValueError):
                    storage.inspect_publication(journal)
                with self.assertRaises(ValueError):
                    storage.recover_publication(journal)
                self.assertEqual({path: path.read_bytes() for path in before}, before)
                self.assertFalse((self.root / '.publication.lock').exists())

    def test_ready_receipt_after_actual_replace_can_finalize_without_rewriting_target(self):
        target, stage, journal = self._prepared_receipt('ready')
        storage.os.replace(stage, target)
        before = target.read_bytes()
        self.assertEqual(storage.inspect_publication(journal)['status'], 'published_unrecorded')
        self.assertEqual(storage.recover_publication(journal)['status'], 'published')
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(stage.exists())

    def test_process_crash_releases_lock_and_ready_publication_recovers(self):
        target = self.root/'index.json'
        target.write_bytes(b'old')
        code = """
import os,sys
from femto import asset_storage as storage
write=storage._write_receipt
def crash(path,data):
    write(path,data)
    if data['state']=='ready':
        os._exit(23)
storage._write_receipt=crash
storage.publish_json(sys.argv[1],{'payload':1})
"""
        result = subprocess.run([sys.executable, '-c', code, str(target)],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(target.read_bytes(), b'old')
        journal = next((self.root/'.publications').glob('*.json'))
        self.assertEqual(storage.inspect_publication(journal)['status'], 'recoverable')
        storage.recover_publication(journal)
        self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'payload': 1})


if __name__ == '__main__':
    unittest.main()
