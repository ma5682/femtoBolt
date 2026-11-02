import json
from pathlib import Path
import tempfile
import unittest

from femto import data_backup
from femto.pipeline.data_maintenance import backup_workspace,inspect_references


class MaintenanceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.outputs=self.root/'outputs';self.outputs.mkdir()
        self.raw=self.root/'datasets/scene_test_local_test';self.raw.mkdir(parents=True)
        (self.raw/'manifest.json').write_text(json.dumps(dict(schema='femto-dataset-v1')))
        (self.raw/'rgb.bin').write_bytes(b'raw-data')
        self.rec=self.outputs/'complete_demo_test/reconstruction';self.rec.mkdir(parents=True)
        self.report=self.rec/'reconstruction_report.json'
        self.report.write_text(json.dumps(dict(source_dataset=str(self.raw))))

    def test_published_capture_chain_and_associated_assets_are_bound(self):
        self.assertIn(str(self.raw),inspect_references(self.root)['references'])
        destination=self.root/'backup'
        result=backup_workspace(self.root,destination)
        self.assertTrue(result['manifest']['reference_audit']['bindings'])
        data_backup.verify(destination)
        self.assertEqual((destination/'contents/datasets/scene_test_local_test/rgb.bin').read_bytes(),b'raw-data')

    def test_new_location_restore_does_not_hide_unavailable_absolute_sources(self):
        destination=self.root/'backup';backup_workspace(self.root,destination)
        original_report=self.report.read_bytes()
        preserved=self.root/'preserved_originals';preserved.mkdir()
        for name in ('outputs','datasets'):
            source=self.root/name;target=preserved/name
            self.assertTrue(source.resolve().is_relative_to(self.root))
            self.assertTrue(target.resolve().is_relative_to(self.root))
            source.rename(target)
        migrated=self.root/'migrated';migrated.mkdir()
        targets={name:migrated/name for name in ('outputs','datasets')}
        restored=data_backup.restore_new(destination,targets,workspace=self.root)
        self.assertFalse(restored['original_locations']);self.assertFalse(restored['paths_rebased'])
        restored_report=migrated/'outputs/complete_demo_test/reconstruction/reconstruction_report.json'
        self.assertEqual(restored_report.read_bytes(),original_report)
        audit=inspect_references(migrated)
        self.assertIn(str(self.raw),audit['missing'])
        # Original-path recovery restores the existing absolute binding unchanged.
        data_backup.restore_new(destination,{name:self.root/name for name in targets},workspace=self.root)
        self.assertEqual(inspect_references(self.root)['missing'],[])

    def test_missing_raw_capture_is_not_reported_as_complete(self):
        self.report.write_text(json.dumps(dict(source_dataset=str(self.root/'missing_dataset'))))
        with self.assertRaisesRegex(FileNotFoundError,'关联资产'):
            backup_workspace(self.root,self.root/'backup')
        self.assertFalse((self.root/'backup').exists())

    def test_explicit_missing_source_archive_remains_labeled_incomplete(self):
        self.report.write_text(json.dumps(dict(source_dataset=str(self.root/'missing_dataset'))))
        result=backup_workspace(self.root,self.root/'backup',allow_existing_missing=True)
        audit=result['manifest']['reference_audit']
        self.assertFalse(audit['source_assets_complete'])
        self.assertEqual(audit['unavailable'][0]['reason'],'missing_before_backup')
        restored=data_backup.restore_new(self.root/'backup',dict(outputs=self.root/'restored_outputs',datasets=self.root/'restored_datasets'),workspace=self.root)
        self.assertFalse(restored['source_assets_complete'])
        self.assertEqual(restored['unavailable_reference_count'],1)

    def test_external_registered_project_requires_explicit_root(self):
        external=self.root/'external';external.mkdir();project=external/'plan.zip';project.write_bytes(b'portable')
        registry=self.outputs/'saved_projects/region_project_history';registry.mkdir(parents=True)
        (registry/'entry.json').write_text(json.dumps(dict(project_file=str(project))))
        with self.assertRaisesRegex(ValueError,'数据根'):
            backup_workspace(self.root,self.root/'backup')
        result=backup_workspace(self.root,self.root/'backup',roots=dict(outputs=self.outputs,datasets=self.root/'datasets',external=external))
        self.assertIn('plan.zip',result['manifest']['roots']['external']['files'])

    def test_passive_portable_lineage_is_not_an_active_missing_dependency(self):
        registry=self.outputs/'saved_projects';registry.mkdir()
        (registry/'record.json').write_text(json.dumps(dict(source_lineage=dict(source_mesh=str(self.root/'deleted_old_source.ply')))))
        self.assertNotIn(str(self.root/'deleted_old_source.ply'),inspect_references(self.root)['references'])
        backup_workspace(self.root,self.root/'backup')

    def test_file_cannot_silently_be_used_as_an_empty_directory_root(self):
        with self.assertRaisesRegex(ValueError,'directories'):
            data_backup.backup(dict(raw=self.raw/'rgb.bin'),self.root/'backup',workspace=self.root)

    def test_workspace_relative_capture_path_uses_reader_binding(self):
        self.report.write_text(json.dumps(dict(source_dataset='datasets/scene_test_local_test')))
        report=inspect_references(self.root)
        self.assertIn(str(self.raw),report['references'])
        self.assertEqual(report['missing'],[])

    def test_real_publication_receipt_resolves_owner_target_not_absent_stage(self):
        from femto.asset_storage import publish_json
        registry=self.outputs/'saved_projects/region_project_history';registry.mkdir(parents=True)
        portable=self.outputs/'retained.zip';portable.write_bytes(b'retained portable')
        target=registry/'record.json'
        publish_json(target,dict(project_file=str(portable)))
        report=inspect_references(self.root)
        self.assertIn(str(target),report['references'])
        self.assertIn(str(portable),report['references'])
        self.assertEqual(report['missing'],[])
        self.assertEqual(report['publication_receipts'][0]['status'],'published')
        backup_workspace(self.root,self.root/'backup')
