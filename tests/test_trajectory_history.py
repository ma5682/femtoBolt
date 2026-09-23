import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_mesh_slice_planner import make_export, pixel_mask
from femto.planning.mesh_slice_planner import plan_mesh_slice
from femto.planning.trajectory_export import export_trajectory
from femto.planning.trajectory_history import generate, previous_trajectories, remove_previous
from femto.planning.planning_io import TaskControl


class TrajectoryHistoryTests(unittest.TestCase):
    def setup_export(self, folder):
        # The product export carries the physician's selection intent (one component over the mesh).
        manifest=make_export(folder,selection=dict(mask=pixel_mask((0,0,199,199))))
        data=json.loads(manifest.read_text())
        annotations=Path(folder)/'annotations.json'
        annotations.write_text(json.dumps({'regions':[dict(data['region'],region_manifest_file=manifest.name)]}))
        row=dict(kind='区域',source=annotations,region_index=0)
        trajectory=plan_mesh_slice(manifest)
        old=export_trajectory(trajectory,folder,include_visuals=False)
        return manifest,row,old

    def test_regenerate_verifies_new_result_then_removes_only_bound_old_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,row,old=self.setup_export(tmp)
            other=Path(tmp)/'trajectory_0123456789ab';other.mkdir()
            (other/'trajectory.json').write_text(json.dumps(dict(schema='femto-surface-trajectory-v1',
                source=dict(region_manifest=str(Path(tmp)/'other_region.json')))))
            before=manifest.read_bytes()
            result=generate(row,TaskControl())
            self.assertFalse(old.exists())
            self.assertTrue(other.exists())
            self.assertTrue((result['folder']/'trajectory_files.json').is_file())
            self.assertEqual(previous_trajectories(manifest),[result['folder']])
            self.assertEqual(manifest.read_bytes(),before)
            self.assertEqual(len(result['removed']),1)

    def test_failed_or_cancelled_generation_preserves_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,row,old=self.setup_export(tmp)
            before=(old/'trajectory.json').read_bytes()
            control=TaskControl();control.set()
            with self.assertRaises(RuntimeError):generate(row,control)
            self.assertEqual((old/'trajectory.json').read_bytes(),before)
            with patch('femto.planning.trajectory_export.load_trajectory',side_effect=ValueError('bad output')):
                with self.assertRaises(ValueError):generate(row,TaskControl())
            self.assertEqual((old/'trajectory.json').read_bytes(),before)

    def test_cleanup_refuses_unrelated_files_and_outside_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest,_,old=self.setup_export(tmp)
            (old/'user-note.txt').write_text('keep')
            removed,warnings=remove_previous(manifest,[old,Path(tmp)],Path(tmp)/'new')
            self.assertEqual(removed,[]);self.assertTrue(warnings)
            self.assertTrue((old/'user-note.txt').exists())


if __name__=='__main__':unittest.main()
