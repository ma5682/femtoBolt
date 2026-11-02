import json
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import sys

from femto.app import qt_main
from femto.data_access import DataLease


class MaintenanceEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for target,value in [('REPO_ROOT',self.root),('QApplication',Mock()),('apply_theme',Mock()),('_logs',Mock())]:
            modifier=patch.object(qt_main,target,value);modifier.start();self.addCleanup(modifier.stop)
        modifier=patch.object(sys,'argv',['workstation','--ui-preview']);modifier.start();self.addCleanup(modifier.stop)
        modifier=patch.object(sys,'excepthook',sys.excepthook);modifier.start();self.addCleanup(modifier.stop)
        qt_main.QApplication.return_value.exec.return_value=0

    def test_gate_held_through_event_loop_and_released_after_exit(self):
        def construct(**kwargs):
            with self.assertRaises(RuntimeError):DataLease(self.root,maintenance=True).acquire()
            return Mock()
        with patch.object(qt_main,'WorkstationWindow',side_effect=construct):
            self.assertEqual(qt_main.main(),0)
        with DataLease(self.root,maintenance=True):pass

    def test_busy_maintenance_does_not_construct_workstation(self):
        with DataLease(self.root,maintenance=True),patch.object(qt_main,'WorkstationWindow') as window:
            with patch.object(qt_main.QMessageBox,'warning') as warning:
                self.assertEqual(qt_main.main(),2)
                warning.assert_called_once();window.assert_not_called()

    def test_pending_restore_blocks_start_and_releases_gate(self):
        folder=self.root/'.data_restores';folder.mkdir()
        (folder/'pending.json').write_text(json.dumps(dict(state='preparing')))
        with patch.object(qt_main,'WorkstationWindow') as window,patch.object(qt_main.QMessageBox,'warning'):
            self.assertEqual(qt_main.main(),2);window.assert_not_called()
        with DataLease(self.root,maintenance=True):pass
