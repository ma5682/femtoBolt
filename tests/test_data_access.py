from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from femto.data_access import DataLease


class DataLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_workstations_share_but_maintenance_is_exclusive(self):
        with DataLease(self.root),DataLease(self.root):
            with self.assertRaisesRegex(RuntimeError,'关闭'):DataLease(self.root,maintenance=True).acquire()
        with DataLease(self.root,maintenance=True):
            with self.assertRaisesRegex(RuntimeError,'维护'):DataLease(self.root).acquire()

    def test_process_exit_releases_maintenance_without_deleting_lock_file(self):
        script='from femto.data_access import DataLease;import sys;lease=DataLease(sys.argv[1],maintenance=True).acquire()'
        subprocess.run([sys.executable,'-c',script,str(self.root)],check=True,capture_output=True)
        self.assertTrue((self.root/'.data_maintenance.lock').is_file())
        with DataLease(self.root):pass

    def test_exception_releases_gate(self):
        with self.assertRaises(ValueError):
            with DataLease(self.root,maintenance=True):raise ValueError('failure')
        with DataLease(self.root,maintenance=True):pass
