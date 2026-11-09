"""Fast lossless archives retain standard NumPy entries and cancellation."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
import zipfile
import numpy as np
from femto.planning import target_surface as target
from femto.planning.region_project import _pack_assets

class FastArchiveTests(unittest.TestCase):
    def test_npz_members_equal_numpy_writer_including_empty_and_fortran(self):
        arrays={'empty':np.empty((0,3,3),dtype='<f8'),'flags':np.array([True,False]),
                'values':np.arange(105,dtype='<f8').reshape(7,15)[:,::2],
                'fortran':np.asfortranarray(np.arange(30,dtype='<i8').reshape(5,6)),
                'special':np.array([np.nan,np.inf,-np.inf,-0.,np.nextafter(0.,1.)])}
        with TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old.npz';np.savez_compressed(old,**{k:arrays[k] for k in sorted(arrays)})
            target.write(root,SimpleNamespace(arrays=arrays,metadata={'schema':'test'}))
            with zipfile.ZipFile(old) as a,zipfile.ZipFile(root/target.ASSET) as b:
                self.assertEqual(a.namelist(),b.namelist())
                for name in a.namelist():self.assertEqual(a.read(name),b.read(name))
            with np.load(root/target.ASSET,allow_pickle=False) as loaded:
                for k,v in arrays.items():
                    self.assertEqual(loaded[k].dtype,v.dtype);self.assertEqual(loaded[k].shape,v.shape)
                    self.assertEqual(loaded[k].tobytes(),v.tobytes())

    def test_cancel_between_arrays_never_publishes_manifest(self):
        with TemporaryDirectory() as temp:
            calls=[]
            def check():
                calls.append(1)
                if len(calls)==3:raise RuntimeError('cancel requested')
            with self.assertRaisesRegex(RuntimeError,'cancel requested'):
                target.write(Path(temp),SimpleNamespace(arrays={'a':np.arange(4),'b':np.arange(5)},metadata={}),check)
            self.assertFalse((Path(temp)/target.MANIFEST).exists())

    def test_portable_members_are_exact_and_pack_checks_cancellation(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);values={'data.npy':bytes(range(256))*400,'target_surface.npz':b'opaque archive bytes','project.json':b'{"schema":"test"}'}
            for name,value in values.items():(root/name).write_bytes(value)
            calls=[];result=_pack_assets(root,lambda:calls.append(1))
            with zipfile.ZipFile(result) as z:
                self.assertEqual(set(z.namelist()),set(values))
                for name,value in values.items():self.assertEqual(z.read(name),value)
            self.assertTrue(calls)

if __name__=='__main__':unittest.main()
