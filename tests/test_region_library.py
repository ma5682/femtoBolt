import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from femto.planning.region_library import digest, rebuild, edit_records, relocate, read_index
from femto.planning.export_history import scan_exports, register_project

class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
    def project(self,name='a.zip',revision='v1',result='result1',rid='region1'):
        path=self.root/name
        assets={n:b'fixture' for n in ['scene_tsdf_mesh_m.ply','rgb.png','depth.npy','image_valid.npy']}
        import hashlib
        data=dict(schema='femto-region-project-v1',project_id='p',revision_id=revision,
            region=dict(annotation_id=rid,name='Area'),geometry_sha256='geometry',origin_reference={'origin':'W0'},
            files={n:hashlib.sha256(v).hexdigest() for n,v in assets.items()},
            plan=dict(plan_binding=dict(result_sha256=result,region_id=rid)))
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('project.json',json.dumps(data))
            for n,v in assets.items():z.writestr(n,v)
        register_project(self.root,path);return path
    def test_group_versions_and_source_isolation(self):
        self.project();self.project('b.zip','v2','result2');self.project('c.zip',rid='other')
        rows=scan_exports(self.root)
        self.assertEqual(len({r['region_key'] for r in rows}),2)
        a,b=[r for r in rows if r['details']['region_id']=='region1']
        self.assertNotEqual(a['version_key'],b['version_key'])
        self.assertIn('待工作台重算',a['status'])
    def test_relocate_same_content_not_name(self):
        p=self.project();row=scan_exports(self.root)[0];moved=self.root/'moved.zip';p.rename(moved)
        self.assertEqual(scan_exports(self.root)[0]['file_status'],'文件丢失')
        p.write_bytes(b'wrong same name')
        with self.assertRaises(ValueError):relocate(self.root,row,p)
        relocate(self.root,row,moved)
        self.assertEqual(scan_exports(self.root)[0]['project'],moved)
    def test_old_without_digest_refuses_relink(self):
        p=self.project();index=next((self.root/'region_project_history').glob('*.json'))
        data=json.loads(index.read_text());data.pop('sha256');index.write_text(json.dumps(data))
        row=scan_exports(self.root)[0]
        with self.assertRaises(ValueError):relocate(self.root,row,p)
    def test_atomic_hide_restore_scope_and_shared_assets(self):
        self.project();self.project('b.zip');rows=rebuild(self.root)
        before={p:digest(p) for p in self.root.glob('*.zip')}
        edit_records(self.root,rows,hidden=True,note='note')
        self.assertEqual(scan_exports(self.root),[])
        edit_records(self.root,rows,hidden=False)
        self.assertTrue(all(r['note']=='note' for r in scan_exports(self.root)))
        with self.assertRaises(ValueError):edit_records(self.root,rows+[{'key':'outside'}],hidden=True)
        old=read_index(self.root)
        with patch('femto.planning.region_library.os.replace',side_effect=OSError('disk')):
            with self.assertRaises(OSError):edit_records(self.root,rows,hidden=True)
        self.assertEqual(read_index(self.root),old)
        self.assertEqual(before,{p:digest(p) for p in before})
    def test_rebuild_without_index_and_cancel(self):
        self.project();rebuild(self.root);(self.root/'region_library.json').unlink()
        self.assertEqual(len(rebuild(self.root)),1)
        before=read_index(self.root)
        def cancel():raise RuntimeError('cancelled')
        with self.assertRaises(RuntimeError):rebuild(self.root,check=cancel)
        self.assertEqual(read_index(self.root),before)
    def test_export_move_keeps_record_and_relinks(self):
        folder=self.root/'region_paths'/'old';folder.mkdir(parents=True)
        source=folder/'path.json'
        source.write_text(json.dumps(dict(region_id='r',segments=[],geometry_sha256='g',origin_reference={'id':'a'})))
        row=rebuild(self.root)[0]
        moved=self.root/'moved';folder.rename(moved)
        self.assertEqual(scan_exports(self.root)[0]['file_status'],'文件丢失')
        relocate(self.root,row,moved/'path.json')
        self.assertEqual(scan_exports(self.root)[0]['folder'],moved)
    def test_discovery_without_project_index(self):
        from femto.planning.region_library import discover_projects
        self.project()
        for p in (self.root/'region_project_history').glob('*.json'):p.unlink()
        self.assertEqual(scan_exports(self.root),[])
        self.assertEqual(len(discover_projects(self.root,self.root)['found']),1)
        self.assertEqual(len(scan_exports(self.root)),1)
    def test_corrupt_index_repair_preserves_bytes(self):
        from femto.planning.region_library import repair_index
        self.project();index=self.root/'region_library.json';index.write_bytes(b'{broken')
        with self.assertRaises(ValueError):scan_exports(self.root)
        backup=repair_index(self.root)
        self.assertEqual(backup.read_bytes(),b'{broken')
        self.assertEqual(len(rebuild(self.root)),1)
    def test_project_internal_damage_detected_after_fresh_import(self):
        p=self.project()
        with zipfile.ZipFile(p) as z:entries={n:z.read(n) for n in z.namelist()}
        entries['rgb.png']=b'tampered'
        with zipfile.ZipFile(p,'w') as z:
            for n,v in entries.items():z.writestr(n,v)
        register_project(self.root,p)
        self.assertEqual(scan_exports(self.root)[0]['file_status'],'损坏')
    def test_new_revision_does_not_inherit_old_note(self):
        self.project();edit_records(self.root,scan_exports(self.root),note='old revision')
        self.project(revision='new')
        self.assertEqual(scan_exports(self.root)[0]['note'],'')
    def test_gui_reuse_filters_and_selection(self):
        import tkinter as tk
        from tkinter import ttk
        import time
        from femto.planning.export_history import show_history
        self.project()
        exported=self.root/'annotations'/'exported';exported.mkdir(parents=True)
        (exported/'area.glb').write_bytes(b'fixture')
        (exported/'annotations.json').write_text(json.dumps({'regions':[{
            'name':'Exported area','annotation_id':'r1','glb_file':'area.glb'}]}),encoding='utf-8')
        root=tk.Tk();root.withdraw()
        try:
            window=show_history(root,self.root)
            self.assertIs(show_history(root,self.root),window)
            def descendants(w):
                result=list(w.winfo_children())
                for c in list(result):result.extend(descendants(c))
                return result
            tree=next(w for w in descendants(window) if isinstance(w,ttk.Treeview))
            deadline=time.monotonic()+10
            while not tree.get_children() and time.monotonic()<deadline:root.update();time.sleep(.01)
            self.assertEqual(len(tree.get_children()),1)
            entry=next(w for w in descendants(window) if isinstance(w,ttk.Entry) and not isinstance(w,ttk.Combobox))
            entry.insert(0,'unmatched');root.update();self.assertEqual(tree.get_children(),())
            entry.delete(0,'end');root.update();self.assertEqual(len(tree.get_children()),1)
            tree.selection_set('0');root.update()
        finally:root.destroy()
    def test_route_export_uses_own_manifest_not_annotation_dependency(self):
        self.project();project=scan_exports(self.root)[0]
        folder=self.root/'region_paths'/'route';folder.mkdir(parents=True)
        data=dict(region=dict(annotation_id='region1',region_manifest_file='original_annotation.json'),
            geometry_sha256='geometry',origin_reference={'origin':'W0'},plan_binding={'result_sha256':'result1'})
        (folder/'path.json').write_text(json.dumps(data))
        row=next(r for r in scan_exports(self.root) if r['kind']=='区域＋路径')
        self.assertEqual(row['region_key'],project['region_key'])
        self.assertEqual(row['version_key'],project['version_key'])
        self.assertNotEqual(row['file_status'],'损坏')
    def test_corrupt_project_and_export_missing(self):
        p=self.project();p.write_bytes(b'corrupt')
        self.assertEqual(scan_exports(self.root)[0]['file_status'],'损坏')
        folder=self.root/'annotations'/'old';folder.mkdir(parents=True)
        (folder/'annotations.json').write_text(json.dumps({'regions':[{'name':'none'},{'name':'lost','glb_file':'missing.glb'}]}))
        states={r['name']:r['file_status'] for r in scan_exports(self.root)}
        self.assertEqual(states['none'],'无模型导出');self.assertEqual(states['lost'],'文件丢失')

if __name__=='__main__':unittest.main()
