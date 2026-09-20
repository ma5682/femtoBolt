import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock
from fixed_rgb_viewer import FixedRGBViewer
from tests.test_fixed_rgb_experience import make_model


class PlanningWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.folder=Path(self.tmp.name)
        self.root=tk.Tk();self.root.withdraw()

    def tearDown(self):
        self.root.destroy();self.tmp.cleanup()

    def test_managed_window_reused_and_history_restored(self):
        from planning_workspace import open_workspace
        history=tk.Toplevel(self.root)
        with mock.patch('fixed_rgb_viewer.FixedRGBViewer') as viewer:
            first=open_workspace(history,self.folder/'plan.zip',on_open=history.withdraw)
            deadline=time.monotonic()+3
            while not viewer.called and time.monotonic()<deadline:self.root.update();time.sleep(.02)
            self.assertTrue(viewer.called)
            second=open_workspace(history,self.folder/'plan.zip',on_open=history.withdraw)
            self.assertIs(first,second);self.assertEqual(viewer.call_count,1)
            self.assertEqual(history.state(),'withdrawn')
            first.destroy();self.root.update()
            self.assertEqual(history.state(),'normal')

    def test_tabs_save_existing_and_save_as_cancel(self):
        file=self.folder/'plan.zip';file.touch()
        model=make_model(self.folder/'source')
        region=model.region([[15,15],[85,15],[85,85],[15,85]],'Region','')
        model.project_id='test';model.saved_plan=None;model.store.mark_saved()
        window=tk.Toplevel(self.root)
        with mock.patch.object(FixedRGBViewer,'work'):
            viewer=FixedRGBViewer(window,file)
        viewer.loaded(model)
        self.assertEqual(len(viewer.workspace_tabs.tabs()),4)
        with mock.patch.object(viewer,'work') as work,mock.patch('fixed_rgb_viewer.filedialog.asksaveasfilename') as dialog:
            viewer.save_region_project();work.assert_called_once();dialog.assert_not_called()
        with mock.patch.object(viewer,'work') as work,mock.patch('fixed_rgb_viewer.filedialog.asksaveasfilename',return_value=''):
            viewer.save_region_project(save_as=True);work.assert_not_called()
        model.store.update_annotation('region',0,name='Edited',note='')
        viewer.refresh_items(('region',region['annotation_id']))
        viewer.project_saved(file,model.store.regions[0],None)
        self.assertFalse(model.store.dirty)
        viewer.close(force=True)

    def test_history_single_window_is_export_only(self):
        from export_history import show_history
        with mock.patch('export_history.scan_exports',return_value=[]):
            show_history(self.root,self.folder);first=self.root._export_history_window
            show_history(self.root,self.folder)
            self.assertIs(first,self.root._export_history_window)
            def texts(w):
                result=[]
                for child in w.winfo_children():
                    if 'text' in child.keys():result.append(str(child.cget('text')))
                    result.extend(texts(child))
                return result
            labels=texts(first)
            self.assertIn('预览区域模型',labels)
            self.assertIn('打开导出文件夹',labels)
            for hidden in ('打开规划工作台','打开工程继续规划','导入已有区域工程','打开工程 ZIP',
                           '移除所选索引（不删文件）','来源详情','扫描目录找工程'):
                self.assertNotIn(hidden,labels)
            # Finish the scheduled history scan before destroying the window.
            deadline=time.monotonic()+.3
            while time.monotonic()<deadline:self.root.update();time.sleep(.02)


if __name__=='__main__':unittest.main()
