"""History actions stay visible, and saved regions are not labelled as broken plans."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest

from femto.planning.export_history import _history_record_status
from femto.planning.trajectory_history import history_rows


def saved_region(root):
    folder = root / 'annotations' / 'saved'
    folder.mkdir(parents=True)
    (folder / 'area.glb').write_bytes(b'fixture')
    region = dict(annotation_id='r1', name='已保存区域', glb_file='area.glb',
                  region_manifest_file='region.json')
    (folder / 'region.json').write_text(json.dumps(dict(
        region=region, file_sha256={'area.glb': hashlib.sha256(b'fixture').hexdigest()})), encoding='utf-8')
    (folder / 'annotations.json').write_text(json.dumps(dict(regions=[region])), encoding='utf-8')
    return folder


class HistoryStatusTests(unittest.TestCase):
    def test_verified_region_without_plan_is_saved_not_legacy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = saved_region(root)
            rows = history_rows(root)
            row = rows[0]
            self.assertEqual(row['file_status'], '导出资产指纹一致')
            self.assertEqual(_history_record_status(row, rows), '区域已保存 · 尚无轨迹')
            self.assertNotIn('旧版本', row['status'])
            (folder / 'area.glb').write_bytes(b'tampered')
            damaged = history_rows(root)[0]
            self.assertEqual(damaged['file_status'], '损坏')
            self.assertIn('损坏', _history_record_status(damaged, [damaged]))

    def test_saved_plan_is_listed_without_guessing_an_annotation_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved_region(root)
            folder = root / 'region_paths' / 'saved_plan'
            folder.mkdir(parents=True)
            (folder / 'plan.glb').write_bytes(b'fixture')
            (folder / 'path.json').write_text(json.dumps(dict(
                region={'name': '已保存区域', 'annotation_id': 'r1'},
                visualization={'file': 'plan.glb'},
                plan_binding={'result_sha256': 'existing-result', 'region_id': 'r1'},
                files={'plan.glb': hashlib.sha256(b'fixture').hexdigest()})), encoding='utf-8')
            rows = history_rows(root)
            self.assertEqual(len(rows), 2)
            plan = next(row for row in rows if row['kind'] == '区域＋路径')
            self.assertNotIn('parent_manifest', plan)
            self.assertIn('路径关联待工作台重算', _history_record_status(plan, rows))

    def test_old_binding_and_damaged_children_remain_explicit(self):
        parent = dict(kind='区域', file_status='导出资产指纹一致', export_manifest=Path('region.json'),
                      details={}, status='区域已保存 · 区域资产核验通过')
        broken = dict(kind='三维表面轨迹', parent_manifest=Path('region.json'), file_status='损坏')
        partial = dict(kind='三维表面轨迹', parent_manifest=Path('region.json'),
                       file_status='轨迹资产指纹一致', result_complete=False)
        self.assertIn('关联轨迹损坏', _history_record_status(parent, [parent, broken]))
        self.assertIn('轨迹未完成', _history_record_status(parent, [parent, partial]))
        self.assertIn('损坏', _history_record_status(parent, [parent, partial, broken]))
        old = dict(parent, file_status='旧版本待验证', status='旧版本待验证；缺少原文件指纹')
        self.assertEqual(_history_record_status(old, [old]), old['status'])
        bound = dict(parent, details={'plan_binding': {'result_sha256': 'r'}},
                     status='导出资产指纹一致；路径关联待工作台重算')
        self.assertEqual(_history_record_status(bound, [bound]), bound['status'])
