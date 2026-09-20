import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from scene_gsplat_split import make_split, validate_split, file_sha256
from scene_gsplat_trainer import means_learning_rate
from scene_to_nerfstudio import convert
from synthetic_scene_demo import generate_dataset
from scene_gsplat_data import load_gsplat_dataset
from PIL import Image


class OptimizationTests(unittest.TestCase):
    def test_held_out_rgbd_changes_cannot_change_seed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = generate_dataset(root / 'source', frames=14, width=128, height=96)
            first = root / 'first'
            convert(source, first)
            dataset = load_gsplat_dataset(first)
            frames = [json.loads(line) for line in (source / 'frames.jsonl').read_text().splitlines()]
            for index in dataset.eval_indices + dataset.test_indices:
                Image.fromarray(np.full((96, 128, 3), 255, np.uint8)).save(source / frames[index]['rgb'])
                Image.fromarray(np.full((96, 128), 1500, np.uint16)).save(source / frames[index]['depth'])
            second = root / 'second'
            convert(source, second)
            self.assertEqual(file_sha256(first / 'rgbd_init_points.ply'), file_sha256(second / 'rgbd_init_points.ply'))
            self.assertNotEqual(dataset.fingerprint, load_gsplat_dataset(second).fingerprint)
            payload = json.loads((second / 'transforms.json').read_text())
            payload['dataset_split']['seed_frame_indices'].append(dataset.test_indices[0])
            (second / 'transforms.json').write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_gsplat_dataset(second)

    def test_strict_partitions_are_disjoint_and_test_is_excluded(self):
        split = make_split(90)
        validate_split(split, 90)
        self.assertEqual([len(split[key]) for key in ('train', 'validation', 'test')], [68, 11, 11])
        split['seed_frame_indices'] = split['train'] + split['test']
        with self.assertRaises(ValueError):
            validate_split(split, 90)

    def test_split_rejects_duplicates(self):
        split = make_split(14)
        split['validation'].append(split['test'][0])
        with self.assertRaises(ValueError):
            validate_split(split, 14)

    def test_lr_schedule_reconstructs_exactly_on_resume(self):
        whole = [means_learning_rate(2.0, step, 6000, 0.01) for step in range(6001)]
        resumed = [means_learning_rate(2.0, step, 6000, 0.01) for step in range(250, 6001)]
        self.assertEqual(whole[250:], resumed)
        self.assertAlmostEqual(whole[-1] / whole[0], 0.01)
        self.assertGreater(whole[250], whole[500])
        self.assertEqual(means_learning_rate(2.0, 6000, 6000, 1.0), whole[0])


if __name__ == '__main__':
    unittest.main()
