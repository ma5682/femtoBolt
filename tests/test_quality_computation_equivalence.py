"""Live quality and batched registration metrics retain prior evidence decisions."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import open3d as o3d

from femto.annotation import selection_intent as intent
from femto.local.local_pose_refinement import metrics


def legacy_metrics(source, target, transform):
    moved=copy.deepcopy(source).transform(transform)
    tree=o3d.geometry.KDTreeFlann(target)
    points=np.asarray(moved.points); colors=np.asarray(moved.colors)
    target_colors=np.asarray(target.colors); distances=[]; errors=[]
    for i,point in enumerate(points):
        count,ids,d=tree.search_knn_vector_3d(point,1)
        if count:
            distances.append(np.sqrt(d[0]))
            if d[0] <= .015**2 and len(colors)==len(points) and len(target_colors)==len(target.points):
                errors.append(float(np.mean(np.abs(colors[i]-target_colors[ids[0]]))))
    d=np.asarray(distances)
    return dict(median_m=float(np.median(d)) if len(d) else None,
                p90_m=float(np.percentile(d,90)) if len(d) else None,
                overlap=float(np.mean(d<=.015)) if len(d) else 0,
                color_error=float(np.median(errors)) if errors else None)


class QualityComputationTests(unittest.TestCase):
    def test_classes_equal_prior_raw_depth_classification_and_persisted_bytes(self):
        rng=np.random.default_rng(801)
        mask=rng.random((48,64))>.25
        reasons=rng.integers(0,5,mask.shape)
        depth=rng.choice([0,249,250,500,2210,2211,np.nan],mask.shape)
        image_valid=rng.random(mask.shape)>.1
        projection=SimpleNamespace(reasons=reasons,depth=depth,minimum=.25,maximum=2.21,
                                   image_valid=image_valid)
        present=(depth>=250)&(depth<=2210)
        expected=np.where(reasons==0,1,np.where(reasons==1,np.where(present,2,3),
                    np.where(reasons==2,4,np.where(reasons==3,5,6)))).astype(np.uint8)
        expected[~image_valid]=6; expected[~mask]=0
        before=[v.copy() for v in (mask,reasons,depth,image_valid)]
        actual=intent.classify_selection_support(mask,projection)
        np.testing.assert_array_equal(actual,expected)
        record=intent.build_selection_support(mask,projection)
        self.assertEqual(record['class_runs'],intent.encode_runs(expected))
        self.assertEqual(record['class_sha256'],intent.digest(expected))
        np.testing.assert_array_equal(intent.support_classes(record),expected)
        intent.validate_selection_support(record,mask,expected_depth_range_m=[.25,2.21])
        for value,original in zip((mask,reasons,depth,image_valid),before):
            np.testing.assert_array_equal(value,original)

    def test_fragmentation_and_depth_range_gates_still_apply(self):
        mask=np.ones((3,3),bool)
        projection=SimpleNamespace(reasons=np.arange(9).reshape(3,3)%5,
                                   depth=np.full((3,3),500),minimum=.25,maximum=2.21)
        with patch.object(intent,'MAX_RUNS',2):
            for action in (intent.classify_selection_support,intent.build_selection_support):
                with self.assertRaisesRegex(ValueError,'fragmented'):action(mask,projection)
        projection.minimum=-1
        with self.assertRaisesRegex(ValueError,'depth range'):
            intent.classify_selection_support(mask,projection)

    def test_registration_metrics_exact_for_colors_gaps_and_thresholds(self):
        rng=np.random.default_rng(29)
        target=o3d.geometry.PointCloud()
        target.points=o3d.utility.Vector3dVector(rng.uniform(-.1,.1,(350,3)))
        target.colors=o3d.utility.Vector3dVector(rng.random((350,3)))
        source=o3d.geometry.PointCloud(target)
        source.colors=o3d.utility.Vector3dVector(rng.random((350,3)))
        original=np.asarray(source.points).copy()
        for offset in (0,.002,.015,np.nextafter(.015,np.inf),.4):
            transform=np.eye(4);transform[0,3]=offset
            self.assertEqual(metrics(source,target,transform),legacy_metrics(source,target,transform))
        source.colors=o3d.utility.Vector3dVector([])
        self.assertEqual(metrics(source,target,np.eye(4)),legacy_metrics(source,target,np.eye(4)))
        np.testing.assert_array_equal(np.asarray(source.points),original)


if __name__=='__main__':unittest.main()
