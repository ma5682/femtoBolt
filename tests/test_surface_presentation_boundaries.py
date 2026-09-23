"""Display-only regressions: selection components are never joined by legacy outlines."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

import numpy as np
from PIL import ImageDraw

from femto.annotation.selection_intent import build_selection_intent
from femto.planning.path_presentation import _block_state_label, render_diagram


class SurfacePresentationBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((120, 240, 3), 100, np.uint8)
        self.ids = np.zeros((120, 240), int)
        mask = np.zeros((120, 240), bool)
        mask[20:81, 20:81] = True
        mask[20:81, 160:221] = True
        # Like a brush export, this single polygon traces each component and then
        # traverses the same between-component bridge in both directions.
        self.region = {
            "triangle_indices": [0],
            "screen_polygon": [[20, 20], [80, 20], [80, 80], [20, 80], [20, 20],
                               [160, 20], [220, 20], [220, 80], [160, 80], [160, 20], [20, 20]],
            "selection_intent": build_selection_intent(mask, method="work_only"),
        }
        self.plan = {
            "row_step_px": 10,
            "segments": [
                [{"pixel": [30, 60], "xyz_m": [0, 0, .5]},
                 {"pixel": [60, 60], "xyz_m": [.03, 0, .5]}],
                [{"pixel": [170, 60], "xyz_m": [.14, 0, .5]},
                 {"pixel": [200, 60], "xyz_m": [.17, 0, .5]}],
            ],
            "blocks": {
                "selection_blocks": True,
                "items": [{"label": "块 1", "chain_indices": [0], "status": "planned"},
                          {"label": "块 2", "chain_indices": [1], "status": "planned"}],
                "summary": {"block_count": 2, "incomplete_block_count": 0,
                            "unplannable_block_count": 0, "coverage_status": "complete",
                            "uncovered_fragment_count": 0},
            },
        }

    def draw(self, **kwargs):
        calls = []
        original = ImageDraw.ImageDraw.line

        def line(draw, xy, fill=None, width=0, joint=None):
            calls.append((np.asarray(xy).copy(), fill))
            return original(draw, xy, fill=fill, width=width, joint=joint)

        with mock.patch.object(ImageDraw.ImageDraw, "line", line):
            image = render_diagram(self.rgb, self.ids, self.region, self.plan, **kwargs)
        return np.asarray(image), calls

    def test_selection_outlines_remain_but_legacy_bridge_is_not_drawn(self):
        before = copy.deepcopy((self.region, self.plan))
        image, calls = self.draw(show_path=False)
        self.assertNotIn("#ee80ff", [color for _, color in calls])
        outlines = [points for points, color in calls if color == "#38e6d8"]
        self.assertEqual(len(outlines), 2)
        for points in outlines:
            np.testing.assert_array_equal(points[0], points[-1])
            self.assertLessEqual(np.ptp(points[:, 0]), 60)
        header = image.shape[0] - self.rgb.shape[0]
        np.testing.assert_array_equal(image[header + 20, 120], [100, 100, 100])
        np.testing.assert_array_equal(image[header + 80, 40], [56, 230, 216])
        self.assertEqual((self.region, self.plan), before)

    def test_single_block_filter_draws_no_between_block_or_hidden_route(self):
        _, calls = self.draw(chain_indices=[1])
        self.assertNotIn("#ee80ff", [color for _, color in calls])
        routes = [points for points, color in calls if color == "#ffd52b"]
        self.assertEqual(len(routes), 1)
        np.testing.assert_array_equal(routes[0][:, 0], [170, 200])
        self.assertEqual(sum(color == "#38e6d8" for _, color in calls), 2)
        for points, _ in calls:
            self.assertLessEqual(np.ptp(points[:, 0]), 60)

    def test_legacy_plan_retains_legacy_polygon_even_with_new_region_intent(self):
        self.plan.pop("blocks")
        _, calls = self.draw(show_path=False)
        old_outlines = [points for points, color in calls if color == "#ee80ff"]
        self.assertEqual(len(old_outlines), 1)
        self.assertGreater(np.ptp(old_outlines[0][:, 0]), 60)
        self.assertNotIn("#38e6d8", [color for _, color in calls])

    def test_selection_plan_does_not_fall_back_to_bridge_if_layers_absent(self):
        self.region.pop("selection_intent")
        _, calls = self.draw(show_path=False)
        self.assertNotIn("#ee80ff", [color for _, color in calls])

    def test_selection_coverage_does_not_fill_legacy_polygon_without_geometry(self):
        self.ids.fill(-1)
        self.plan["coverage"] = {"triangle_indices": [], "covered_samples_per_face": [],
                                 "sampled_coverage_percent": 0.0}
        image, _ = self.draw(show_path=False, show_coverage=True)
        header = image.shape[0] - self.rgb.shape[0]
        np.testing.assert_array_equal(image[header + 70, 60], [24, 24, 24])

    def test_incomplete_label_distinguishes_chains_from_breaks(self):
        cases = [
            ({"chain_count": 4, "internal_gap_count": 3}, "未完成：4 段路线 / 3 处断开"),
            ({"chain_count": 4, "internal_gap_count": 2}, "未完成：4 段路线 / 2 处断开"),
            ({"chain_count": 2}, "未完成：2 段路线 / 1 处断开"),
            ({"chain_count": 0}, "未完成：0 段路线 / 0 处断开"),
            ({}, "未完成：2 段路线 / 1 处断开"),
        ]
        for continuity, expected in cases:
            with self.subTest(continuity=continuity):
                item = {"status": "incomplete", "continuity": continuity, "chain_indices": [0, 1]}
                before = copy.deepcopy(item)
                self.assertEqual(_block_state_label(item, True), expected)
                self.assertEqual(item, before)

    def test_other_status_labels_keep_existing_meaning(self):
        self.assertEqual(_block_state_label({"status": "planned"}, True), "单链完成")
        self.assertEqual(_block_state_label({"status": "planned"}, False), "已规划")
        self.assertEqual(_block_state_label({"status": "unplannable", "reason": "no_rows"}, True),
                         "未规划：no_rows")


if __name__ == "__main__":
    unittest.main()
