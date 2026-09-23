"""Selection intent: the physician's pixel mask is authoritative, triangles are derived geometry."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from femto.annotation.fixed_rgb_geometry import FixedProjection
from femto.annotation.scene_annotations import AnnotationStore
from femto.annotation.selection_intent import (
    INTENT_SCHEMA, SELECTION_RULE, SUPPORT_SCHEMA, build_selection_intent, build_selection_support,
    component_labels, components_of_pixels, decode_runs, describe_selection, encode_runs, intent_mask,
    restrict_selection, selection_contours, selection_layers, support_classes, tint_selection,
    validate_selection_intent, validate_selection_support,
)
from tests.test_fixed_rgb_experience import K, POLYGON, make_model, quad, ring


def box(mask, u0, v0, u1, v1, value=True):
    mask[v0:v1 + 1, u0:u1 + 1] = value
    return mask


class SelectionIntentRecordTests(unittest.TestCase):
    def test_runs_round_trip_exactly_and_reject_corruption(self):
        mask = box(np.zeros((40, 60), bool), 5, 5, 29, 29)
        runs = encode_runs(mask)
        self.assertTrue(np.array_equal(decode_runs(runs, [60, 40], values=(0, 1)).astype(bool), mask))
        with self.assertRaises(ValueError):
            decode_runs(runs[:-2], [60, 40], values=(0, 1))
        with self.assertRaises(ValueError):
            decode_runs(runs, [61, 40], values=(0, 1))
        with self.assertRaises(ValueError):
            decode_runs([2, 2400], [60, 40], values=(0, 1))

    def test_components_holes_and_forbidden_are_enumerated_not_inferred(self):
        mask = box(np.zeros((40, 60), bool), 5, 5, 29, 29)
        box(mask, 12, 12, 19, 19, False)          # explicit unselected hole
        box(mask, 40, 5, 54, 29)                  # second, separate component
        box(mask, 2, 35, 5, 37)                   # tiny third component
        forbidden = box(np.zeros((40, 60), bool), 12, 12, 19, 19)
        intent = build_selection_intent(mask, forbidden_mask=forbidden, method="work_minus_forbidden")
        self.assertEqual(intent["schema"], INTENT_SCHEMA)
        self.assertEqual([c["component_id"] for c in intent["components"]], [0, 1, 2])
        self.assertEqual([c["pixel_count"] for c in intent["components"]], [625 - 64, 375, 12])
        self.assertEqual(intent["components"][0]["hole_ids"], [0])
        hole = intent["explicit_unselected_holes"][0]
        self.assertEqual((hole["component_id"], hole["pixel_count"], hole["painted_forbidden_pixel_count"]), (0, 64, 64))
        self.assertEqual(intent["missing_geometry_policy"], "retain_coverage_requirement_not_unselected")
        checked = validate_selection_intent(intent, [60, 40])
        self.assertTrue(np.array_equal(intent_mask(checked), mask))
        labels = component_labels(checked)
        self.assertEqual(components_of_pixels(labels, [[10.2, 10.4], [45, 10], [15, 15], [-1, 3]]).tolist(), [0, 1, -1, -1])
        # Diagonal touch is a separation, never a bridge.
        diagonal = box(box(np.zeros((10, 10), bool), 2, 2, 3, 3), 4, 4, 5, 5)
        self.assertEqual(len(build_selection_intent(diagonal, method="work_only")["components"]), 2)
        # Tampering with the mask, the counts or the topology is rejected.
        for tamper in (lambda v: v.__setitem__("selected_pixel_count", 1),
                       lambda v: v["components"][0].__setitem__("pixel_count", 1),
                       lambda v: v.__setitem__("explicit_unselected_holes", []),
                       lambda v: v.__setitem__("selected_mask_sha256", "0" * 64),
                       lambda v: v.__setitem__("missing_geometry_policy", "treat_as_unselected"),
                       lambda v: v.__setitem__("image_size", [61, 40])):
            broken = copy.deepcopy(intent)
            tamper(broken)
            with self.assertRaises(ValueError):
                validate_selection_intent(broken, [60, 40])

    def test_support_classes_distinguish_unselected_missing_depth_missing_mesh_and_conflict(self):
        mask = box(np.zeros((40, 60), bool), 5, 5, 29, 29)
        projection = SimpleNamespace(reasons=np.full((40, 60), 1, np.uint8), depth=np.full((40, 60), 500, np.uint16),
                                     minimum=.25, maximum=2.21)
        projection.reasons[5:20, 5:30] = 0          # measured mesh
        projection.reasons[25:30, 5:30] = 3         # depth conflict
        projection.depth[20:25] = 0                 # no reference depth at all
        support = build_selection_support(mask, projection)
        self.assertEqual(support["schema"], SUPPORT_SCHEMA)
        counts = support["counts"]
        self.assertEqual(counts["measured_mesh"], 15 * 25)
        self.assertEqual(counts["no_surface_no_reference_depth"], 5 * 25)
        self.assertEqual(counts["no_surface_reference_depth_present"], 0)
        self.assertEqual(counts["depth_conflict"], 5 * 25)
        self.assertEqual(support["geometry_unconfirmed_pixel_count"], 10 * 25)
        self.assertEqual(support["derived_surface"]["status"], "not_generated")
        classes = support_classes(validate_selection_support(support, mask))
        self.assertEqual(int((classes > 0).sum()), int(mask.sum()))
        self.assertTrue(np.all(classes[~mask] == 0))
        broken = copy.deepcopy(support)
        broken["counts"]["measured_mesh"] += 1
        with self.assertRaises(ValueError):
            validate_selection_support(broken, mask)
        with self.assertRaises(ValueError):
            validate_selection_support(support, box(mask.copy(), 0, 0, 2, 2))

    def test_restriction_only_removes_pixels_and_layers_render(self):
        mask = box(np.zeros((40, 60), bool), 5, 5, 29, 29)
        intent = build_selection_intent(mask, method="work_only")
        projection = SimpleNamespace(reasons=np.zeros((40, 60), np.uint8), depth=np.full((40, 60), 500, np.uint16),
                                     minimum=.25, maximum=2.21)
        region = dict(selection_intent=intent, selection_support=build_selection_support(mask, projection))
        removed = box(np.zeros((40, 60), bool), 10, 10, 12, 12)
        restricted = restrict_selection(region, [removed])
        self.assertEqual(restricted["selection_intent"]["selected_pixel_count"], 625 - 9)
        self.assertEqual(len(restricted["selection_intent"]["explicit_unselected_holes"]), 1)
        self.assertEqual(restricted["selection_support"]["selected_pixel_count"], 625 - 9)
        validate_selection_support(restricted["selection_support"], intent_mask(restricted["selection_intent"]))
        layers = selection_layers(restricted)
        self.assertEqual(int(layers["holes"].sum()), 9)
        contours = selection_contours(layers)
        self.assertEqual(len(contours["components"]), 1)
        self.assertEqual(len(contours["holes"]), 1)
        rgb = np.full((40, 60, 3), 120, np.uint8)
        tinted = tint_selection(rgb, layers)
        self.assertFalse(np.array_equal(tinted[11, 11], rgb[11, 11]))
        self.assertTrue(np.array_equal(tinted[7, 7], rgb[7, 7]))
        self.assertIsNone(selection_layers({"triangle_indices": [0]}))


class SelectionIntentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_region_created_from_mask_carries_intent_and_support_through_save_and_load(self):
        model = make_model(self.folder, ring())
        mask = box(np.zeros((100, 100), bool), 0, 0, 98, 98)
        box(mask, 45, 45, 55, 55, False)  # explicit hole over the ring hole (no mesh there anyway)
        selection = dict(mask=mask, forbidden_mask=None, method="work_only")
        item = model.region(POLYGON, "ring", "", selection=selection)
        self.assertEqual(item["selection_rule"], SELECTION_RULE)
        self.assertEqual(item["selection_intent"]["selected_pixel_count"], 99 * 99 - 121)
        self.assertEqual(len(item["selection_intent"]["explicit_unselected_holes"]), 1)
        support = item["selection_support"]
        self.assertEqual(support["selected_pixel_count"], 99 * 99 - 121)
        self.assertEqual(support["geometry_supported_pixel_count"] + support["geometry_unconfirmed_pixel_count"], 99 * 99 - 121)
        self.assertGreater(support["counts"]["no_surface_reference_depth_present"], 0)
        self.assertEqual(item["geometry_support"]["selection_geometry"], "selection_mask")
        self.assertEqual(sorted(item["triangle_indices"]), list(range(8)))
        path = model.store.save(self.folder / "saved")
        data = json.loads(path.read_text(encoding="utf-8"))
        saved = data["regions"][0]
        self.assertEqual(saved["selection_intent"], item["selection_intent"])
        manifest = json.loads((path.parent / saved["region_manifest_file"]).read_text(encoding="utf-8-sig"))
        self.assertEqual(manifest["region"]["selection_intent"]["selected_mask_sha256"], item["selection_intent"]["selected_mask_sha256"])
        self.assertEqual(manifest["region"]["selection_support"]["class_sha256"], support["class_sha256"])
        # Reload recomputes the derived diagnosis but never the intent; a tampered intent is refused.
        stale = copy.deepcopy(data)
        stale["regions"][0]["selection_support"]["counts"]["measured_mesh"] += 1
        stale_path = self.folder / "stale.json"
        stale_path.write_text(json.dumps(stale, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(ValueError):
            model.load(stale_path)
        model.load(path)
        self.assertEqual(model.store.regions[0]["selection_intent"], item["selection_intent"])
        self.assertEqual(model.store.regions[0]["selection_support"], support)
        tampered = copy.deepcopy(data)
        tampered["regions"][0]["selection_intent"]["selected_pixel_count"] -= 1
        tampered_path = self.folder / "tampered.json"
        tampered_path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(ValueError):
            model.load(tampered_path)
        legacy = copy.deepcopy(data)
        legacy["regions"][0].pop("selection_intent")
        legacy_path = self.folder / "legacy.json"
        legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "selection"):
            model.load(legacy_path)

    def test_mask_selection_matches_physician_pixels_not_polygon_simplification(self):
        model = make_model(self.folder, quad())
        # The quad's two triangles have centroids at (63, 37) and (37, 63); select only the right half.
        mask = box(np.zeros((100, 100), bool), 50, 0, 99, 99)
        item = model.region(POLYGON, "half", "", selection=dict(mask=mask, method="work_only"))
        self.assertEqual(item["triangle_indices"], [0])
        self.assertIsNone(model.region(POLYGON, "none", "", selection=dict(mask=box(np.zeros((100, 100), bool), 0, 0, 1, 1), method="work_only")))
        legacy = model.region(POLYGON, "polygon", "")
        self.assertNotIn("selection_intent", legacy)
        self.assertEqual(legacy["selection_rule"], "visible_triangle_centroid_in_fixed_reference_polygon")

    def test_legacy_exclusion_restricts_the_intent_mask(self):
        model = make_model(self.folder, quad())
        mask = box(np.zeros((100, 100), bool), 0, 0, 98, 98)
        model.region(POLYGON, "whole", "", selection=dict(mask=mask, method="work_only"))
        region = model.store.set_exclusions(0, [[[40, 40], [60, 40], [60, 60], [40, 60]]])
        self.assertLess(region["selection_intent"]["selected_pixel_count"], 99 * 99)
        self.assertEqual(len(region["selection_intent"]["explicit_unselected_holes"]), 1)
        self.assertEqual(region["selection_intent"]["painted_forbidden_pixel_count"],
                         99 * 99 - region["selection_intent"]["selected_pixel_count"])
        validate_selection_support(region["selection_support"], intent_mask(region["selection_intent"]))

    def test_region_project_round_trip_keeps_intent(self):
        from femto.planning.region_project import open_project, save_project
        root = self.folder
        source = root / "original.ply"
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector([[0, 0, .5], [.04, 0, .5], [.04, .04, .5], [0, .04, .5]]),
                                         o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]]))
        o3d.io.write_triangle_mesh(str(source), mesh)
        store = AnnotationStore(mesh, source)
        projection = FixedProjection(mesh, np.diag([500., 500., 1.]), np.full((50, 50), 500, np.uint16))
        view = {"reference_frame_id": 0, "session_id": "test", "rgb_sha256": "original-reference"}
        mask = box(np.zeros((50, 50), bool), 0, 0, 39, 39)
        intent = build_selection_intent(mask, method="work_only")
        support = build_selection_support(mask, projection)
        region = store.add_region(dict(triangle_indices=[0, 1], screen_polygon=[[0, 0], [40, 0], [40, 40], [0, 40]],
                                       view_reference=view, selection_rule=SELECTION_RULE,
                                       selection_intent=intent, selection_support=support))
        model = SimpleNamespace(store=store, projection=projection, rgb=np.full((50, 50, 3), 100, np.uint8),
                                view_reference=view, max_radius_mm=0., reconstruction=root / "reconstruction")
        bundle = save_project(model, region, root / "a.zip")
        with zipfile.ZipFile(bundle) as archive:
            project = json.loads(archive.read("project.json"))
        self.assertEqual(project["region"]["selection_intent"]["selected_mask_sha256"], intent["selected_mask_sha256"])
        loaded = open_project(bundle, root / "work")
        self.assertEqual(loaded.store.regions[0]["selection_intent"], intent)
        self.assertEqual(loaded.store.regions[0]["selection_support"], support)
        del loaded
        import gc
        gc.collect()


if __name__ == "__main__":
    unittest.main()
