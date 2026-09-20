from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from complete_demo import _classify_data_kind
from scene_gsplat_acceptance import input_provenance
from scene_scale_validation import _lineage


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class DataKindClassificationTests(unittest.TestCase):
    def test_complete_demo_only_accepts_canonical_femto_device(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw_capture"
            dataset = root / "posed_capture"
            _write_json(raw / "manifest.json", {"camera": {"device": "Generic RGB-D Camera"}})
            _write_json(
                dataset / "manifest.json",
                {"camera": {"device": "Orbbec Femto Bolt"}, "source_dataset": str(raw)},
            )
            self.assertEqual(_classify_data_kind(dataset), "unknown")

            _write_json(raw / "manifest.json", {"camera": {"device": "Orbbec Femto Bolt"}})
            self.assertEqual(_classify_data_kind(dataset), "real_femto")

            _write_json(dataset / "manifest.json", {"source_dataset": str(root / "missing")})
            self.assertEqual(_classify_data_kind(dataset), "unknown")

    def test_acceptance_keeps_non_femto_camera_unknown(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "capture"
            nerfstudio = root / "nerfstudio"
            _write_json(dataset / "manifest.json", {"camera": {"device": "Other Depth Camera"}})
            _write_json(nerfstudio / "conversion_manifest.json", {"source_dataset": str(dataset)})
            self.assertEqual(input_provenance(nerfstudio)["data_kind"], "unknown")

    def test_scale_ignores_unverified_real_femto_summary_claim(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "capture"
            reconstruction = root / "reconstruction"
            _write_json(dataset / "manifest.json", {"camera": {"device": "Other Depth Camera"}})
            _write_json(
                reconstruction / "reconstruction_report.json",
                {"source_dataset": str(dataset)},
            )
            _write_json(root / "demo_summary.json", {"data_kind": "real_femto"})
            self.assertEqual(_lineage(reconstruction)["data_kind"], "unknown")

    def test_scale_accepts_source_manifest_femto_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw_capture"
            dataset = root / "posed_capture"
            reconstruction = root / "reconstruction"
            _write_json(raw / "manifest.json", {"camera": {"device": "Orbbec Femto Bolt"}})
            _write_json(dataset / "manifest.json", {"source_dataset": str(raw)})
            _write_json(
                reconstruction / "reconstruction_report.json",
                {"source_dataset": str(dataset)},
            )
            _write_json(root / "demo_summary.json", {"data_kind": "unknown"})
            self.assertEqual(_lineage(reconstruction)["data_kind"], "real_femto")

    def test_synthetic_marker_wins_over_camera_label(self) -> None:
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "capture"
            _write_json(
                dataset / "manifest.json",
                {
                    "status": "synthetic_generated",
                    "camera": {"device": "Orbbec Femto Bolt"},
                },
            )
            self.assertEqual(_classify_data_kind(dataset), "synthetic")


if __name__ == "__main__":
    unittest.main()
