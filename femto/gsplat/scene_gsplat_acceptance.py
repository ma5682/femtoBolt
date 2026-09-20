"""Assemble the real-data acceptance report for a trained direct-gsplat run.

Nothing here re-computes a metric.  Every number is copied from the artefact that
already owns it — the trainer's evaluation, the Nerfstudio conversion manifest,
the 3DGS input quality report and the pose quality report — so the acceptance
summary can never disagree with the gates that produced it.  Missing inputs are
reported as ``null`` with the reason, never silently defaulted to a passing value.

Acceptance thresholds and the depth definition (unrendered holes scored as a 0 m
prediction, coverage reported separately) stay in ``scene_gsplat_trainer``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from femto.io_utils import load_json, write_json
from femto.dataset.scene_data_kind import classify_dataset_data_kind, load_dataset_manifest_chain


from femto.paths import REPO_ROOT as ROOT
OUTPUTS_ROOT = ROOT / "outputs"
LATEST_MARKER = OUTPUTS_ROOT / "latest_direct_gsplat.txt"
REPORT_SCHEMA = "femto-direct-gsplat-acceptance-v1"


def _maybe_json(path: Path) -> dict[str, Any] | None:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_run(value: Path | None) -> Path:
    if value is None:
        if not LATEST_MARKER.is_file():
            raise ValueError("No trained direct gsplat run is available")
        value = Path(LATEST_MARKER.read_text(encoding="utf-8-sig").strip())
    run = Path(value).resolve()
    if not (run / "training_report.json").is_file():
        raise ValueError(f"Not a direct gsplat result directory: {run}")
    return run


def input_provenance(nerfstudio_dir: Path) -> dict[str, Any]:
    """Follow existing dataset lineage; a conversion manifest alone is NOT real evidence."""
    conversion = _maybe_json(nerfstudio_dir / "conversion_manifest.json") or {}
    demo = _maybe_json(nerfstudio_dir.parent / "demo_summary.json") or {}
    source = conversion.get("source_dataset")
    loaded_chain, lineage_complete = load_dataset_manifest_chain(Path(source) if source else None)
    chain = [{"path": str(path), "manifest": manifest} for path, manifest in loaded_chain]
    raw = chain[-1] if chain else {}
    manifests = [item["manifest"] for item in chain]
    source_kind = classify_dataset_data_kind(Path(source) if source else None)
    synthetic = demo.get("mode") == "synthetic" or source_kind == "synthetic"
    camera = raw.get("manifest", {}).get("camera", {})
    kind = "synthetic" if synthetic else source_kind
    fixed = bool(demo.get("fixed_camera")) or any(
        bool(m.get("parameters", {}).get("fixed_camera")) for m in manifests
    )
    reconstruction = _maybe_json(nerfstudio_dir.parent / "reconstruction" / "reconstruction_report.json")
    shared_pose = None
    if reconstruction and conversion.get("source_dataset") and reconstruction.get("source_dataset"):
        shared_pose = Path(conversion["source_dataset"]).resolve() == Path(reconstruction["source_dataset"]).resolve()
    return {
        "data_kind": kind,
        "raw_dataset": raw.get("path"),
        "source_dataset": conversion.get("source_dataset"),
        "raw_frame_count": raw.get("manifest", {}).get("frame_count"),
        "valid_view_count": conversion.get("frame_count"),
        "fixed_camera": fixed,
        "camera": camera,
        "dataset_chain": [item["path"] for item in chain],
        "dataset_lineage_complete": lineage_complete,
        "pose_source": manifests[0].get("pose", {}).get("source") if manifests else None,
        "geometry_and_gaussian_share_pose_dataset": shared_pose,
        "pose_note": "Research/demo RGB-D odometry only; drift, no loop closure, relocalization or global pose graph optimization.",
    }


def _pose_summary(nerfstudio_dir: Path, conversion: dict[str, Any] | None) -> dict[str, Any]:
    """Describe where the poses came from and how well they scored.

    Sequential RGB-D odometry is explicitly labelled research/demo grade: it has
    no loop closure or relocalization and must not be presented as a production
    pose source.
    """
    summary: dict[str, Any] = {
        "translation_span_m": (conversion or {}).get("translation_span_m"),
        "rotation_span_deg": (conversion or {}).get("rotation_span_deg"),
        "posed_dataset": (conversion or {}).get("source_dataset"),
        "pose_grade": "unknown",
        "pose_source": None,
    }
    demo_summary = _maybe_json(nerfstudio_dir.parent / "demo_summary.json")
    if demo_summary:
        summary["pose_source"] = demo_summary.get("pose_stage")
        summary["fixed_camera"] = demo_summary.get("fixed_camera")
        summary["raw_dataset"] = demo_summary.get("raw_dataset")
    posed_dataset = summary["posed_dataset"]
    if posed_dataset:
        quality = _maybe_json(Path(posed_dataset) / "pose_quality_report.json")
        if quality:
            summary["pose_grade"] = "research_demo_sequential_rgbd_odometry"
            summary["pose_quality"] = {
                "status": quality.get("status"),
                "source_frame_count": quality.get("source_frame_count"),
                "accepted_frame_count": quality.get("frame_count"),
                "rejected_frame_count": quality.get("rejected_frame_count"),
                "gates": quality.get("parameters"),
                "limitations": quality.get("limitations"),
            }
            pairs = quality.get("pairs") or []
            fitness = [item.get("fitness") for item in pairs if isinstance(item.get("fitness"), (int, float))]
            rmse = [item.get("inlier_rmse_m") for item in pairs if isinstance(item.get("inlier_rmse_m"), (int, float))]
            if fitness:
                summary["pose_quality"]["min_fitness"] = float(min(fitness))
            if rmse:
                summary["pose_quality"]["max_inlier_rmse_m"] = float(max(rmse))
    if summary["pose_grade"] == "unknown" and summary.get("pose_source") == "existing_valid_pose":
        summary["pose_grade"] = "dataset_supplied_pose"
    summary["production_pose_note"] = (
        "Research/demo pose only. Sequential RGB-D odometry has cumulative drift, no loop closure, "
        "no independently validated relocalization and no global pose graph optimization; not production pose."
    )
    return summary


def build_acceptance_report(run: Path) -> dict[str, Any]:
    training = load_json(run / "training_report.json")
    evaluation = training.get("evaluation") or {}
    aggregate = evaluation.get("aggregate") or {}
    checks = evaluation.get("checks") or {}
    thresholds = evaluation.get("thresholds") or {}
    configuration = _maybe_json(run / "training_config.json") or {}

    dataset_root = training.get("dataset_root")
    nerfstudio_dir = Path(dataset_root) if dataset_root else run
    conversion = _maybe_json(nerfstudio_dir / "conversion_manifest.json")
    input_quality = _maybe_json(nerfstudio_dir / "quality" / "3dgs_input_report.json")
    depth_valid = (input_quality or {}).get("depth_valid_ratio") or {}
    provenance = input_provenance(nerfstudio_dir)
    source_dataset = provenance.get("source_dataset")
    validation = _maybe_json(Path(source_dataset) / "quality" / "validation_report.json") if source_dataset else None
    reconstruction = _maybe_json(nerfstudio_dir.parent / "reconstruction" / "reconstruction_report.json")

    peak_bytes = training.get("peak_cuda_memory_bytes")
    report = {
        "schema": REPORT_SCHEMA,
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "run": str(run),
        "backend": "direct nerfstudio-project/gsplat public API (Apache-2.0)",
        "profile": configuration.get("profile"),
        "iterations": training.get("iterations"),
        "data_kind": provenance["data_kind"],
        "provenance": provenance,
        "runtime": training.get("runtime"),
        "training_command": training.get("training_command"),
        "dataset_validation": validation,
        "reconstruction": reconstruction,
        "input": {
            "nerfstudio_dir": str(nerfstudio_dir),
            "dataset_fingerprint_sha256": training.get("dataset_fingerprint_sha256"),
            "accepted_views": (conversion or {}).get("frame_count"),
            "train_views": training.get("train_frames"),
            "held_out_views": training.get("eval_frames") or evaluation.get("held_out_frames"),
            "seed_points": ((conversion or {}).get("initialization") or {}).get("point_count"),
            "input_depth_valid_ratio": {
                "median": depth_valid.get("median"),
                "mean": depth_valid.get("mean", (validation or {}).get("metrics", {}).get("valid_depth_ratio_mean")),
                "min": depth_valid.get("min"),
                "max": depth_valid.get("max"),
            },
            "input_quality_warnings": (input_quality or {}).get("warnings"),
            "photometric_unstable": (input_quality or {}).get("photometric_unstable"),
        },
        "pose": _pose_summary(nerfstudio_dir, conversion),
        "metrics": {
            "psnr": aggregate.get("psnr"),
            "ssim": aggregate.get("ssim"),
            "lpips": aggregate.get("lpips"),
            "depth_rmse_m": aggregate.get("depth_rmse_m"),
            "depth_mae_m": aggregate.get("depth_mae_m"),
            "depth_abs_rel": aggregate.get("depth_abs_rel"),
            "depth_coverage_ratio": aggregate.get("depth_coverage_ratio"),
        },
        "thresholds": thresholds,
        "checks": checks,
        "depth_definition": evaluation.get("depth_definition"),
        "lpips_error": evaluation.get("lpips_error"),
        "lpips_backbone": evaluation.get("lpips_backbone"),
        "model": {
            "initial_gaussians": training.get("initial_gaussians"),
            "final_gaussians": training.get("final_gaussians"),
            "training_elapsed_s": training.get("elapsed_s"),
            "peak_cuda_memory_bytes": peak_bytes,
            "peak_cuda_memory_mib": round(peak_bytes / (1024 * 1024), 1) if peak_bytes else None,
            "model_ply": training.get("model_ply"),
            "model_spz": training.get("model_spz"),
            "spz_error": training.get("spz_error"),
        },
        "accepted": bool(training.get("accepted", False)),
        "real_femto_rendering_accepted": provenance["data_kind"] == "real_femto" and bool(training.get("accepted", False)),
        "scope_disclaimer": (
            "Rendering-quality acceptance of the reconstruction pipeline. Not a clinical or "
            "robot-positioning claim; metric scale is validated separately on the geometry layer."
        ),
    }
    scale = _maybe_json(nerfstudio_dir.parent / "reconstruction" / "scale_validation.json")
    report["scale_validation"] = (
        {
            "report": str(nerfstudio_dir.parent / "reconstruction" / "scale_validation.json"),
            "geometry_source": scale.get("geometry_source"),
            "summary": scale.get("summary"),
        }
        if scale
        else {"available": False, "reason": "run scene_scale_validation.py on this reconstruction"}
    )
    # Do not inherit an unrelated or synthetic scale report as real acceptance.
    if scale:
        report["scale_validation"]["measurements"] = scale.get("measurements")
        report["scale_validation"]["scope_disclaimer"] = scale.get("scope_disclaimer")
        report["scale_validation"]["dataset_matches"] = (
            scale.get("lineage", {}).get("source_dataset") == provenance["source_dataset"]
        )
    report["end_to_end_status"] = (
        "requires_scale_review" if report["real_femto_rendering_accepted"]
        and report["scale_validation"].get("dataset_matches") else "BLOCKED"
    )
    report["scale_acceptance_note"] = "No engineering tolerance was supplied; numeric scale error is not automatically a PASS."
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize direct gsplat acceptance for a real capture")
    parser.add_argument("run", nargs="?", type=Path, help="direct_gsplat_* directory; defaults to the latest")
    parser.add_argument("--output", type=Path, help="Report path; defaults into the run directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run = resolve_run(args.run)
        report = build_acceptance_report(run)
        output = args.output or (run / "acceptance_report.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        report["report_path"] = str(output)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "blocked", "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
