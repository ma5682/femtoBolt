"""Comparison ancestry and progress formatting without a GUI dependency."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from femto.io_utils import load_json_object
from femto.ui_i18n import pick


def comparison_source(result_or_dataset: str | Path) -> Path:
    """Resolve recorded ancestry without inferring a new source."""
    root = Path(result_or_dataset).resolve()
    if root.name == "reconstruction":
        root = root.parent
    expected = None
    report_path = root / "reconstruction" / "reconstruction_report.json"
    if report_path.is_file():
        publication = root / "publication.json"
        if publication.is_file() and load_json_object(publication).get("status") != "published":
            raise ValueError("This result was not published; choose a completed result or its original posed dataset")
        context = load_json_object(root / "reconstruction" / "local_geometry.json")
        expected = (context.get("session_id"), context.get("reference_frame_id"))
        root = Path(load_json_object(report_path)["source_dataset"]).resolve()
    seen: set[Path] = set()
    for _ in range(8):
        if root in seen:
            raise ValueError("Cyclic comparison source ancestry")
        seen.add(root)
        manifest = load_json_object(root / "manifest.json")
        local = manifest.get("local_capture", {})
        identity = (local.get("session_id"), local.get("reference_frame_id"))
        if not local.get("measured_only") or not identity[0] or type(identity[1]) is not int:
            raise ValueError("Local session/reference evidence is missing")
        if expected is None:
            expected = identity
        if identity != expected:
            raise ValueError("Comparison source crosses a session or initial reference")
        if local.get("temporal_support_report"):
            root = Path(load_json_object(local["temporal_support_report"])["source_dataset"]).resolve()
        elif manifest.get("refinement_source_dataset"):
            root = Path(manifest["refinement_source_dataset"]).resolve()
        elif manifest.get("pose", {}).get("source") == "online_rgbd_research_grade":
            return root
        else:
            raise ValueError("Original online local poses are unavailable; no fallback origin is assumed")
    raise ValueError("Comparison source ancestry exceeds the eight-level budget")


def progress_text(report: Any, language: str = "zh") -> str:
    labels = {
        "running": pick(language, "执行中", "Running"),
        "complete": pick(language, "已完成", "Complete"),
        "failed": pick(language, "失败", "Failed"),
        "cancelled": pick(language, "已取消", "Cancelled"),
        "adopted": pick(language, "已采用", "Adopted"),
        "not_adopted": pick(language, "未采用", "Not adopted"),
        "baseline": pick(language, "原始基线", "Original baseline"),
    }
    lines = [pick(language, "对照状态：", "Comparison state: ") + labels.get(report.get("state"), str(report.get("state", "—")))]
    if report.get("session_id"):
        lines.append(pick(language, "会话：", "Session: ") + report["session_id"])
    if report.get("source_frame_count") is not None:
        lines.append(pick(language, "原始帧 / 接纳视角：", "Source frames / accepted views: ") +
                     f"{report['source_frame_count']} / {len(report.get('accepted_frame_ids', []))}")
    operations = {
        "validate": pick(language, "来源与门禁检查", "Source and gate checks"),
        "refine": pick(language, "精配准", "Refinement"),
        "temporal": pick(language, "五帧二值筛选", "Five-frame binary filter"),
        "reconstruct": pick(language, "几何重建与固定视图", "Reconstruction and fixed projection"),
    }
    for stage in report.get("stages", []):
        label = operations.get(stage.get("operation"), stage.get("operation", "—"))
        line = f"{stage.get('name', '')}: {label} — {labels.get(stage.get('state'), stage.get('state', '—'))}"
        if stage.get("elapsed_s") is not None:
            line += f" · {stage['elapsed_s']:.1f} s"
        lines.append(line)
    for variant in report.get("variants", []):
        status = variant.get("optimization", {}).get("status")
        line = variant["name"] + ": " + labels.get(status, str(status))
        if variant.get("same_as"):
            line += pick(language, "；几何与 ", "; geometry identical to ") + variant["same_as"]
        lines.append(line)
    if report.get("error"):
        lines.append(pick(language, "原因：", "Reason: ") + str(report["error"]))
    if report.get("state") != "running":
        for key, title in (("source_unchanged", pick(language, "原始文件哈希保持", "Source hashes unchanged")),
                           ("latest_unchanged", pick(language, "业务最新索引保持", "Latest business marker unchanged"))):
            value = report.get(key)
            lines.append(title + ": " + (pick(language, "未记录", "Not recorded") if value is None else str(value)))
    return "\n".join(lines)


__all__ = ["comparison_source", "progress_text"]
