"""Persist and replay every P3 trial, including rejected coverage and the v3 base."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh

from femto.planning import coverage_archive as archive, coverage_refinement as refinement
from femto.planning.coverage_workflow import bundle
from femto.planning.surface_trajectory import SurfaceTrajectory

PREFIXES = tuple(f'refinement_{i}_' for i in range(3))
ASSETS = {prefix+name for prefix in PREFIXES for name in {archive.MANIFEST, archive.ASSET} | archive.HINT_ASSETS}
_ISSUER = object()


@dataclass(frozen=True)
class VerifiedAssets:
    """In-memory receipt for one fully verified native publication, never serialized."""
    binding_sha256: str
    issuer: object


def _asset_binding(trajectory, ledger, certificate):
    from femto.planning.refinement_validation import semantic_payload
    from femto.planning.planning_revision import digest
    if trajectory.planner['algorithm_version'] == 'nominal-variable-width-surface-v1':
        if type(certificate) is not refinement.coverage.Certificate:raise ValueError('变宽模型完整覆盖证据缺失')
        history = []
    else:
        if not isinstance(certificate, refinement.RefinedCertificate):raise ValueError('补充规划完整证据缺失')
        history = certificate.history
    return digest(dict(route=semantic_payload(trajectory), target=ledger.metadata, coverage=certificate.metadata,
                       history=[c.metadata for c in history]))


def issue_verified_assets(trajectory, ledger, certificate):
    """Only called after native target/proof readback, replay and publication succeed."""
    return VerifiedAssets(_asset_binding(trajectory, ledger, certificate), _ISSUER)


def require_verified_assets(receipt, trajectory, ledger, certificate, check=lambda: None):
    if (not isinstance(receipt, VerifiedAssets) or receipt.issuer is not _ISSUER
            or receipt.binding_sha256 != _asset_binding(trajectory, ledger, certificate)):
        raise ValueError('本次验证凭据与目标、路线或补充证据不一致')
    # Hash the actual arrays again. A receipt cannot bless a mutated cache,
    # even if its metadata was left unchanged since the native publication.
    for value in (certificate, *getattr(certificate, 'history', ())):
        refinement.coverage._checked_assets(value, check)
        if value.metadata['algorithm'] in refinement.coverage.RECALCULATION_ALGORITHMS:
            from femto.planning.coverage_recalculation import require_guide
            require_guide(value, check)
    from femto.planning.planning_revision import array_record
    if (set(ledger.arrays) != set(ledger.metadata['arrays'])
            or any(array_record(a) != ledger.metadata['arrays'].get(k) for k,a in ledger.arrays.items())):
        raise ValueError('本次验证后的目标数组已改变')


def history_metadata(trajectory) -> list[dict]:
    if trajectory.planner['algorithm_version'] not in (*refinement.ALGORITHMS,'nominal-variable-width-refined-surface-v1'):return []
    report = trajectory.gap_report['refinement'];records = [report['base_coverage']]
    for item in report['rounds']:
        if item['certificate_index'] is not None:
            if item['certificate_index'] != len(records):raise ValueError('补充覆盖资产索引不连续')
            records.append(item['coverage'])
    if len(records) > len(PREFIXES):raise ValueError('补充覆盖资产数量超过预算')
    return records


def check_inventory(names: set[str], trajectory, coverage_metadata=None) -> None:
    history = history_metadata(trajectory)
    expected = set().union(*(archive.asset_names(metadata, prefix=PREFIXES[i]) for i, metadata in enumerate(history)))
    present = {name for name in names if name.startswith('refinement_') and name.endswith((archive.MANIFEST, archive.ASSET))}
    if present != expected:raise ValueError('补充规划证据资产缺失、孤立或混用了其他轮次')
    archive.check_hint_inventory(names, coverage_metadata)
    for index, metadata in enumerate(history):archive.check_hint_inventory(names, metadata, prefix=PREFIXES[index])


def local_mesh(ledger):
    faces = ledger.arrays['faces'][ledger.arrays['region_faces']]
    ids, remap = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(vertices=ledger.arrays['vertices'][ids], faces=remap.reshape(-1, 3), process=False)


def replay(certificate, trajectory, ledger, check=lambda: None):
    from femto.planning.refinement_validation import semantic_payload
    if not isinstance(certificate, refinement.RefinedCertificate):raise ValueError('补充规划证据资产缺失')
    records = history_metadata(trajectory)
    if len(records) != len(certificate.history):raise ValueError('补充规划历史证据数量不一致')
    for expected, value in zip(records, certificate.history):
        refinement.coverage._checked_assets(value, check)
        if expected != value.metadata:raise ValueError('补充覆盖证据与轮次记录不一致')
    refinement.coverage._checked_assets(certificate, check)
    base = SurfaceTrajectory.from_dict(trajectory.gap_report['refinement']['base_trajectory'])
    base_ledger = refinement.rebind_target(ledger, base.planning_context['revision'])
    if base_ledger.metadata != base.planning_context['target']:raise ValueError('补充规划原目标被改变')
    domain, seeds, model, route = bundle(base, base_ledger)
    refinement.coverage.validate(certificate.history[0], domain, seeds, model, route_binding=route, check=check)
    if trajectory.planner['algorithm_version']=='nominal-variable-width-refined-surface-v1':
        from femto.planning.variable_width_refinement import refine as reproduce_refinement
    else:reproduce_refinement=refinement.refine
    rebuilt, target, fresh = reproduce_refinement(base, base_ledger, certificate.history[0], local_mesh(ledger),
        base.planning_context['revision']['inputs']['selection'],
        budget=refinement.RefinementBudget(**trajectory.gap_report['refinement']['budget']), check=check,
        frontiers=[c.metadata['metrics'] for c in certificate.history[1:]],
        candidate_policy=trajectory.gap_report['refinement'].get('candidate_policy'),
        coverage_policy=trajectory.gap_report['refinement'].get('coverage_policy'))
    if (semantic_payload(rebuilt) != semantic_payload(trajectory) or target.metadata != ledger.metadata
            or fresh.metadata != certificate.metadata or len(fresh.history) != len(certificate.history)
            or any(a.metadata != b.metadata for a,b in zip(fresh.history, certificate.history))):
        raise ValueError('补充候选、逐边验收、覆盖增益或停止原因无法完整重现')
    return certificate


def read(folder, metadata, trajectory, ledger, check=lambda: None):
    check_inventory({p.name for p in Path(folder).iterdir() if p.is_file()}, trajectory, metadata)
    final = archive.load_assets(folder, check)
    if final.metadata != metadata:raise ValueError('补充规划最终覆盖清单不一致')
    values = tuple(archive.load_assets(folder, check, prefix=PREFIXES[i]) for i in range(len(history_metadata(trajectory))))
    return replay(refinement.RefinedCertificate(final.metadata, final.arrays, values), trajectory, ledger, check)


def write(folder, certificate, trajectory, ledger, check=lambda: None, *, verified_assets=None):
    from femto.planning.planning_io import durable
    if not isinstance(certificate, refinement.RefinedCertificate):raise ValueError('补充规划证据资产缺失，不能只保存摘要')
    if len(certificate.history) != len(history_metadata(trajectory)):raise ValueError('补充历史与路线记录不一致')
    if verified_assets is not None:require_verified_assets(verified_assets, trajectory, ledger, certificate, check)
    archive.write_assets(folder, certificate, check)
    for index, value in enumerate(certificate.history):archive.write_assets(folder, value, check, prefix=PREFIXES[index])
    if verified_assets is None:
        read(folder, certificate.metadata, trajectory, ledger, check)
    else:
        # The same immutable numerical content was replayed in this save job.
        # Re-read every new file with the bounded reader and compare all hashes;
        # standalone opens still take the full replay path above.
        check_inventory({p.name for p in Path(folder).iterdir() if p.is_file()}, trajectory, certificate.metadata)
        for prefix, expected in [('', certificate)]+list(zip(PREFIXES, certificate.history)):
            actual=archive.load_assets(folder, check, prefix=prefix)
            if actual.metadata != expected.metadata:raise ValueError('已验证补充证据的写回内容不一致')
    for path in Path(folder).iterdir():
        if path.name in ASSETS | {archive.MANIFEST, archive.ASSET} | archive.HINT_ASSETS:durable(path)
