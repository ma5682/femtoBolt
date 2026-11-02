"""Headless versioned nominal coverage evaluation/replay without a camera."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from femto.planning import coverage_archive, coverage_certificate, coverage_domain, footprint_model
from femto.planning.planning_io import staging_directory

CASE_SCHEMA = 'femto-nominal-coverage-case-v1'


@contextmanager
def inputs(case: Path | None, project: Path | None, scratch: Path, check, *, route_schema=None) -> Iterator[tuple]:
    if case is not None:
        if case.stat().st_size > 64*1024**2:raise ValueError('名义实验输入超过 64 MiB')
        record = json.loads(case.read_text(encoding='utf-8-sig'))
        if (record.get('schema') != CASE_SCHEMA or record.get('coordinate_frame') != 'world_rdf'
                or record.get('unit') != 'metre' or record.get('execution_ready') is not False
                or record.get('physical_accuracy_verified') is not False):
            raise ValueError('名义实验必须声明数值模型、RDF / 米及研发语义')
        domain = coverage_domain.CoverageDomain.mesh(record['vertices'], record['faces'])
        model = footprint_model.validate(record['footprint'])
        yield domain, record['seeds'], model, None
        return
    from femto.planning.region_project import open_project
    with staging_directory(scratch, '.coverage-input-') as workspace:
        model = open_project(project, workspace=workspace, progress=lambda _: check())
        if model.saved_plan is None:raise ValueError('独立覆盖验证需要含 P1 台账与已接纳路线的便携方案')
        domain,seeds,footprint,route=coverage_domain.from_plan(model, model.store.regions[0], model.saved_plan, check)
        if route_schema == 'femto-accepted-v3-coverage-source-v2':
            from femto.planning.coverage_workflow import route_binding
            from femto.planning.surface_trajectory import SurfaceTrajectory
            route=route_binding(SurfaceTrajectory.from_dict(model.saved_plan['surface_trajectory']))
        yield domain,seeds,footprint,route


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='P2 名义模型覆盖实验与复算，不改变原路线')
    commands = parser.add_subparsers(dest='action', required=True)
    evaluate = commands.add_parser('evaluate', help='计算并保存到新的 outputs 子目录')
    verify = commands.add_parser('verify', help='使用指定原输入独立复算已有结果')
    for command in (evaluate, verify):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument('--case', type=Path, help='明确声明 RDF / 米的数值实验 JSON')
        source.add_argument('--project', type=Path, help='含 P1 台账及原 v3 路线的便携 .region.zip')
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--max-depth', type=int, default=4)
    evaluate.add_argument('--seconds', type=float, default=45.)
    evaluate.add_argument('--algorithm', choices=('v1','v2'), default='v1', help='v1 保留 B1 调度；v2 使用 B2 分层调度')
    verify.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    def check():
        if time.monotonic()-started > 600:raise TimeoutError('独立覆盖实验超过 600 秒总预算')
    try:
        if args.action == 'evaluate':
            destination = args.output.resolve()
            if destination == ROOT/'outputs' or not destination.is_relative_to((ROOT/'outputs').resolve()):
                raise ValueError('独立实验结果必须放在当前工作区 outputs 的新子目录')
            if destination.exists():raise FileExistsError('输出目录已存在，请使用新目录')
        scratch = ROOT/'outputs'; scratch.mkdir(exist_ok=True)
        route_schema=None
        if args.action=='verify':
            manifest=args.archive/coverage_archive.MANIFEST
            if not manifest.is_file() or manifest.stat().st_size>coverage_archive.MAX_MANIFEST_BYTES:
                raise ValueError('覆盖清单缺失或超过读取预算')
            metadata=json.loads(manifest.read_text(encoding='utf-8-sig'))
            coverage_certificate.validate_metadata(metadata)
            route_schema=metadata['inputs']['route']['schema']
        elif args.algorithm=='v2':route_schema='femto-accepted-v3-coverage-source-v2'
        with inputs(args.case, args.project, scratch, check,route_schema=route_schema) as (domain, seeds, model, route):
            if args.action == 'evaluate':
                certificate = coverage_certificate.evaluate(domain, seeds, model, route_binding=route, check=check,
                    algorithm=coverage_certificate.CURRENT_ALGORITHM if args.algorithm=='v2' else coverage_certificate.ALGORITHM,
                    budget=coverage_certificate.CoverageBudget(max_depth=args.max_depth, seconds=args.seconds))
                folder = coverage_archive.save(destination, certificate, domain, seeds, model, route_binding=route, check=check)
            else:
                folder = args.archive.resolve()
                certificate = coverage_archive.read(folder, domain, seeds, model, route_binding=route, check=check)
            print(json.dumps(dict(action=args.action, archive=str(folder),
                coverage_computation_sha256=certificate.metadata['coverage_computation_sha256'],
                summary=certificate.summary, metrics=certificate.metadata['metrics']), ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':raise SystemExit(main())
