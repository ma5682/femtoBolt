"""One save request's target and route verification receipt; no global cache.

Native export must complete its independent target replay and publication before
portable export can consume this receipt. Coverage proofs keep their own existing
verification path. This session retains digests and scoped identity metadata,
never models or arrays.
"""
from __future__ import annotations
from femto.performance_detail import observed
from femto.operation_diagnostics import diagnosed

from contextlib import contextmanager
from threading import get_ident

from femto.planning import planning_revision
from femto.io_utils import sha256_file


class SaveValidationSession:
    def __init__(self):
        self._state = 'new'
        self._binding = None
        self._thread = None
        self._portable_target_id = None

    def __enter__(self):
        if self._state != 'new':
            raise ValueError('保存验证会话不能重复使用')
        self._thread = get_ident()
        self._state = 'active'
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._binding = None
        self._thread = None
        self._portable_target_id = None
        self._state = 'closed'

    def _require(self, state):
        if self._thread != get_ident() or self._state != state:
            self.close()
            raise ValueError('保存验证会话已失效或不属于当前保存任务')

    @staticmethod
    @diagnosed('save.bind_and_rehash_inputs')
    @observed('save.bind_and_rehash_inputs')
    def _capture(model, region, plan, ledger, check):
        check()
        context = plan['planning_context']
        planning_revision.require_current(model, region, plan['treatment_head'], context['revision'])
        if ledger is None or ledger.metadata != context['target']:
            raise ValueError('保存验证会话缺少当前原目标')
        actual = {}
        for name, array in ledger.arrays.items():
            check()
            actual[name] = planning_revision.array_record(array)
        if actual != ledger.metadata['arrays']:
            raise ValueError('保存验证后的原目标数组已改变')
        # Plain coverage evidence can share the native replay only when its
        # exact arrays are rehashed together with every other save dependency.
        # Other certificate families keep their existing verification paths.
        from femto.planning import coverage_workflow, coverage_certificate
        certificate = coverage_workflow.cached(model, plan)
        coverage = None
        if (type(certificate) is coverage_certificate.Certificate
                and plan['surface_trajectory']['planner']['algorithm_version'] == 'quality-constrained-surface-v3'
                and certificate.metadata['algorithm'] not in coverage_certificate.RECALCULATION_ALGORITHMS):
            coverage_certificate._checked_assets(certificate, check)
            coverage = certificate.metadata
        # Bind full plan/region, including route, quality, algorithms and budgets.
        # require_current hashes live RGB/depth/mask/calibration/mesh/reference;
        # the source file is hashed separately, not trusted via store.digest.
        result = planning_revision.digest(dict(
            plan=plan, region=region, target=ledger.metadata, arrays=actual, coverage=coverage,
            source_sha256=sha256_file(model.store.source, check)))
        check()
        return result

    def _prepare_native(self, model, region, plan, ledger, check):
        self._require('route_check_started')
        try:
            if self._binding != self._capture(model, region, plan, ledger, check):
                raise ValueError('路线核验期间输入已变化，本次验证会话失效')
            self._state = 'prepared'
        except BaseException:
            self.close()
            raise

    def _begin_route_validation(self, model, region, plan, ledger, check):
        """Bind before route validation, so later reuse covers that exact input."""
        self._require('active')
        try:
            self._binding = self._capture(model, region, plan, ledger, check)
            self._state = 'route_check_started'
        except BaseException:
            self.close()
            raise

    def _native_published(self, model, region, plan, ledger, check):
        self._require('prepared')
        try:
            if self._binding != self._capture(model, region, plan, ledger, check):
                raise ValueError('保存期间输入已变化，本次验证会话失效')
            self._state = 'published'
        except BaseException:
            self.close()
            raise

    @contextmanager
    def consume(self, model, region, plan, ledger, check):
        self._require('published')
        try:
            self._state = 'consuming'
            # The portable writer may reuse only this exact validated target.
            # Retain its identity, not its arrays, within this one-use scope.
            self._portable_target_id = id(ledger)
            self.recheck(model, region, plan, ledger, check)
            yield self
        finally:
            self.close()

    def recheck(self, model, region, plan, ledger, check):
        self._require('consuming')
        try:
            if id(ledger) != self._portable_target_id:
                raise ValueError('保存目标不属于当前验证会话')
            if self._binding != self._capture(model, region, plan, ledger, check):
                raise ValueError('保存期间输入已变化，本次验证会话失效')
        except BaseException:
            self.close()
            raise

    def write_verified_coverage(self, folder, model, region, plan, ledger, certificate, check):
        """Write and rehash a plain proof already replayed in this save task."""
        from pathlib import Path
        from femto.planning import coverage_archive, coverage_certificate, coverage_workflow
        from femto.planning.planning_io import durable
        self.recheck(model, region, plan, ledger, check)
        if (type(certificate) is not coverage_certificate.Certificate
                or coverage_workflow.cached(model, plan) is not certificate
                or plan['surface_trajectory']['planner']['algorithm_version'] != 'quality-constrained-surface-v3'
                or certificate.metadata['algorithm'] in coverage_certificate.RECALCULATION_ALGORITHMS):
            raise ValueError('本次覆盖验证结果不支持复用')
        coverage_archive.write_assets(folder, certificate, check)
        readback = coverage_archive.load_assets(folder, check)
        if readback.metadata != certificate.metadata:
            raise ValueError('覆盖资产回读与已验证结果不一致')
        for name in coverage_archive.asset_names(certificate.metadata):
            durable(Path(folder)/name)
        self.recheck(model, region, plan, ledger, check)
