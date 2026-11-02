"""One synchronous automatic task's independently replayed original proof.

This capability never caches candidate/final proof acceptance and is invalid
outside its thread or context. All inputs, target bytes and proof assets are
rechecked before any reuse; mutations cause failure rather than fallback.
"""
from threading import get_ident
from femto.planning import coverage_certificate as coverage,planning_revision


def _domain_record(domain,check):
    ledger=domain.ledger
    if ledger is None or domain.revision is None:raise ValueError('原证据复用需要完整目标台账')
    arrays={}
    for name,value in sorted(ledger.arrays.items()):
        check();arrays[name]=planning_revision.array_record(value)
    geometry={}
    for name in ('vertices','faces','source_faces','areas','quality','representative_faces'):
        check();geometry[name]=planning_revision.array_record(getattr(domain,name))
    parents=[]
    for (face,pixel),children in sorted(domain._parent_children.items()):
        check()
        if (type(face) is not int or type(pixel) is not int or type(children) is not set
                or any(type(child) is not int for child in children)):
            raise ValueError('原证据复用目标分区无效')
        parents.append([face,pixel,sorted(children)])
    return dict(binding=domain.binding,revision=domain.revision,ledger=ledger.metadata,
                arrays=arrays,geometry=geometry,parent_children=parents)


class _OldCoverageSession:
    """Retain one just-replayed old proof for this synchronous task only."""
    def __init__(self,model,region,plan,certificate,domain,seeds,footprint,route,check,*,guide=None):
        self.model,self.region,self.plan,self.certificate=model,region,plan,certificate
        self.domain,self.seeds,self.footprint,self.route,self.check=domain,seeds,footprint,route,check
        self.guide=guide
        self.state='new';self.thread=None;self.snapshot=None

    def _digest(self,domain,seeds,footprint,route):
        self.check();coverage._checked_assets(self.certificate,self.check)
        guide=None
        if self.certificate.metadata.get('algorithm') in coverage.RECALCULATION_ALGORITHMS:
            from femto.planning.coverage_recalculation import require_guide
            guide=require_guide(self.certificate,self.check).metadata
        supplied_guide=None
        if self.guide is not None:
            coverage._checked_assets(self.guide,self.check)
            supplied_guide=self.guide.metadata
        return planning_revision.digest(dict(plan=self.plan,region=self.region,certificate=self.certificate.metadata,
            guide=guide,supplied_guide=supplied_guide,domain=_domain_record(domain,self.check),seeds=seeds,footprint=footprint,route=route))

    def __enter__(self):
        if self.state!='new':raise ValueError('原证据复用上下文不能重复使用')
        self.state='active';self.thread=get_ident()
        try:
            before=self._digest(self.domain,self.seeds,self.footprint,self.route)
            planning_revision.require_current(self.model,self.region,self.plan['treatment_head'],self.plan['planning_context']['revision'])
            coverage.validate(self.certificate,self.domain,self.seeds,self.footprint,route_binding=self.route,check=self.check,guide=self.guide)
            after=self._digest(self.domain,self.seeds,self.footprint,self.route)
            if before!=after:raise ValueError('原证据核验期间输入发生变化')
            self.snapshot=after
            self.require(self.model,self.region,self.plan,self.certificate,self.domain,self.seeds,self.footprint,self.route,self.check)
        except BaseException:
            self.close();raise
        return self

    def require(self,model,region,plan,certificate,domain,seeds,footprint,route,check):
        if self.state!='active' or self.thread!=get_ident():raise ValueError('原证据复用上下文已关闭或不属于当前线程')
        if (model is not self.model or region is not self.region or plan is not self.plan
                or certificate is not self.certificate or check is not self.check):
            raise ValueError('原证据复用只能用于当前同步任务')
        for _ in range(2):
            if self._digest(domain,seeds,footprint,route)!=self.snapshot:
                raise ValueError('原证据复用期间输入或目标发生变化')
            planning_revision.require_current(model,region,plan['treatment_head'],plan['planning_context']['revision'])
            self.check()

    def close(self):
        self.state='closed';self.thread=None
        self.model=self.region=self.plan=self.certificate=self.domain=self.seeds=self.footprint=self.route=None
        self.guide=None

    def __exit__(self,*args):self.close()
