"""One synchronous research screen's freshly validated source route.

Only its own validated source object is reused, never candidate acceptance,
coverage, or any state across requests. Full plan, region, route and live input
fingerprints are checked before each use and after the enclosing operation.
"""
from femto.planning import planning_revision, planning_state
from threading import get_ident


class _ScreenSource:
    def __init__(self,model,region,plan,check):
        self.model,self.region,self.plan,self.check=model,region,plan,check
        check();self.plan_hash=planning_revision.digest(plan);self.region_hash=planning_revision.digest(region)
        self.revision=plan['planning_context']['revision'];self.head=plan['treatment_head']
        planning_revision.require_current(model,region,self.head,self.revision)
        self.trajectory=planning_state.validated_surface(plan,model.store.mesh,region,check=check)
        self.trajectory_hash=planning_revision.digest(self.trajectory.to_dict())
        self._state='new';self._thread=None

    def __enter__(self):
        if self._state!='new':raise ValueError('源路线复用上下文不能重复使用')
        self._state='active';self._thread=get_ident()
        try:self.verify()
        except BaseException:self.close();raise
        return self

    def __exit__(self,*exc):
        self.close()

    def close(self):
        self._state='closed';self._thread=None
        self.model=self.region=self.plan=self.trajectory=self.head=self.revision=None

    def verify(self):
        if self._state!='active' or self._thread!=get_ident():
            raise ValueError('源路线复用上下文已关闭或不属于当前线程')
        self.check()
        if (planning_revision.digest(self.plan)!=self.plan_hash
                or planning_revision.digest(self.region)!=self.region_hash
                or planning_revision.digest(self.trajectory.to_dict())!=self.trajectory_hash):
            raise ValueError('自动筛选期间已验证的源路线或选区发生变化')
        planning_revision.require_current(self.model,self.region,self.head,self.revision)
        self.check()
        if (planning_revision.digest(self.plan)!=self.plan_hash
                or planning_revision.digest(self.region)!=self.region_hash
                or planning_revision.digest(self.trajectory.to_dict())!=self.trajectory_hash):
            raise ValueError('自动筛选期间已验证的源路线或选区发生变化')
        planning_revision.require_current(self.model,self.region,self.head,self.revision)

    def read(self,model,region,plan):
        if model is not self.model or region is not self.region or plan is not self.plan:
            raise ValueError('源路线复用只能用于当前同步筛选')
        self.verify();return self.trajectory
