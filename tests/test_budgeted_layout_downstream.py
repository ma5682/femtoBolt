"""Opt-in local layout remains compatible with established coverage workflows."""
from pathlib import Path
import tempfile
import unittest

from test_coverage_workflow import fixture
from femto.planning import coverage_certificate,coverage_workflow,coverage_refinement
from femto.planning import refinement_workflow,strict_workflow,surface_workspace,planning_state
from femto.planning.fixed_width_layout import OPTIMIZED_PATCH_VERSION
from femto.planning.treatment_head import head_parameters


class BudgetedLayoutDownstreamTests(unittest.TestCase):
    routing = None
    layout = OPTIMIZED_PATCH_VERSION
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.model,cls.region,_,_,_=fixture(Path(cls.tmp.name)/'source')
        cls.model.store.regions=[cls.region]
        head=head_parameters('fixed_width_auto',effective_treatment_width_mm=5,overlap_percent=20)
        cls.plan=surface_workspace.generate(cls.model,cls.region,head,layout_algorithm=cls.layout,
            **({'routing_algorithm':cls.routing} if cls.routing else {}))
        proof,ledger=coverage_workflow.calculate(cls.model,cls.region,cls.plan,
            budget=coverage_certificate.CoverageBudget(max_depth=1,max_added_cells=3000))
        cls.attached=coverage_workflow.attach(cls.model,cls.region,cls.plan,proof,ledger)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_strict_route_and_restore_validate_budgeted_base(self):
        from test_variable_footprint import profile
        from femto.planning import variable_width_workflow
        strict=strict_workflow.calculate(self.model,self.region,self.attached)
        planning_state.validate(strict,self.model.store.mesh,self.region)
        variable=variable_width_workflow.calculate(self.model,self.region,strict,profile(.0025))
        planning_state.validate(variable,self.model.store.mesh,self.region)
        strict_again=variable_width_workflow.restore(self.model,self.region,variable)
        self.assertEqual(strict_again['surface_trajectory'],strict['surface_trajectory'])
        restored=strict_workflow.restore(self.model,self.region,strict_again)
        self.assertEqual(restored['surface_trajectory'],self.plan['surface_trajectory'])
        self.assertFalse(restored['execution_ready'])


    def test_refinement_preserves_and_validates_budgeted_base(self):
        # Strict conversion clears caches; restore them independently for this test.
        proof,ledger=coverage_workflow.calculate(self.model,self.region,self.plan,
            budget=coverage_certificate.CoverageBudget(max_depth=1,max_added_cells=3000))
        attached=coverage_workflow.attach(self.model,self.region,self.plan,proof,ledger)
        result=refinement_workflow.calculate(self.model,self.region,attached,
            budget=coverage_refinement.RefinementBudget(max_rounds=0))
        planning_state.validate(result,self.model.store.mesh,self.region)
        restored=refinement_workflow.restore(self.model,self.region,result)
        self.assertEqual(restored['surface_trajectory'],self.plan['surface_trajectory'])
        self.assertFalse(restored['execution_ready'])


from femto.planning.continuous_quality import IMPROVED_ROUTING,SUPPORT_ROUTING,BALANCED_ROUTING,SEAM_ROUTING,SEAM_BALANCED_ROUTING,PRESERVED_DETOUR_ROUTING,SHORT_DETOUR_ROUTING


class ImprovedRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = IMPROVED_ROUTING


class SupportRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = SUPPORT_ROUTING


class BalancedRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = BALANCED_ROUTING


class SeamRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = SEAM_ROUTING


class SeamBalancedRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = SEAM_BALANCED_ROUTING


class PreservedDetourRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = PRESERVED_DETOUR_ROUTING


class ShortDetourRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = SHORT_DETOUR_ROUTING


from femto.planning.fixed_width_layout import FOLDBACK_PATCH_VERSION


class FoldbackRoutingDownstreamTests(BudgetedLayoutDownstreamTests):
    routing = SEAM_BALANCED_ROUTING
    layout = FOLDBACK_PATCH_VERSION
