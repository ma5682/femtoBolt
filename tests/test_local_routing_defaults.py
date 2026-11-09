"""New local requests use bounded routing; existing plans keep recorded policies."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from femto.planning import planning_revision as revision


class LocalRoutingDefaultsTests(unittest.TestCase):
    def test_fixed_width_local_routing_does_not_depend_on_face_count(self):
        model=SimpleNamespace(store=SimpleNamespace(local_context={'reconstruction_scope':{'schema':'test'}}))
        for size in (0,1,100,2000,4999,5000,10000):
            with self.subTest(size=size):
                region={'triangle_indices':list(range(size))}
                self.assertEqual(revision.default_algorithms(model,region,{'mode':'fixed_width_auto'}),
                                 revision.LOCAL_DENSE_LAYOUT_ALGORITHMS)

    def test_nonlocal_and_direct_spacing_defaults_remain_explicit(self):
        local=SimpleNamespace(store=SimpleNamespace(local_context={'reconstruction_scope':{'schema':'test'}}))
        nonlocal_model=SimpleNamespace(store=SimpleNamespace(local_context={}))
        region={'triangle_indices':[0,1]}
        self.assertEqual(revision.default_algorithms(nonlocal_model,region,{'mode':'fixed_width_auto'}),revision.ORDERED_LAYOUT_ALGORITHMS)
        self.assertEqual(revision.default_algorithms(local,region,{'mode':'direct_row_spacing'}),revision.ALGORITHMS)

    def test_saved_supported_policies_are_replayed_without_upgrading_defaults(self):
        for algorithms in (revision.ALGORITHMS,revision.AUTO_LAYOUT_ALGORITHMS,
                           revision.ORDERED_LAYOUT_ALGORITHMS,revision.LOCAL_DENSE_LAYOUT_ALGORITHMS):
            before=deepcopy(algorithms)
            saved={'inputs':{'algorithms':before}}
            model,region,head=object(),object(),{'mode':'fixed_width_auto'}
            with patch.object(revision,'capture',return_value={'sentinel':'saved-policy'}) as capture:
                actual=revision.capture_for(model,region,head,saved)
            self.assertEqual(actual,{'sentinel':'saved-policy'})
            capture.assert_called_once_with(model,region,head,algorithms=before)
            self.assertEqual(saved['inputs']['algorithms'],before)


if __name__=='__main__':unittest.main()
