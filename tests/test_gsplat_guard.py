import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
try:
    import torch
    from gsplat.strategy import DefaultStrategy
except ImportError:
    torch = None
from scene_gsplat_guard import ResourceLimitError, check_memory, bounded_strategy_class
import scene_gsplat_trainer as trainer


@unittest.skipIf(torch is None, 'Requires dedicated .venv-gsplat runtime')
class GuardTests(unittest.TestCase):
    def test_memory_boundaries_and_low_profile(self):
        config = trainer.PROFILES['low_vram']
        self.assertEqual((config.downscale,config.max_gaussians),(4,400000))
        check_memory(dict(cuda_reserved_mib=100,cuda_free_mib=2000),config)
        with self.assertRaises(ResourceLimitError):
            check_memory(dict(cuda_reserved_mib=5120,cuda_free_mib=2000),config)
        with self.assertRaises(ResourceLimitError):
            check_memory(dict(cuda_reserved_mib=100,cuda_free_mib=1535),config)

    def make_state(self, n, cap):
        arrays = dict(means=torch.zeros(n,3),scales=torch.full((n,3),-3.),quats=torch.tensor([[1.,0,0,0]]).repeat(n,1),
                      opacities=torch.zeros(n),sh0=torch.zeros(n,1,3),shN=torch.zeros(n,15,3))
        params = {k:torch.nn.Parameter(v) for k,v in arrays.items()}
        optimizers = trainer._make_optimizers(params,1.)
        sum(p.sum() for p in params.values()).backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        strategy = trainer._create_strategy(replace(trainer.TrainerConfig(),max_init_points=n,max_gaussians=cap))
        strategy.verbose=False
        state = strategy.initialize_state(scene_scale=1.)
        state.update(grad2d=torch.ones(n),count=torch.ones(n))
        return strategy,params,optimizers,state

    def test_growth_budget_before_split_and_adam_alignment(self):
        for small in (False,True):
            strategy,params,optimizers,state = self.make_state(10,14)
            if small:
                with torch.no_grad(): params['scales'].fill_(-8.)
            duplicate,split = strategy._grow_gs(params,optimizers,state,1000)
            self.assertLessEqual(10+duplicate+2*split,14)
            n = len(params['means'])
            self.assertLessEqual(n,14)
            for name,optimizer in optimizers.items():
                self.assertIs(optimizer.param_groups[0]['params'][0],params[name])
                self.assertEqual(optimizer.state[params[name]]['exp_avg'].shape,params[name].shape)
            self.assertEqual(len(state['grad2d']),n)
            self.assertEqual(len(state['count']),n)

    def test_full_cap_does_not_grow_and_still_prunes(self):
        strategy,params,optimizers,state=self.make_state(10,10)
        self.assertEqual(strategy._grow_gs(params,optimizers,state,1000),(0,0))
        with torch.no_grad(): params['opacities'][0]=-20
        self.assertEqual(strategy._prune_gs(params,optimizers,state,1000),1)
        self.assertEqual(len(params['means']),9)

    def test_resource_stop_does_not_publish_or_overwrite_checkpoint(self):
        with TemporaryDirectory() as tmp:
            output=Path(tmp)
            checkpoint=output/'checkpoints/latest.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'last-good-checkpoint')
            with mock.patch.object(trainer,'_train_impl',side_effect=ResourceLimitError('test reserve')), mock.patch.object(trainer,'cuda_memory_snapshot',return_value={}):
                with self.assertRaises(ResourceLimitError):
                    trainer.train(Path('input'),output,trainer.TrainerConfig())
            self.assertEqual(checkpoint.read_bytes(),b'last-good-checkpoint')
            report=json.loads(next(output.glob('resource_stop_*.json')).read_text(encoding='utf-8'))
            self.assertEqual(report['status'],'stopped_resource_limit')
            self.assertFalse((output/'training_report.json').exists())

    def test_cuda_oom_is_reported_as_protected_stop(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.object(trainer,'_train_impl',side_effect=torch.cuda.OutOfMemoryError('simulated OOM')), mock.patch.object(trainer,'cuda_memory_snapshot',return_value={}):
                with self.assertRaises(ResourceLimitError):
                    trainer.train(Path('input'),Path(tmp),trainer.TrainerConfig())
            self.assertEqual(len(list(Path(tmp).glob('resource_stop_*.json'))),1)

    def test_real_allocation_is_not_attempted_when_memory_check_fails(self):
        config = trainer.TrainerConfig()
        with mock.patch.object(trainer,'guard_cuda',side_effect=ResourceLimitError('reserve')), mock.patch('gsplat.rendering.rasterization') as raster:
            with self.assertRaises(ResourceLimitError):
                trainer._rasterize({},None,'cuda:0',config)
            raster.assert_not_called()


if __name__=='__main__': unittest.main()
