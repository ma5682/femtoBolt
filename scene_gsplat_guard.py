"""Bound growth before allocation and stop cleanly on observed CUDA pressure."""
class ResourceLimitError(RuntimeError):
    pass


def cuda_memory_snapshot(device):
    import torch
    free, total = torch.cuda.mem_get_info(device)
    return dict(cuda_allocated_mib=round(torch.cuda.memory_allocated(device)/2**20, 1),
                cuda_reserved_mib=round(torch.cuda.memory_reserved(device)/2**20, 1),
                cuda_peak_mib=round(torch.cuda.max_memory_allocated(device)/2**20, 1),
                cuda_free_mib=round(free/2**20, 1), cuda_total_mib=round(total/2**20, 1))


def check_memory(snapshot, config):
    if snapshot['cuda_reserved_mib'] >= config.max_cuda_reserved_mib:
        raise ResourceLimitError(f"CUDA reserved {snapshot['cuda_reserved_mib']} MiB reached limit {config.max_cuda_reserved_mib} MiB")
    if snapshot['cuda_free_mib'] < config.min_cuda_free_mib:
        raise ResourceLimitError(f"CUDA free {snapshot['cuda_free_mib']} MiB below reserve {config.min_cuda_free_mib} MiB")


def guard_cuda(device, config):
    snapshot = cuda_memory_snapshot(device)
    check_memory(snapshot, config)
    return snapshot


def bounded_strategy_class(base):
    """Wrap pinned gsplat 1.5.3 growth, preserving its pruning/optimizer logic.

    Each selected parent can append at most two rows during split (before the
    parent is removed). Budget this transient count, not merely the final count.
    Screen-radius growth is deliberately disabled so gradients are the only gate.
    """
    class BoundedStrategy(base):
        def _grow_gs(self, params, optimizers, state, step):
            import torch
            n = len(params['means'])
            if n > self.max_gaussians:
                raise ResourceLimitError(f'Gaussian count {n} exceeds cap {self.max_gaussians}')
            if self.refine_scale2d_stop_iter != 0:
                raise ResourceLimitError('Bounded strategy requires radius-based growth disabled')
            if params['means'].is_cuda:
                guard_cuda(params['means'].device, self.guard_config)
            budget = max(0, (self.max_gaussians - n)//2)
            if budget == 0:
                return 0, 0
            with torch.no_grad():
                score = state['grad2d'] / state['count'].clamp_min(1)
                candidates = torch.where(torch.isfinite(score) & (score > self.grow_grad2d))[0]
                if len(candidates) > budget:
                    candidates = candidates[torch.topk(score[candidates], budget).indices]
                mask = torch.zeros_like(score, dtype=torch.bool)
                mask[candidates] = True
                state['grad2d'] = torch.where(mask, state['grad2d'], 0.)
                result = super()._grow_gs(params, optimizers, state, step)
            if len(params['means']) > self.max_gaussians:
                raise ResourceLimitError('Gaussian cap invariant failed')
            return result
    return BoundedStrategy
