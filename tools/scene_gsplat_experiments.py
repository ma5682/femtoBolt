"""Sequential validation-only tuning followed by one untouched test evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # tools/ scripts import the femto package from the repository root
from femto.gsplat.gsplat_backend import runtime_environment, runtime_python
from femto.io_utils import write_json as dump
from femto.dataset.scene_gsplat_split import file_sha256


def rank(report: dict) -> tuple:
    metrics = report['evaluation']['aggregate']
    return (not report['accepted'], metrics['depth_rmse_m'], metrics['lpips'], -metrics['psnr'])


def run_suite(input_dir: Path, output: Path, iterations: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='femto-gsplat-experiments-v1', input=str(input_dir.resolve()),
                iterations=iterations, seed=20260903, downscale=4, max_gaussians=400000,
                selection_rule='Validation five checks first, then lower depth RMSE, LPIPS, higher PSNR',
                test_policy='Test RGB-D excluded from seed, optimization and all parameter selection',
                status='running', runs=[], stages=[])
    dump(output / 'experiment_report.json', plan)
    cache = {}

    def candidate(gradient: float, depth: float, antialiased: bool, ratio: float = 0.01) -> dict:
        key = (gradient, depth, antialiased, ratio)
        if key in cache:
            return cache[key]
        folder = output / f'run_{len(cache):02d}'
        folder.mkdir()
        command = [str(runtime_python()), '-m', 'femto.gsplat.scene_gsplat_trainer',
                   'train', '--input', str(input_dir.resolve()), '--output', str(folder.resolve()),
                   '--profile', 'low_vram', '--iterations', str(iterations),
                   '--grow-grad2d', str(gradient), '--depth-weight', str(depth),
                   '--means-lr-final-ratio', str(ratio), '--no-publish-latest',
                   '--antialiased' if antialiased else '--no-antialiased']
        dump(folder / 'command.json', {'argv': command})
        print(f'START {folder.name}: {key}', flush=True)
        with (folder / 'trainer_stdout.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=REPO_ROOT, env=runtime_environment(), stdout=log,
                                    stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            plan['status'] = 'failed'
            plan['failed_run'] = str(folder)
            dump(output / 'experiment_report.json', plan)
            raise RuntimeError(f'Experiment failed: {folder}')
        report = json.loads((folder / 'training_report.json').read_text(encoding='utf-8'))
        item = dict(run=str(folder.resolve()), grow_grad2d=gradient, depth_weight=depth,
                    antialiased=antialiased, means_lr_final_ratio=ratio, accepted=report['accepted'],
                    evaluation=report['evaluation'], gaussians=report['final_gaussians'],
                    elapsed_s=report['elapsed_s'], peak_cuda_memory_bytes=report['peak_cuda_memory_bytes'])
        cache[key] = item
        plan['runs'].append(item)
        dump(output / 'experiment_report.json', plan)
        print(f'DONE {folder.name}: {item["evaluation"]["aggregate"]}', flush=True)
        return item

    candidate(0.0002, 0.10, True, 1.0)
    best = candidate(0.0002, 0.10, True)
    for stage, values in [('grow_grad2d', [0.0002, 0.0004, 0.0006, 0.0010]),
                          ('depth_weight', [0.10, 0.20, 0.35]),
                          ('antialiased', [True, False])]:
        gradient = best['grow_grad2d'] if stage != 'grow_grad2d' else 0.0002
        depth = best['depth_weight'] if stage == 'antialiased' else 0.10
        trials = [candidate(value if stage == 'grow_grad2d' else gradient,
                            value if stage == 'depth_weight' else depth,
                            value if stage == 'antialiased' else True) for value in values]
        best = min(trials, key=rank)
        plan['stages'].append(dict(stage=stage, candidates=[item['run'] for item in trials], winner=best['run']))
        dump(output / 'experiment_report.json', plan)
    selected = Path(best['run'])
    plan['selected_run'] = str(selected)
    plan['selected_checkpoint_sha256'] = file_sha256(selected / 'checkpoints/latest.pt')
    plan['status'] = 'selection_locked_before_test'
    dump(output / 'experiment_report.json', plan)
    command = [str(runtime_python()), str(Path(__file__).resolve()), '--evaluate-run', str(selected),
               '--input', str(input_dir.resolve()), '--output', str((output / 'strict_test').resolve())]
    with (output / 'test_stdout.log').open('w', encoding='utf-8') as log:
        subprocess.run(command, env=runtime_environment(), stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    plan['strict_test'] = json.loads((output / 'strict_test/evaluation.json').read_text(encoding='utf-8'))
    plan['status'] = 'completed'
    dump(output / 'experiment_report.json', plan)


def evaluate_run(run: Path, input_dir: Path, output: Path) -> None:
    import torch
    from femto.gsplat.scene_gsplat_data import load_gsplat_dataset
    from femto.gsplat.scene_gsplat_trainer import TrainerConfig, _load_checkpoint, evaluate_model
    config = TrainerConfig(**json.loads((run / 'training_config.json').read_text(encoding='utf-8')))
    dataset = load_gsplat_dataset(input_dir, downscale=config.downscale, minimum_frames=12)
    checkpoint = _load_checkpoint(run / 'checkpoints/latest.pt', dataset, 'cpu')
    if len(checkpoint['params']['means']) > config.max_gaussians:
        raise ValueError('Checkpoint exceeds Gaussian cap')
    device = torch.device('cuda:0')
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(config.max_cuda_reserved_mib * 2**20 / total, 1 - config.min_cuda_free_mib * 2**20 / total)
    if fraction <= 0:
        raise ValueError('Insufficient CUDA memory budget')
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    params = {name: value.to(device) for name, value in checkpoint['params'].items()}
    output.mkdir(parents=True, exist_ok=False)
    report = evaluate_model(params, dataset, device, config, output, partition='test')
    report['selected_run'] = str(run.resolve())
    report['checkpoint_sha256'] = file_sha256(run / 'checkpoints/latest.pt')
    report['scale_validation'] = 'Not performed; rendering depth is not physical scale validation'
    dump(output / 'evaluation.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=6000)
    parser.add_argument('--evaluate-run', type=Path)
    arguments = parser.parse_args()
    if arguments.evaluate_run:
        evaluate_run(arguments.evaluate_run, arguments.input, arguments.output)
    else:
        run_suite(arguments.input, arguments.output, arguments.iterations)
