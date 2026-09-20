"""Deterministic RGB-D initialization, validation and untouched test partitions."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_split(frame_count: int) -> dict:
    test = list(range(7, frame_count, 8))
    validation = list(range(3, frame_count, 8))
    if frame_count >= 3:
        if not test:
            test = [frame_count - 1]
        if not validation:
            validation = [frame_count - 2]
    training = [index for index in range(frame_count) if index not in test + validation]
    return dict(schema='femto-gsplat-split-v1', train=training, validation=validation,
                test=test, seed_frame_indices=training.copy(),
                scope='Held out from RGB-D seed and Gaussian optimization; supplied camera poses shared.')


def validate_split(split: dict, frame_count: int) -> None:
    groups = [split.get(name) for name in ('train', 'validation', 'test')]
    if any(not isinstance(group, list) or not group for group in groups):
        raise ValueError('Strict split requires nonempty train, validation and test partitions')
    flat = [index for group in groups for index in group]
    if any(type(index) is not int for index in flat) or sorted(flat) != list(range(frame_count)):
        raise ValueError('Strict partitions must be disjoint and cover every frame exactly once')
    if split.get('seed_frame_indices') != split['train']:
        raise ValueError('Seed PLY must use training frames only')
