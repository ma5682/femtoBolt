"""Independent, immutable installation records; never written to RGB-D evidence.

Importing a matrix proves structure only. No importer grants measured precision
or real execution eligibility. Synthetic records retain their explicit source.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

import numpy as np

from femto.asset_storage import publish_json, resolve_relative
from femto.dataset.provenance import value_hash
from femto.dataset.scene_dataset_validator import validate_pose
from femto.io_utils import load_json_object
from femto.paths import installation_calibration_root

SCHEMA = 'femto-installation-transform-v1'
ROTATION_ATOL = 1e-6  # float64 serialization structure tolerance, not physical accuracy.
RDF_AXES = '+X right, +Y down, +Z forward'


def _name(value):
    if not isinstance(value, str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9]*', value) is None:
        raise ValueError('Frame names must be explicit alphanumeric identifiers')
    return value


def transform_name(target, source):
    _name(target); _name(source)
    return 'T_'+target+source if len(target) == len(source) == 1 else 'T_'+target+'_'+source


def rigid_matrix(value):
    pose = np.asarray(value, dtype=np.float64)
    issue = validate_pose(pose)
    if issue or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(issue or 'Invalid homogeneous row')
    if (not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=ROTATION_ATOL, rtol=0)
            or abs(np.linalg.det(pose[:3, :3])-1) > ROTATION_ATOL):
        raise ValueError('Calibration rotation fails 1e-6 structural tolerance')
    return tuple(tuple(float(v) for v in row) for row in pose)


@dataclass(frozen=True)
class TransformRecord:
    target: str
    source_frame: str
    matrix: tuple
    calibration_id: str
    revision: str
    source: str
    device_ids: tuple[str, ...]
    installation_ids: tuple[str, ...]
    evidence_json: str
    target_axes: str
    source_axes: str
    dynamic: bool = False
    timestamp_s: float | None = None
    clock_domain: str | None = None
    unit: str = 'metre'
    raw_unit: str = 'metre'
    raw_rotation_convention: str = 'matrix'

    def __post_init__(self):
        _name(self.target); _name(self.source_frame)
        if self.target == self.source_frame:
            raise ValueError('Installation records require two distinct frames')
        pose = rigid_matrix(self.matrix)
        if self.unit != 'metre' or self.source not in ('synthetic', 'measured_import_unverified'):
            raise ValueError('Unsupported transform units or provenance')
        if type(self.dynamic) is not bool:
            raise ValueError('Dynamic/static relation must be explicit')
        for name in ('calibration_id', 'revision', 'target_axes', 'source_axes', 'raw_unit', 'raw_rotation_convention'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError('Missing transform '+name)
        for frame, axes in ((self.target, self.target_axes), (self.source_frame, self.source_axes)):
            if frame in ('C', 'W0') and axes != RDF_AXES:
                raise ValueError('C is color optical RDF and W0 retains initial color RDF')
        for name in ('device_ids', 'installation_ids'):
            if isinstance(getattr(self, name), (str, bytes)):
                raise ValueError('Device and installation IDs must be explicit sequences')
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)) or any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError('Installation and device binding evidence is required')
            object.__setattr__(self, name, values)
        evidence = json.loads(self.evidence_json)
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError('Explicit source evidence is required')
        if self.source == 'synthetic' and evidence.get('kind') != 'synthetic_test':
            raise ValueError('Synthetic source must not masquerade as measured calibration')
        if self.source != 'synthetic':
            files = evidence.get('files')
            if (not isinstance(files, list) or not files or any(not isinstance(f, dict)
                    or not isinstance(f.get('path'), str) or not f['path'].strip()
                    or not isinstance(f.get('sha256'), str)
                    or re.fullmatch(r'[0-9a-f]{64}', f['sha256']) is None for f in files)):
                raise ValueError('Imported calibration needs explicit source file paths and SHA-256 references')
        if self.dynamic:
            if (type(self.timestamp_s) not in (int, float) or not np.isfinite(self.timestamp_s)
                    or self.timestamp_s < 0 or not isinstance(self.clock_domain, str) or not self.clock_domain):
                raise ValueError('Dynamic pose requires a timestamp and clock domain')
        elif self.timestamp_s is not None or self.clock_domain is not None:
            raise ValueError('Fixed installation transform must not be relabeled as a timed pose')
        object.__setattr__(self, 'matrix', tuple(tuple(float(v) for v in row) for row in pose))
        object.__setattr__(self, 'evidence_json', json.dumps(evidence, sort_keys=True, allow_nan=False))

    @property
    def direction(self):
        return transform_name(self.target, self.source_frame)

    def to_dict(self):
        return dict(schema=SCHEMA, version=1, target=self.target, source_frame=self.source_frame,
                    matrix=[list(row) for row in self.matrix], transform_direction=self.direction,
                    calibration_id=self.calibration_id, revision=self.revision, source=self.source,
                    device_ids=list(self.device_ids), installation_ids=list(self.installation_ids),
                    evidence=json.loads(self.evidence_json), target_axes=self.target_axes,
                    source_axes=self.source_axes, dynamic=self.dynamic, timestamp_s=self.timestamp_s,
                    clock_domain=self.clock_domain, unit=self.unit, raw_unit=self.raw_unit,
                    raw_rotation_convention=self.raw_rotation_convention, structure_status='valid',
                    measurement_status='synthetic_only' if self.source == 'synthetic' else 'unverified',
                    precision_m=None, execution_ready=False, physical_accuracy_verified=False)

    @property
    def digest(self):
        return value_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get('schema') != SCHEMA or type(value.get('version')) is not int or value['version'] != 1:
            raise ValueError('Unsupported installation calibration schema/version')
        record = cls(value['target'], value['source_frame'], value['matrix'], value['calibration_id'],
                     value['revision'], value['source'], value['device_ids'], value['installation_ids'],
                     json.dumps(value['evidence'], allow_nan=False), value['target_axes'], value['source_axes'],
                     value['dynamic'], value['timestamp_s'], value['clock_domain'], value['unit'],
                     value['raw_unit'], value['raw_rotation_convention'])
        if record.to_dict() != value:
            raise ValueError('Imported matrix cannot confer measured calibration qualification')
        return record


def save_record(record, *, outputs=None):
    root = installation_calibration_root(outputs)
    path = resolve_relative(root, record.calibration_id+'/'+record.revision+'.json')
    publish_json(path, record.to_dict(), create_only=True)
    return path


def load_record(path):
    from pathlib import Path
    if Path(path).stat().st_size > 1024**2:
        raise ValueError('Installation record exceeds 1 MiB')
    return TransformRecord.from_dict(load_json_object(path))


def evidence_status(record):
    from pathlib import Path
    from femto.io_utils import sha256_file
    if record.source == 'synthetic':
        return dict(status='synthetic_only', measurement_qualified=False, execution_ready=False,
                    physical_accuracy_verified=False)
    failures = []
    for item in json.loads(record.evidence_json)['files']:
        try:
            if sha256_file(Path(item['path'])) != item['sha256']:
                raise ValueError('Calibration evidence digest changed')
        except (OSError, ValueError) as error:
            failures.append(dict(path=item['path'], reason=str(error)))
    return dict(status='changed_or_missing' if failures else 'source_verified_measurement_unverified',
                failures=failures, measurement_qualified=False, execution_ready=False,
                physical_accuracy_verified=False)
