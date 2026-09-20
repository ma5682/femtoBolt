"""Read-only calibration identity and first valid RGB-D reference provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np


CALIBRATION_FILES = (
    'camera_metadata.json', 'color_intrinsic.json', 'depth_intrinsic.json',
    'depth_to_color.json', 'distortion.json', 'factory_calibration.json',
)


def calibration_identity(dataset):
    files = {name: hashlib.sha256((Path(dataset) / 'calibration' / name).read_bytes()).hexdigest()
             for name in CALIBRATION_FILES}
    serialized = json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return dict(algorithm='sha256', sha256=hashlib.sha256(serialized).hexdigest(), files=files)


def reference_provenance(session_id, record, calibration):
    reference = record.get('reference_frame_id')
    if (not isinstance(session_id, str) or not session_id
            or type(reference) is not int or reference < 0 or record.get('frame_id') != reference
            or record.get('state', 'tracking') != 'tracking'):
        raise ValueError('Invalid initial reference frame')
    transform = np.asarray(record.get('T_WC'), dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform, np.eye(4), rtol=0, atol=1e-9)):
        raise ValueError('Initial reference pose is not identity')
    timestamps = {key: record.get(key) for key in
                  ('capture_timestamp_us', 'color_timestamp_us', 'depth_timestamp_us')}
    if any(type(value) is not int or value < 0 for value in timestamps.values()):
        raise ValueError('Invalid initial RGB-D timestamp')
    if abs(timestamps['color_timestamp_us'] - timestamps['depth_timestamp_us']) > 5000:
        raise ValueError('Initial reference RGB-D is not synchronized')
    return dict(schema='femto-local-reference-v1', session_id=session_id,
                reference_frame_id=reference, **timestamps,
                coordinate_frame='world_rdf', unit='metre', transform_direction='T_W0_Ct',
                axes='+X right, +Y down, +Z forward',
                origin='first_valid_color_camera', calibration=calibration)


def validate_reference_provenance(proof, dataset, frame, pose):
    """Validate a new provenance bundle against the actual posed dataset bytes."""
    record = dict(frame_id=frame.get('frame'), reference_frame_id=pose.get('reference_frame_id'),
                  T_WC=pose.get('T_WC'), capture_timestamp_us=frame.get('camera_timestamp_us'),
                  color_timestamp_us=frame.get('color_timestamp_us'),
                  depth_timestamp_us=frame.get('depth_timestamp_us'))
    expected = reference_provenance(pose.get('session_id'), record, calibration_identity(dataset))
    if proof != expected or pose.get('reference_provenance') != expected:
        raise ValueError('Initial reference / calibration provenance mismatch')
    return expected
