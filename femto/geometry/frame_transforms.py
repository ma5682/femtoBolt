"""T_AB maps B to A, metre translations; no missing-extrinsic identity fallback."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femto.dataset.installation_calibration import RDF_AXES, TransformRecord, rigid_matrix, transform_name


@dataclass(frozen=True)
class TransformChain:
    target: str
    source_frame: str
    matrix: tuple
    dependencies: tuple[TransformRecord, ...]
    target_axes: str | None = None
    source_axes: str | None = None
    timestamp_s: float | None = None
    clock_domain: str | None = None

    def __post_init__(self):
        transform_name(self.target, self.source_frame)
        records = tuple(self.dependencies)
        if not records or any(not isinstance(r, TransformRecord) for r in records):
            raise ValueError('Transform chains require explicit transform evidence')
        conventions = {}
        for record in records:
            for name, axes in ((record.target, record.target_axes), (record.source_frame, record.source_axes)):
                if name in conventions and conventions[name] != axes:
                    raise ValueError('Transform axis conventions do not match')
                conventions[name] = axes
        for endpoint, field in ((self.target, 'target_axes'), (self.source_frame, 'source_axes')):
            axes = getattr(self, field) or conventions.get(endpoint) or (RDF_AXES if endpoint == 'W0' else None)
            if not isinstance(axes, str) or not axes or endpoint in ('C', 'W0') and axes != RDF_AXES:
                raise ValueError('Missing or conflicting endpoint axis convention')
            if endpoint in conventions and conventions[endpoint] != axes:
                raise ValueError('Endpoint convention conflicts with transform evidence')
            object.__setattr__(self, field, axes)
        if self.timestamp_s is not None:
            if (type(self.timestamp_s) not in (int, float) or not np.isfinite(self.timestamp_s)
                    or self.timestamp_s < 0 or not isinstance(self.clock_domain, str) or not self.clock_domain):
                raise ValueError('Timed transform chain requires finite timestamp and clock domain')
        elif self.clock_domain is not None:
            raise ValueError('Static transform chain cannot have a dynamic clock domain')
        object.__setattr__(self, 'dependencies', records)
        object.__setattr__(self, 'matrix', rigid_matrix(self.matrix))

    def apply(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError('Expected finite metre XYZ')
        matrix = np.asarray(self.matrix)
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def to_dict(self):
        return dict(schema='femto-transform-chain-v1', target=self.target, source_frame=self.source_frame,
                    transform_direction=transform_name(self.target, self.source_frame), unit='metre',
                    matrix=[list(r) for r in self.matrix], source=[r.source for r in self.dependencies],
                    target_axes=self.target_axes, source_axes=self.source_axes,
                    timestamp_s=self.timestamp_s, clock_domain=self.clock_domain,
                    dependencies=[dict(id=r.calibration_id, revision=r.revision, sha256=r.digest) for r in self.dependencies],
                    execution_ready=False, physical_accuracy_verified=False)


def _chain(value):
    if isinstance(value, TransformRecord):
        return TransformChain(value.target, value.source_frame, value.matrix, (value,),
                              value.target_axes, value.source_axes, value.timestamp_s, value.clock_domain)
    if not isinstance(value, TransformChain):
        raise ValueError('Missing or invalid transform')
    return value


def compose(*values, synchronization_tolerance_s=0):
    if not values:
        raise ValueError('Missing transform chain')
    chain = _chain(values[0])
    if (type(synchronization_tolerance_s) not in (int, float)
            or not np.isfinite(synchronization_tolerance_s) or synchronization_tolerance_s < 0):
        raise ValueError('Invalid transform synchronization tolerance')
    matrix = np.asarray(chain.matrix)
    dependencies = chain.dependencies
    source = chain.source_frame
    source_axes = chain.source_axes
    times = [(chain.timestamp_s, chain.clock_domain)] if chain.timestamp_s is not None else []
    for value in values[1:]:
        following = _chain(value)
        if source != following.target:
            raise ValueError('Transform endpoints cannot be composed')
        # Endpoints also carry conventions, not merely matching labels.
        if source_axes != following.target_axes:
            raise ValueError('Transform axis conventions do not match')
        matrix = matrix @ np.asarray(following.matrix)
        source = following.source_frame
        source_axes = following.source_axes
        dependencies += following.dependencies
        if following.timestamp_s is not None:
            times.append((following.timestamp_s, following.clock_domain))
    if times and (len({domain for _, domain in times}) != 1
                  or max(t for t, _ in times)-min(t for t, _ in times) > synchronization_tolerance_s):
        raise ValueError('Dynamic transforms are not synchronized in one clock domain')
    return TransformChain(chain.target, source, matrix, dependencies, chain.target_axes, source_axes,
                          times[0][0] if times else None, times[0][1] if times else None)


def inverse(value):
    chain = _chain(value)
    matrix = np.linalg.inv(np.asarray(chain.matrix))
    return TransformChain(chain.source_frame, chain.target,
                          matrix, chain.dependencies, chain.source_axes, chain.target_axes,
                          chain.timestamp_s, chain.clock_domain)


def camera_to_tool(hand_eye, tool_mount, *, rigid_installation_verified):
    if (not isinstance(hand_eye, TransformRecord) or not isinstance(tool_mount, TransformRecord)
            or rigid_installation_verified is not True or hand_eye.dynamic or tool_mount.dynamic
            or not set(hand_eye.installation_ids) & set(tool_mount.installation_ids)
            or not set(hand_eye.device_ids) & set(tool_mount.device_ids)):
        raise ValueError('Camera and tool must have verified rigid relative mounting')
    if (hand_eye.target, hand_eye.source_frame, tool_mount.target, tool_mount.source_frame) != ('F', 'C', 'F', 'U'):
        raise ValueError('Expected T_FC and T_FU')
    return compose(inverse(tool_mount), hand_eye)


class SpatialContext:
    """Caller-owned live applicability; history chains remain immutable.

    This iteration supports explicit synthetic contexts only. Measured imports
    remain unverified. Rebuilding W0 after base relocation needs a new session.
    """
    def __init__(self, *, session_id, hand_eye, tool_mount, synchronization_tolerance_s,
                 max_pose_age_s, clock_domain, budget_source):
        if (not isinstance(hand_eye, TransformRecord) or not isinstance(tool_mount, TransformRecord)
                or hand_eye.source != 'synthetic' or tool_mount.source != 'synthetic'
                or not isinstance(budget_source, str) or not budget_source.startswith('synthetic:')
                or type(synchronization_tolerance_s) not in (int, float)
                or type(max_pose_age_s) not in (int, float)
                or not np.isfinite([synchronization_tolerance_s, max_pose_age_s]).all()
                or synchronization_tolerance_s <= 0 or max_pose_age_s <= 0
                or not session_id or not clock_domain):
            raise ValueError('Real calibration and timing are not verified; only explicit synthetic context is available')
        camera_to_tool(hand_eye, tool_mount, rigid_installation_verified=True)
        self.session_id = session_id
        self.hand_eye, self.tool_mount = hand_eye, tool_mount
        self.sync_s, self.max_age_s = synchronization_tolerance_s, max_pose_age_s
        self.clock_domain, self.budget_source = clock_domain, budget_source
        self.base_to_w0 = None
        self.invalid_reason = 'reference_not_established'
        self.mount_valid = True
        self.requires_new_session = False
        self.generation = 0
        self._installation_binding = hand_eye.digest, tool_mount.digest
        self._pose_binding = None

    def establish_initial_reference(self, flange, *, frame_id, reference_frame_id,
                                    camera_timestamp_s, base_state, installation_evidence):
        if (self.requires_new_session or self.base_to_w0 is not None or frame_id != reference_frame_id or type(frame_id) is not int
                or frame_id < 0 or not installation_evidence
                or base_state not in ('locked', 'fixed_with_evidence') or not self.mount_valid):
            raise ValueError('Only the evidenced initial color-camera frame can establish a new W0')
        self._pose(flange, camera_timestamp_s)
        self._pose_binding = flange.device_ids, flange.installation_ids
        chain = inverse(compose(flange, self.hand_eye))
        self.base_to_w0 = TransformChain('W0', 'B', chain.matrix, chain.dependencies, RDF_AXES, chain.source_axes)
        self.invalid_reason = None
        self.generation += 1
        return self.base_to_w0

    def _pose(self, flange, timestamp):
        if (not isinstance(flange, TransformRecord) or (flange.target, flange.source_frame) != ('B', 'F')
                or not flange.dynamic or flange.source != 'synthetic'
                or flange.clock_domain != self.clock_domain or type(timestamp) not in (int, float)
                or not np.isfinite(timestamp) or abs(flange.timestamp_s-timestamp) > self.sync_s):
            raise ValueError('Missing synchronized flange pose in the same clock domain')
        if (not set(flange.device_ids) & set(self.hand_eye.device_ids)
                or not set(flange.installation_ids) & set(self.hand_eye.installation_ids)
                or (self.hand_eye.digest, self.tool_mount.digest) != self._installation_binding
                or self._pose_binding is not None and (flange.device_ids, flange.installation_ids) != self._pose_binding):
            raise ValueError('Pose device/installation or calibration binding changed')

    def current(self, flange, *, now_s, clock_domain, camera_timestamp_s=None):
        """Return camera/tool chains only with an explicit synchronized camera time."""
        if self.invalid_reason or self.base_to_w0 is None or not self.mount_valid:
            raise ValueError(self.invalid_reason or 'installation_invalid')
        self._pose(flange, camera_timestamp_s)
        if (clock_domain != self.clock_domain or type(now_s) not in (int, float) or not np.isfinite(now_s)
                or not 0 <= now_s-flange.timestamp_s <= self.max_age_s):
            raise ValueError('Pose is stale or from another clock domain')
        return dict(camera=compose(self.base_to_w0, flange, self.hand_eye),
                    tool=compose(self.base_to_w0, flange, self.tool_mount))

    def invalidate(self, reason, *, mount=False):
        self.invalid_reason = reason
        self.base_to_w0 = None
        self.requires_new_session = True
        self.mount_valid = self.mount_valid and not mount
        self.generation += 1
