"""Physician selection intent on the fixed reference image, kept apart from mesh support.

The selected pixel mask is the authority for what the physician asked to cover. Triangle
indices are only the derived geometry binding: they cannot express holes without mesh,
missing depth or several separately selected areas. Three pixel states exist inside the
image: not selected (holes, painted forbidden, never painted), selected with measured
mesh support, and selected without reliable geometry ("几何待确认 / 需要补采"). Missing
geometry keeps the coverage requirement; it never becomes "unselected".

Nothing here reads or writes RGB, depth, meshes, poses or W0, and nothing here creates XYZ.
"""
from __future__ import annotations
from typing import Any

import hashlib

import cv2
import numpy as np


INTENT_SCHEMA = 'femto-selection-intent-v1'
SUPPORT_SCHEMA = 'femto-selection-support-v1'
IMAGE_FRAME = 'undistorted_reference_rgb'
SELECTION_RULE = 'visible_triangle_centroid_pixel_in_selection_mask'
CONNECTIVITY = '4_connected_selected_pixels_8_connected_unselected_pixels'
MISSING_GEOMETRY_POLICY = 'retain_coverage_requirement_not_unselected'
SELECTION_METHODS = ('work_only', 'work_minus_forbidden', 'full_minus_forbidden')
SUPPORT_CLASSES = {
    0: 'unselected',
    1: 'measured_mesh',
    2: 'no_surface_reference_depth_present',
    3: 'no_surface_no_reference_depth',
    4: 'outside_depth_range',
    5: 'depth_conflict',
    6: 'no_source_pixel',
}
UNCONFIRMED_CLASSES = (2, 3, 4, 5, 6)
SUPPORT_SOURCE = 'bound_mesh_first_hit_and_reference_raw_depth'
SUPPORT_EVIDENCE = 'reference_frame_raw_depth_only_not_multiview_cloud'
SUPPORT_INTERPRETATION = 'image_pixel_classification_not_surface_area'
DERIVED_SURFACE_STATUS = 'not_generated'
DERIVED_SURFACE_REASON = 'pending_user_confirmed_method_and_thresholds'
CLASS_MEANING = {
    'measured_mesh': '有实测 Mesh 支持',
    'no_surface_reference_depth_present': '参考帧有深度但 Mesh 缺面（派生表面待用户确认方法与门限）',
    'no_surface_no_reference_depth': '无 Mesh 且参考帧无深度（需要补采）',
    'outside_depth_range': '表面超出有效深度范围（需要补采）',
    'depth_conflict': 'Mesh 与参考深度不一致（深度歧义，需要补采）',
    'no_source_pixel': '去畸变后无有效原始像素（需要补采）',
}
MAX_PIXELS = 8_388_608
MAX_RUNS = 2_000_000
MAX_COMPONENTS = 4096
MAX_HOLES = 4096


def _image_size(value: Any) -> tuple[int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(v) is not int or v <= 0 for v in value) or value[0] * value[1] > MAX_PIXELS):
        raise ValueError('Invalid selection image size')
    return int(value[0]), int(value[1])


def encode_runs(array: Any) -> list[int]:
    """Row-major value/run pairs `[value, count, value, count, ...]`; exact and compact."""
    flat = np.ascontiguousarray(array).reshape(-1)
    if flat.size == 0:
        raise ValueError('Cannot encode an empty mask')
    if flat.dtype == bool:
        flat = flat.astype(np.uint8)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, flat.size]
    if len(starts) > MAX_RUNS:
        raise ValueError('Selection mask is too fragmented to record')
    runs = np.c_[flat[starts].astype(np.int64), (ends - starts).astype(np.int64)].reshape(-1)
    return [int(v) for v in runs]


def decode_runs(runs: Any, image_size: Any, *, values: Any) -> np.ndarray:
    width, height = _image_size(image_size)
    if not isinstance(runs, list) or not runs or len(runs) % 2 or len(runs) // 2 > MAX_RUNS:
        raise ValueError('Invalid selection run-length record')
    if any(type(v) is not int for v in runs):
        raise ValueError('Selection runs must be integers')
    labels = np.asarray(runs[0::2], dtype=np.int64)
    counts = np.asarray(runs[1::2], dtype=np.int64)
    allowed = np.asarray(sorted(set(int(v) for v in values)), dtype=np.int64)
    if np.any(counts <= 0) or not np.isin(labels, allowed).all() or int(counts.sum()) != width * height:
        raise ValueError('Selection runs do not cover the reference image exactly')
    return np.repeat(labels.astype(np.int32), counts).reshape(height, width)


def digest(array: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array).astype(np.uint8)).tobytes()).hexdigest()


def _as_mask(mask: Any, image_size: Any = None) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0 or array.size > MAX_PIXELS:
        raise ValueError('Selection mask must be a bounded two-dimensional image')
    if image_size is not None:
        width, height = _image_size(image_size)
        if array.shape != (height, width):
            raise ValueError('Selection mask does not match the reference image size')
    return array.astype(bool)


def label_components(mask: Any) -> tuple[np.ndarray, int]:
    """4-connected selected components, numbered 1.. by first raster occurrence (0 = unselected)."""
    mask = _as_mask(mask)
    count, raw = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    if count <= 1:
        return np.zeros(mask.shape, np.int32), 0
    flat = raw.reshape(-1)
    present, first = np.unique(flat[flat > 0], return_index=True)
    order = present[np.argsort(first, kind='stable')]
    mapping = np.zeros(count, np.int32)
    mapping[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return mapping[raw], int(len(order))


def _bbox(pixels_y: np.ndarray, pixels_x: np.ndarray) -> list[int]:
    return [int(pixels_x.min()), int(pixels_y.min()), int(pixels_x.max()), int(pixels_y.max())]


def describe_selection(mask: Any, forbidden_mask: Any = None) -> dict[str, Any]:
    """Components, explicit unselected holes and their enclosing component; pure image topology."""
    mask = _as_mask(mask)
    forbidden = np.zeros(mask.shape, bool) if forbidden_mask is None else _as_mask(forbidden_mask, [mask.shape[1], mask.shape[0]])
    labels, count = label_components(mask)
    if count > MAX_COMPONENTS:
        raise ValueError(f'Selection has more than {MAX_COMPONENTS} separate components')
    components = []
    for index in range(1, count + 1):
        ys, xs = np.nonzero(labels == index)
        components.append(dict(component_id=index - 1, pixel_count=int(len(xs)), bbox=_bbox(ys, xs), hole_ids=[]))
    holes: list[dict[str, Any]] = []
    if count:
        bg_count, background = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=8)
        border = np.unique(np.concatenate((background[0], background[-1], background[:, 0], background[:, -1])))
        enclosed = [int(v) for v in range(1, bg_count) if v not in set(int(b) for b in border)]
        if len(enclosed) > MAX_HOLES:
            raise ValueError(f'Selection has more than {MAX_HOLES} enclosed unselected holes')
        if enclosed:
            # Unselected pixel -> 4-adjacent selected component, vectorised over the whole image.
            pairs = []
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                shifted = np.roll(labels, (dy, dx), axis=(0, 1))
                if dy == 1:
                    shifted[0, :] = 0
                elif dy == -1:
                    shifted[-1, :] = 0
                if dx == 1:
                    shifted[:, 0] = 0
                elif dx == -1:
                    shifted[:, -1] = 0
                touching = (background > 0) & (shifted > 0)
                pairs.append(np.c_[background[touching], shifted[touching]])
            adjacency = np.unique(np.concatenate(pairs), axis=0) if pairs else np.zeros((0, 2), np.int64)
            for hole_index, bg_label in enumerate(enclosed):
                parents = sorted(int(v) for v in adjacency[adjacency[:, 0] == bg_label, 1])
                if len(parents) != 1:
                    raise ValueError('Selection hole enclosure is ambiguous')
                ys, xs = np.nonzero(background == bg_label)
                record = dict(hole_id=hole_index, component_id=parents[0] - 1, pixel_count=int(len(xs)),
                              bbox=_bbox(ys, xs), painted_forbidden_pixel_count=int(forbidden[ys, xs].sum()))
                holes.append(record)
                components[parents[0] - 1]['hole_ids'].append(hole_index)
    return dict(components=components, holes=holes, selected_pixel_count=int(mask.sum()),
                painted_forbidden_pixel_count=int(forbidden.sum()))


def build_selection_intent(mask: Any, *, forbidden_mask: Any = None, method: str,
                           source: str = 'physician_square_brush') -> dict[str, Any]:
    mask = _as_mask(mask)
    if method not in SELECTION_METHODS:
        raise ValueError('Unknown selection method')
    if not mask.any():
        raise ValueError('Selection intent needs at least one selected pixel')
    height, width = mask.shape
    forbidden = None if forbidden_mask is None else _as_mask(forbidden_mask, [width, height])
    topology = describe_selection(mask, forbidden)
    record = dict(schema=INTENT_SCHEMA, coordinate_frame=IMAGE_FRAME, unit='pixel', image_size=[width, height],
                  source=source, selection_method=method, connectivity=CONNECTIVITY,
                  selected_mask_runs=encode_runs(mask), selected_mask_sha256=digest(mask),
                  selected_pixel_count=topology['selected_pixel_count'],
                  painted_forbidden_runs=None if forbidden is None else encode_runs(forbidden),
                  painted_forbidden_sha256=None if forbidden is None else digest(forbidden),
                  painted_forbidden_pixel_count=topology['painted_forbidden_pixel_count'],
                  components=topology['components'], explicit_unselected_holes=topology['holes'],
                  missing_geometry_policy=MISSING_GEOMETRY_POLICY,
                  semantics=dict(selected='physician_requested_coverage',
                                 unselected='explicitly_not_selected_including_holes_and_painted_forbidden',
                                 missing_geometry='data_quality_problem_keeps_coverage_requirement',
                                 work_blocks='one_block_per_4_connected_selected_component'),
                  authority='selected_mask_is_authoritative_triangle_indices_are_derived_geometry_binding')
    return record


def validate_selection_intent(value: Any, image_size: Any = None) -> dict[str, Any]:
    """Reject anything that is not an internally consistent intent record; recompute derived topology."""
    if not isinstance(value, dict) or value.get('schema') != INTENT_SCHEMA:
        raise ValueError('Invalid selection intent record')
    if value.get('coordinate_frame') != IMAGE_FRAME or value.get('unit') != 'pixel':
        raise ValueError('Selection intent image coordinate contract mismatch')
    size = list(_image_size(value.get('image_size')))
    if image_size is not None and list(_image_size(image_size)) != size:
        raise ValueError('Selection intent image size does not match the reference image')
    if value.get('selection_method') not in SELECTION_METHODS:
        raise ValueError('Unknown selection method in selection intent')
    if value.get('missing_geometry_policy') != MISSING_GEOMETRY_POLICY:
        raise ValueError('Unsupported missing geometry policy')
    mask = decode_runs(value.get('selected_mask_runs'), size, values=(0, 1)).astype(bool)
    if not mask.any() or value.get('selected_mask_sha256') != digest(mask):
        raise ValueError('Selection intent mask digest mismatch')
    forbidden = None
    if value.get('painted_forbidden_runs') is not None:
        forbidden = decode_runs(value['painted_forbidden_runs'], size, values=(0, 1)).astype(bool)
        if value.get('painted_forbidden_sha256') != digest(forbidden):
            raise ValueError('Selection intent forbidden mask digest mismatch')
    elif value.get('painted_forbidden_sha256') is not None:
        raise ValueError('Selection intent forbidden mask digest without a mask')
    topology = describe_selection(mask, forbidden)
    if (value.get('selected_pixel_count') != topology['selected_pixel_count']
            or value.get('painted_forbidden_pixel_count') != topology['painted_forbidden_pixel_count']
            or value.get('components') != topology['components']
            or value.get('explicit_unselected_holes') != topology['holes']):
        raise ValueError('Selection intent components/holes do not match its mask')
    return dict(value, image_size=size, components=topology['components'],
                explicit_unselected_holes=topology['holes'])


def intent_mask(intent: dict[str, Any]) -> np.ndarray:
    return decode_runs(intent['selected_mask_runs'], intent['image_size'], values=(0, 1)).astype(bool)


def intent_forbidden_mask(intent: dict[str, Any]) -> np.ndarray | None:
    if intent.get('painted_forbidden_runs') is None:
        return None
    return decode_runs(intent['painted_forbidden_runs'], intent['image_size'], values=(0, 1)).astype(bool)


def component_labels(intent: dict[str, Any]) -> np.ndarray:
    """0 = unselected, k+1 = selection component k; same numbering as `components[].component_id`."""
    labels, count = label_components(intent_mask(intent))
    if count != len(intent['components']):
        raise ValueError('Selection intent component count mismatch')
    return labels


def pixel_indices(uv: Any, image_size: Any) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV integer pixel of projected coordinates and whether it lies inside the image."""
    width, height = _image_size(image_size)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1)
    pix = np.zeros(uv.shape, np.int64)
    pix[finite] = np.floor(uv[finite] + .5).astype(np.int64)
    inside = finite & (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    return pix, inside


def components_of_pixels(labels: np.ndarray, uv: Any) -> np.ndarray:
    """Selection component id (0-based) per projected point; -1 when unselected or outside."""
    height, width = labels.shape
    pix, inside = pixel_indices(uv, [width, height])
    result = np.full(len(pix), -1, np.int64)
    result[inside] = labels[pix[inside, 1], pix[inside, 0]].astype(np.int64) - 1
    return result


def build_selection_support(mask: Any, projection: Any) -> dict[str, Any]:
    """Classify every selected pixel from the bound mesh first hit and the reference raw depth.

    Evidence is the single reference frame only; the fused multi-view cloud is not bound
    here. Classes describe why XYZ is or is not available; they never invent XYZ.
    """
    mask = _as_mask(mask)
    reasons = np.asarray(projection.reasons)
    depth = np.asarray(projection.depth)
    if reasons.shape != mask.shape or depth.shape != mask.shape:
        raise ValueError('Selection mask does not match the projection image')
    minimum, maximum = float(projection.minimum), float(projection.maximum)
    depth_present = (depth >= minimum * 1000) & (depth <= maximum * 1000)
    classes = np.where(reasons == 0, 1,
                       np.where(reasons == 1, np.where(depth_present, 2, 3),
                                np.where(reasons == 2, 4, np.where(reasons == 3, 5, 6)))).astype(np.uint8)
    image_valid = getattr(projection, 'image_valid', None)
    if image_valid is not None:
        classes[~np.asarray(image_valid, bool)] = 6
    classes[~mask] = 0
    counts = {name: int((classes == code).sum()) for code, name in SUPPORT_CLASSES.items() if code}
    unconfirmed = int(sum(counts[SUPPORT_CLASSES[code]] for code in UNCONFIRMED_CLASSES))
    height, width = mask.shape
    return dict(schema=SUPPORT_SCHEMA, coordinate_frame=IMAGE_FRAME, unit='pixel', image_size=[width, height],
                source=SUPPORT_SOURCE, evidence=SUPPORT_EVIDENCE,
                classes={str(code): name for code, name in SUPPORT_CLASSES.items()},
                class_runs=encode_runs(classes), class_sha256=digest(classes),
                counts=counts, selected_pixel_count=int(mask.sum()),
                geometry_supported_pixel_count=counts['measured_mesh'],
                geometry_unconfirmed_pixel_count=unconfirmed,
                depth_range_m=_depth_range(minimum, maximum),
                interpretation=SUPPORT_INTERPRETATION,
                derived_surface=dict(status=DERIVED_SURFACE_STATUS, reason=DERIVED_SURFACE_REASON,
                                     candidate_pixel_count=counts['no_surface_reference_depth_present']),
                physical_accuracy_verified=False)


def _depth_range(minimum: Any, maximum: Any) -> list[float]:
    values = [minimum, maximum]
    if (len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values)
            or not all(np.isfinite(float(v)) for v in values) or not 0 < float(minimum) < float(maximum)):
        raise ValueError('Invalid selection support depth range')
    return [float(minimum), float(maximum)]


def validate_selection_support(value: Any, mask: Any, *, expected_depth_range_m: Any = None) -> dict[str, Any]:
    """Recompute the classification and lock every field the classes were derived from.

    `depth_range_m`, `source`, `evidence`, the class mapping, the interpretation and the
    `derived_surface` state are inputs or contracts of the record, not display text: a
    changed depth range would silently change what downstream candidate generation does,
    and `derived_surface.status` must stay `not_generated` until the user decides admission.
    Callers that know the range the bound dataset was reconstructed with pass it as
    `expected_depth_range_m`; the record alone cannot tell a wrong-but-plausible range."""
    if not isinstance(value, dict) or value.get('schema') != SUPPORT_SCHEMA:
        raise ValueError('Invalid selection support record')
    if value.get('coordinate_frame') != IMAGE_FRAME or value.get('unit') != 'pixel':
        raise ValueError('Selection support image coordinate contract mismatch')
    if (value.get('source') != SUPPORT_SOURCE or value.get('evidence') != SUPPORT_EVIDENCE
            or value.get('interpretation') != SUPPORT_INTERPRETATION
            or value.get('classes') != {str(code): name for code, name in SUPPORT_CLASSES.items()}):
        raise ValueError('Selection support source / evidence / class contract mismatch')
    depth_range = value.get('depth_range_m')
    if not isinstance(depth_range, list) or len(depth_range) != 2:
        raise ValueError('Invalid selection support depth range')
    depth_range = _depth_range(*depth_range)
    if expected_depth_range_m is not None and not np.allclose(
            _depth_range(*expected_depth_range_m), depth_range, rtol=0, atol=1e-9):
        raise ValueError('Selection support depth range does not match the bound dataset depth range')
    mask = _as_mask(mask, value.get('image_size'))
    classes = decode_runs(value.get('class_runs'), value['image_size'], values=SUPPORT_CLASSES.keys())
    if value.get('class_sha256') != digest(classes) or not np.array_equal(classes > 0, mask):
        raise ValueError('Selection support classes do not match the selection mask')
    counts = {name: int((classes == code).sum()) for code, name in SUPPORT_CLASSES.items() if code}
    unconfirmed = int(sum(counts[SUPPORT_CLASSES[code]] for code in UNCONFIRMED_CLASSES))
    if (value.get('counts') != counts or value.get('geometry_supported_pixel_count') != counts['measured_mesh']
            or value.get('geometry_unconfirmed_pixel_count') != unconfirmed
            or value.get('selected_pixel_count') != int(mask.sum())
            or value.get('physical_accuracy_verified') is not False):
        raise ValueError('Selection support counts do not match its classes')
    if value.get('derived_surface') != dict(status=DERIVED_SURFACE_STATUS, reason=DERIVED_SURFACE_REASON,
                                            candidate_pixel_count=counts['no_surface_reference_depth_present']):
        raise ValueError('Selection support derived_surface state is not the locked not_generated record')
    return dict(value)


def support_classes(support: dict[str, Any]) -> np.ndarray:
    return decode_runs(support['class_runs'], support['image_size'], values=SUPPORT_CLASSES.keys()).astype(np.uint8)


def restrict_selection(region: dict[str, Any], removed_masks: Any) -> dict[str, Any]:
    """Return region copies of intent/support with extra unselected pixels; never adds pixels."""
    intent = validate_selection_intent(region['selection_intent'])
    mask = intent_mask(intent)
    forbidden = intent_forbidden_mask(intent)
    removed = np.zeros(mask.shape, bool)
    for item in removed_masks:
        removed |= _as_mask(item, intent['image_size'])
    mask &= ~removed
    # A later exclusion is more painted-forbidden intent, whether or not a forbidden mask existed.
    forbidden = removed.copy() if forbidden is None else (forbidden | removed)
    result = dict(region, selection_intent=build_selection_intent(
        mask, forbidden_mask=forbidden, method=intent['selection_method'], source=intent['source']))
    if region.get('selection_support') is not None:
        classes = support_classes(region['selection_support'])
        classes[~mask] = 0
        support = dict(region['selection_support'])
        counts = {name: int((classes == code).sum()) for code, name in SUPPORT_CLASSES.items() if code}
        support.update(class_runs=encode_runs(classes), class_sha256=digest(classes), counts=counts,
                       selected_pixel_count=int(mask.sum()), geometry_supported_pixel_count=counts['measured_mesh'],
                       geometry_unconfirmed_pixel_count=int(sum(counts[SUPPORT_CLASSES[c]] for c in UNCONFIRMED_CLASSES)))
        support['derived_surface'] = dict(support.get('derived_surface', {}),
                                          candidate_pixel_count=counts['no_surface_reference_depth_present'])
        result['selection_support'] = support
    return result


def component_support_counts(support: dict[str, Any] | None, labels: np.ndarray, component_id: int) -> dict[str, Any] | None:
    """Pixel classes of one selection component; None when no support record exists."""
    if support is None:
        return None
    classes = support_classes(support)
    inside = labels == component_id + 1
    counts = {name: int(((classes == code) & inside).sum()) for code, name in SUPPORT_CLASSES.items() if code}
    return dict(selected_pixels=int(inside.sum()), geometry_supported_pixels=counts['measured_mesh'],
                geometry_unconfirmed_pixels=int(sum(counts[SUPPORT_CLASSES[c]] for c in UNCONFIRMED_CLASSES)),
                unconfirmed_by_reason={name: counts[name] for code, name in SUPPORT_CLASSES.items() if code in UNCONFIRMED_CLASSES},
                interpretation='image_pixel_counts_not_surface_area')


def selection_layers(region: Any) -> dict[str, Any] | None:
    """Display layers for previews: component labels, holes, painted forbidden, support classes."""
    if not isinstance(region, dict) or not isinstance(region.get('selection_intent'), dict):
        return None
    intent = region['selection_intent']
    try:
        mask = intent_mask(intent)
        labels, _ = label_components(mask)
        forbidden = intent_forbidden_mask(intent)
        classes = None
        if isinstance(region.get('selection_support'), dict):
            classes = support_classes(region['selection_support'])
        holes = np.zeros(mask.shape, bool)
        if intent.get('explicit_unselected_holes'):
            bg_count, background = cv2.connectedComponents((~mask).astype(np.uint8), connectivity=8)
            border = set(int(v) for v in np.unique(np.concatenate((background[0], background[-1], background[:, 0], background[:, -1]))))
            holes = (background > 0) & ~np.isin(background, sorted(border))
    except (ValueError, KeyError, TypeError):
        return None
    return dict(mask=mask, labels=labels, holes=holes, forbidden=forbidden, classes=classes,
                components=intent.get('components', []), image_size=intent['image_size'])


def tint_selection(rgb: np.ndarray, layers: dict[str, Any]) -> np.ndarray:
    """Return a tinted copy: red = explicit unselected inside/forbidden, violet/orange = geometry to confirm."""
    image = np.asarray(rgb).astype(np.float32).copy()
    if image.shape[:2] != tuple(layers['mask'].shape):
        return np.asarray(rgb).copy()
    unselected = layers['holes'].copy()
    if layers['forbidden'] is not None:
        unselected |= layers['forbidden']
    image[unselected] = image[unselected] * .45 + np.array([200, 40, 60], np.float32) * .55
    classes = layers['classes']
    if classes is not None:
        derived = classes == 2
        missing = np.isin(classes, [3, 4, 5, 6])
        image[derived] = image[derived] * .4 + np.array([170, 90, 230], np.float32) * .6
        image[missing] = image[missing] * .4 + np.array([245, 150, 40], np.float32) * .6
    return np.clip(image, 0, 255).astype(np.uint8)


def selection_contours(layers: dict[str, Any]) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Outlines (component id, Nx2 pixel polyline) for component boundaries and for holes."""
    result: dict[str, list[tuple[int, np.ndarray]]] = {'components': [], 'holes': []}
    labels = layers['labels']
    for component in layers['components']:
        piece = (labels == component['component_id'] + 1).astype(np.uint8)
        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 2:
                result['components'].append((component['component_id'], contour.reshape(-1, 2).astype(np.float64)))
    if layers['holes'].any():
        contours, _ = cv2.findContours(layers['holes'].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for index, contour in enumerate(contours):
            if len(contour) >= 2:
                result['holes'].append((index, contour.reshape(-1, 2).astype(np.float64)))
    return result
