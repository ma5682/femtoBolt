"""Read-only reference-pixel diagnosis; never changes geometry or saved support classes."""
import numpy as np

from femto.annotation.fixed_rgb_geometry import GeometryBudget
from femto.annotation.selection_intent import classify_selection_support


LABELS = {
    1: '网格支持且参考深度一致',
    2: '有参考深度，但无网格表面',
    3: '无网格表面且参考深度缺失',
    4: '网格表面超出有效深度范围',
    5: '参考深度与网格冲突',
    6: '无有效源图像像素',
    7: '有网格支持，但参考深度缺失或超范围',
}
COLORS = {1: (34, 211, 166, 95), 2: (173, 116, 240, 160),
          3: (255, 174, 66, 170), 4: (255, 174, 66, 170),
          5: (255, 174, 66, 170), 6: (255, 174, 66, 170),
          7: (173, 116, 240, 160)}
LIMITS = ('比例按选区图像像素计算，不是表面积或治疗覆盖率。'
          '标定、固定参考坐标与来源准入沿用真实记录加载器；'
          '此检查不能定位每个像素的跟踪或配准失败原因，也不证明实物精度。')


def assess_selection(model, work, forbidden, cancel=None, *, check=None):
    """Refine mesh-only support as class 7 without rewriting legacy records."""
    check = check or GeometryBudget(cancel).check
    check()
    result = dict(status='unavailable', classes=None, rgba=None, counts={},
                  selected=0, supported=0, pending=0, ratio=None, message='几何支持待检查')
    try:
        shape = model.rgb.shape[:2]
        work, forbidden = np.asarray(work), np.asarray(forbidden)
        if work.shape != shape or forbidden.shape != shape or work.ndim != 2:
            raise ValueError('选区与参考影像尺寸不一致')
        if not np.isin(work, [0, 1]).all() or not np.isin(forbidden, [0, 1]).all():
            raise ValueError('选区掩膜无效')
        mask = work.astype(bool) & ~forbidden.astype(bool)
        result['selected'] = int(mask.sum())
        if not mask.any():
            return dict(result, status='empty', message='尚无有效工作选区；请涂抹选入，历史禁区继续扣除。')
        projection = model.projection
        k = np.asarray(projection.k)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError('参考标定不可用')
        for name in ('reasons', 'depth', 'valid', 'new_support', 'z', 'ids'):
            if np.asarray(getattr(projection, name)).shape != shape:
                raise ValueError('投影证据尺寸不一致')
        if not np.isin(projection.reasons, range(5)).all():
            raise ValueError('投影原因未知')
        if not np.array_equal(projection.reasons == 0, projection.valid):
            raise ValueError('投影支持状态不一致')
        if not np.isfinite([projection.minimum, projection.maximum]).all() or not 0 < projection.minimum < projection.maximum:
            raise ValueError('有效深度范围不可用')
        if not getattr(model, 'view_reference', None):
            raise ValueError('记录缺少参考影像绑定')
        check()
        classes = classify_selection_support(mask, projection)
        depth = np.asarray(projection.depth)
        measured = np.isfinite(depth) & (depth >= projection.minimum * 1000) & (depth <= projection.maximum * 1000)
        classes[(classes == 1) & (~measured | projection.new_support)] = 7
        # Fail closed on inconsistent loaded evidence instead of displaying green.
        supported = classes == 1
        if np.any(supported & (~np.isfinite(projection.z) | (projection.ids < 0))):
            raise ValueError('网格交点证据不可用')
        check()
        counts = {code: int((classes == code).sum()) for code in LABELS}
        rgba = np.zeros((*shape, 4), np.uint8)
        for code, color in COLORS.items():
            check(); rgba[classes == code] = color
        good = counts[1]
        pending = result['selected'] - good
        return dict(result, status='checked', classes=classes, rgba=rgba, counts=counts,
                    supported=good, pending=pending, ratio=good/result['selected'],
                    message='选区存在缺失或待检查位置，请核对或补扫。' if pending else '选区像素均有网格支持且参考深度一致；仍需核对模型与路线。')
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        check()
        return dict(result, message='无法核验当前选区：'+str(error)+'；不能视为质量合格。')


def require_geometry(result):
    if result['status'] != 'checked' or not result['supported']:
        raise ValueError(result['message']+' 当前没有可核验的网格与参考深度一致范围，请检查数据或重新采集。')
