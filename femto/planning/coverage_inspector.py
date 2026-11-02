"""Read-only coverage layers and omission localization from a bound inspection."""
from __future__ import annotations
from typing import Any, Callable

from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LAYER_NAMES={'selection':'圈选范围','effective_geometry':'有效几何','forbidden':'禁区剔除','boundary_margin':'边界预留',
             'no_geometry':'无几何支持','selection_limit':'选区离散限制','covered':'采样覆盖','uncovered':'采样未覆盖'}
COLORS={'selection':(230,180,245),'effective_geometry':(80,160,175),'forbidden':(225,55,65),
        'boundary_margin':(75,125,255),'selection_limit':(175,150,65),'no_geometry':(180,65,205),'covered':(40,235,110),'uncovered':(255,150,40)}


def font(size: int=17) -> Any:
    path=Path('C:/Windows/Fonts/msyh.ttc')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default()


def crop_bounds(report: Any,item: Any) -> tuple[int, int, int, int]:
    width,height=report['reference_image_size']
    if item is None:return (0,0,width,height)
    x0,y0,x1,y1=item['bbox_pixel'];padding=max(18,.2*max(x1-x0,y1-y0))
    return (max(0,int(x0-padding)),max(0,int(y0-padding)),min(width,int(x1+padding)+1),min(height,int(y1+padding)+1))


def render_map(rgb: Any,plan: Any,enabled: Any=None,item: Any=None,zoom: bool=False,header: bool=True) -> Any:
    report=plan['coverage_inspection'];enabled=set(LAYER_NAMES) if enabled is None else set(enabled)
    base=(np.asarray(rgb)*.30).astype(np.uint8)
    # Pixel layers are display evidence only, never a filled surface coverage claim.
    for name in ('selection','effective_geometry','no_geometry','selection_limit','forbidden','boundary_margin'):
        if name not in enabled:continue
        alpha=.15 if name=='selection' else .65
        color=np.asarray(COLORS[name])
        for y,a,b in report['layers'][name]['pixel_runs']:base[y,a:b]=(base[y,a:b]*(1-alpha)+color*alpha).astype(np.uint8)
    image=Image.fromarray(base);draw=ImageDraw.Draw(image)
    for sample in report['samples']:
        if not sample['eligible']:continue
        key='covered' if sample['covered'] else 'uncovered'
        if key not in enabled:continue
        x,y=sample['pixel'];draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5),fill=COLORS[key])
    if item is not None:
        for index in item['sample_indices']:
            x,y=report['samples'][index]['pixel'];draw.ellipse((x-3,y-3,x+3,y+3),outline='white',width=2)
        for y,a,b in item['pixel_runs']:draw.line((a,y,b-1,y),fill='white',width=1)
        x0,y0,x1,y1=item['bbox_pixel'];draw.rectangle((x0-4,y0-4,x1+4,y1+4),outline='white',width=2)
    if zoom:image=image.crop(crop_bounds(report,item))
    if not header:return image
    # Keep legends legible even with a small input image.
    picture=image.copy()
    factor=min(1400/picture.width,900/picture.height,8 if zoom else 1)
    picture=picture.resize((max(1,int(picture.width*factor)),max(1,int(picture.height*factor))),Image.Resampling.NEAREST if factor>1 else Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(max(1100,picture.width),picture.height+180),(18,28,38));canvas.paste(picture,(0,180))
    draw=ImageDraw.Draw(canvas);f=font();cov=plan['coverage'];den=report['denominator']
    lines=['覆盖检查：离散采样与排除范围（研发） · '+plan['plan_binding']['algorithm_version'],
        f"分母：允许且可信的圈内采样代表面积 {den['sampled_allowed_supported_area_m2']*1e6:.2f} mm²；禁区和无支持不计入，边界预留仍计入。",
        f"采样覆盖 {cov['sampled_coverage_percent']:.2f}%；每面3点，面面积/3加权；非均匀密度，不保证完整曲面覆盖。",
        '无几何支持面积未知。禁区/边界为像素图层；覆盖只画采样点，不把整面染绿当作已覆盖。',
        '圈选淡紫 / 有效几何青 / 禁区红 / 预留蓝 / 无几何紫 / 离散限制棕 / 覆盖点绿 / 未覆盖点橙',
        '方案结果 '+plan['plan_binding']['result_sha256'][:20]+'… · 白色高亮为选中遗漏；选中项可缩放定位。']
    for i,line in enumerate(lines):draw.text((12,7+i*28),line,font=f,fill='#e4edf4')
    return canvas


def show_inspector(parent: Any, model: Any, plan: Any, is_current: Any = lambda: True) -> Any:
    from femto.planning.qt_reports import CoverageInspector
    return CoverageInspector(parent, model, plan, is_current)
