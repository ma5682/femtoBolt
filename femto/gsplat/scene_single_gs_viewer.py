"""Fixed optical centre and orientation; zoom changes only the image crop/K."""
from __future__ import annotations
from typing import Any

from femto.paths import REPO_ROOT
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import numpy as np
from femto.io_utils import load_json, write_json, write_json_strict
from femto.gsplat.scene_single_gs import SCHEMA, MARKER, SingleConfig
from femto.ui_i18n import pick


def enable_native_dpi() -> None:
    """Opt this viewer process into physical pixels before creating windows."""
    if os.name != 'nt': return
    import ctypes
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError,OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError,OSError):
        pass


def native_render_size(width: int, height: int) -> tuple[Any, Any]:
    """Use source detail even in a small window, independently of UI scaling."""
    if width <= 0 or height <= 0: raise ValueError('Invalid source dimensions')
    scale=min(1.,1920/width,1080/height)
    return max(1,round(width*scale)),max(1,round(height*scale))


@dataclass
class FixedViewport:
    zoom: float = 1.
    cx: float = .5
    cy: float = .5

    def constrain(self) -> None:
        if not np.isfinite([self.zoom,self.cx,self.cy]).all(): raise ValueError('Invalid viewport')
        self.zoom=float(np.clip(self.zoom,1,16))
        radius=.5/self.zoom
        self.cx=float(np.clip(self.cx,radius,1-radius))
        self.cy=float(np.clip(self.cy,radius,1-radius))

    def source_point(self, x: Any, y: Any) -> tuple[Any, Any]:
        return self.cx+(x-.5)/self.zoom, self.cy+(y-.5)/self.zoom

    def zoom_at(self, factor: float, x: float=.5, y: float=.5) -> None:
        sx,sy=self.source_point(x,y)
        self.zoom=float(np.clip(self.zoom*factor,1,16))
        self.cx=sx-(x-.5)/self.zoom; self.cy=sy-(y-.5)/self.zoom
        self.constrain()

    def select(self,x0: Any,y0: Any,x1: Any,y1: Any) -> None:
        x0,x1=sorted(np.clip([x0,x1],0,1)); y0,y1=sorted(np.clip([y0,y1],0,1))
        if min(x1-x0,y1-y0)<.01: return
        self.cx,self.cy=self.source_point((x0+x1)/2,(y0+y1)/2)
        self.zoom=min(16,self.zoom/max(x1-x0,y1-y0))
        self.constrain()

    def intrinsic(self, K: Any, source_width: int, source_height: int, width: int, height: int) -> Any:
        self.constrain()
        result=np.array(K,dtype=np.float32).copy()
        result[0,2]-=(self.cx-.5/self.zoom)*source_width
        result[1,2]-=(self.cy-.5/self.zoom)*source_height
        result[0]*=width*self.zoom/source_width
        result[1]*=height*self.zoom/source_height
        return result


def load_model_metadata(output: str | Path) -> Any:
    output=Path(output)
    meta=load_json(output/'single_report.json')
    if meta.get('schema')!=SCHEMA or not meta.get('completed') or not meta.get('fixed_view'):
        raise ValueError('Select a completed single-frame fixed-view model')
    model=output/'model.ply'
    if hashlib.sha256(model.read_bytes()).hexdigest()!=meta['model_sha256']:
        raise ValueError('Model fingerprint mismatch')
    return meta


class ViewportModelMismatch(ValueError):
    pass


def load_viewport(path: Any, meta: Any) -> Any:
    saved=load_json(path)
    if not isinstance(saved,dict): raise ValueError('Invalid saved region')
    if saved.get('model_sha256')!=meta['model_sha256'] or saved.get('source_fingerprint')!=meta['source_fingerprint']:
        raise ViewportModelMismatch('Saved region belongs to a different source/model')
    values=saved['viewport']
    if not isinstance(values,dict) or set(values)!={'zoom','cx','cy'}:
        raise ValueError('Invalid saved viewport fields')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in values.values()):
        raise ValueError('Invalid viewport number')
    view=FixedViewport(**values)
    if not 1<=view.zoom<=16 or not .5/view.zoom<=view.cx<=1-.5/view.zoom or not .5/view.zoom<=view.cy<=1-.5/view.zoom:
        raise ValueError('Saved viewport is outside the source image')
    return view


class SingleViewer:
    def __init__(self, output, language='zh'):
        enable_native_dpi()
        from PySide6.QtCore import QRect, Qt, QTimer
        from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton, QRubberBand, QVBoxLayout, QWidget
        from PIL import Image
        self.messagebox=QMessageBox; self.Image=Image
        self.app=QApplication.instance() or QApplication([])
        self.output=Path(output).resolve(); self.meta=load_model_metadata(self.output)
        self.language=language; self.view=FixedViewport(); self.start=None; self.box=None
        self.pool=ThreadPoolExecutor(max_workers=1); self.future=None; self.dirty=True; self.closed=False
        self.renderer=None; self.image=None; self.photo=None
        owner=self
        class ViewerWindow(QWidget):
            def closeEvent(self, event):
                owner._shutdown()
                super().closeEvent(event)

        class RenderCanvas(QLabel):
            def resizeEvent(self,event):
                owner.request()
                super().resizeEvent(event)

            def mousePressEvent(self,event):
                if event.button()==Qt.MouseButton.LeftButton: owner.press(event)

            def mouseMoveEvent(self,event):
                if event.buttons() & Qt.MouseButton.LeftButton: owner.drag(event)

            def mouseReleaseEvent(self,event):
                if event.button()==Qt.MouseButton.LeftButton: owner.release(event)

            def wheelEvent(self,event):
                owner.wheel(event); event.accept()

        self.root=ViewerWindow()
        self.root.setWindowTitle(pick(self.language, '单帧 3DGS · 固定视角','Single-frame 3DGS · Fixed view')+' | '+Path(self.meta.get('source_model') or self.meta['source_photo']).name)
        self.root.resize(1100,730)
        layout=QVBoxLayout(self.root)
        toolbar=QHBoxLayout(); layout.addLayout(toolbar)
        for zh,en,action in [('放大','Zoom in',lambda:self.zoom(1.25)),('缩小','Zoom out',lambda:self.zoom(.8)),
                             ('复位','Reset',self.reset),('保存区域','Save region',self.save),('回读区域','Load region',self.load)]:
            button=QPushButton(pick(self.language,zh,en)); button.clicked.connect(action); toolbar.addWidget(button)
        toolbar.addStretch()
        hint=QLabel(pick(self.language, '滚轮缩放；左键框选放大。视角锁定，仅重建可见表面；尺度未经过实物验证。',
            'Wheel: zoom; left drag: zoom to region. Fixed view; observed surface only; scale unvalidated.'))
        hint.setWordWrap(True); layout.addWidget(hint)
        self.canvas=RenderCanvas(); self.canvas.setStyleSheet('background: #101720;')
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from PySide6.QtWidgets import QSizePolicy
        self.canvas.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        layout.addWidget(self.canvas,1)
        self.box=QRubberBand(QRubberBand.Shape.Rectangle,self.canvas)
        self._rect_type=QRect
        self.status=QLabel(str(self.output)); layout.addWidget(self.status)
        self.timer=QTimer(self.root); self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll); self.timer.start()

    def request(self) -> None: self.dirty=True
    def reset(self) -> None: self.view=FixedViewport(); self.request()
    def zoom(self,factor: float,x: float=.5,y: float=.5) -> None: self.view.zoom_at(factor,x,y); self.request()

    def display_rect(self) -> tuple[Any, Any, Any, Any]:
        cw,ch=max(1,self.canvas.width()),max(1,self.canvas.height())
        scale=min(cw/self.meta['width'],ch/self.meta['height'])
        w,h=max(1,round(scale*self.meta['width'])),max(1,round(scale*self.meta['height']))
        return (cw-w)/2,(ch-h)/2,w,h

    def normalized(self,event: Any) -> tuple[Any, Any]:
        x,y,w,h=self.display_rect()
        return (event.position().x()-x)/w,(event.position().y()-y)/h

    def press(self,event: Any) -> None:
        if self.future is not None or self.dirty: return
        x,y=self.normalized(event)
        if 0<=x<=1 and 0<=y<=1:
            self.start=(x,y,event.position().toPoint())
            self.box.setGeometry(self._rect_type(self.start[2],self.start[2]))
            self.box.show()

    def drag(self,event: Any) -> None:
        if self.start is None: return
        self.box.setGeometry(self._rect_type(self.start[2],event.position().toPoint()).normalized())

    def release(self,event: Any) -> None:
        if self.start is not None:
            self.view.select(*self.start[:2],*self.normalized(event)); self.request()
        self.start=None
        self.box.hide()

    def wheel(self,event: Any) -> None:
        if self.future is not None or self.dirty: return
        x,y=self.normalized(event)
        delta=event.angleDelta().y()
        if 0<=x<=1 and 0<=y<=1 and delta:
            self.zoom(1.2 if delta>0 else 1/1.2,x,y)

    def save(self) -> None:
        temporary=None
        parent = getattr(self, 'root', None)
        try:
            payload=dict(model_sha256=self.meta['model_sha256'],source_fingerprint=self.meta['source_fingerprint'],
                         viewport=vars(self.view),source_photo=self.meta['source_photo'])
            write_json_strict(
                self.output/'viewport.json', payload, durable=True, temporary_prefix='viewport_')
            self.status.setText(pick(self.language, '当前区域已保存，可随时回读','Current region saved and ready to load'))
        except (OSError,ValueError) as error:
            self.messagebox.critical(parent,'3DGS',pick(self.language, '区域保存失败，请检查模型目录是否可写。','Could not save region. Check that the model folder is writable.'))
        finally:
            if temporary is not None:
                try: temporary.unlink(missing_ok=True)
                except OSError: pass

    def load(self) -> None:
        parent = getattr(self, 'root', None)
        try:
            loaded=load_viewport(self.output/'viewport.json',self.meta)
        except FileNotFoundError:
            self.messagebox.information(parent,'3DGS',pick(self.language, '当前模型尚未保存区域。请先框选或缩放，再点击“保存区域”；以后可用“回读区域”恢复。新建模型不会自动继承旧模型的区域。',
                'No region has been saved for this model. Zoom or select a region, then click Save region. Load region restores it later. New models do not inherit old regions.'))
            return
        except ViewportModelMismatch:
            self.messagebox.warning(parent,'3DGS',pick(self.language, '保存的区域属于其他模型，无法在当前模型回读。请打开原模型，或在当前模型重新保存区域。',
                'This region belongs to another model. Open that model or save a new region in the current model.'))
            return
        except (ValueError,KeyError,TypeError,OverflowError):
            self.messagebox.critical(parent,'3DGS',pick(self.language, '区域文件内容损坏或格式不正确。当前视图未改变，请重新选择并保存区域。',
                'The region file is damaged or invalid. The current view is unchanged. Select and save the region again.'))
            return
        except OSError:
            self.messagebox.critical(parent,'3DGS',pick(self.language, '无法读取区域文件，请检查模型目录的访问权限。',
                'Cannot read the region file. Check access to the model folder.'))
            return
        self.view=loaded
        self.request()
        self.status.setText(pick(self.language, '已恢复保存的区域','Saved region restored'))

    def render(self,K: Any,w: int,h: int) -> Any:
        from femto.gsplat.scene_gsplat_guard import guard_cuda
        from femto.gsplat.scene_gsplat_viewer import TrainedGaussianRenderer
        guard_cuda('cuda:0',SingleConfig())
        if self.renderer is None:
            self.renderer=TrainedGaussianRenderer(self.output/'model.ply',initial_width=w,initial_height=h,max_render_pixels=1920*1080)
        return self.renderer.render(np.eye(4,dtype=np.float32),K,w,h)

    def poll(self) -> None:
        parent = getattr(self, 'root', None)
        from PySide6.QtGui import QImage, QPixmap
        if self.closed: return
        if self.future is not None and self.future.done():
            try:
                result=self.future.result()
                x,y,w,h=self.display_rect()
                self.image=self.Image.fromarray(result)
                pixels=np.ascontiguousarray(self.image.resize((w,h),self.Image.Resampling.LANCZOS))
                image=QImage(pixels.data,w,h,pixels.strides[0],QImage.Format.Format_RGB888).copy()
                self.photo=QPixmap.fromImage(image)
                self.canvas.setPixmap(self.photo)
                self.status.setText(f"{self.meta['gaussian_count']:,} Gaussian | {result.shape[1]}×{result.shape[0]} | {self.view.zoom:.2f}× | {self.output.name}")
            except Exception as error:
                self.status.setText(str(error)); self.dirty=False
                self.messagebox.critical(parent,'3DGS',str(error))
            self.future=None
        if self.dirty and self.future is None:
            self.dirty=False
            w,h=native_render_size(self.meta['width'],self.meta['height'])
            K=self.view.intrinsic(self.meta['K'],self.meta['width'],self.meta['height'],w,h)
            self.future=self.pool.submit(self.render,K,w,h)

    def _shutdown(self) -> None:
        if self.closed: return
        self.closed=True; self.timer.stop()
        self.pool.shutdown(wait=False,cancel_futures=True)

    def close(self) -> None:
        self._shutdown(); self.root.close()

    def run(self) -> None:
        self.root.show(); self.app.exec()


def open_viewer(output: str | Path,language: str='zh') -> Any:
    import subprocess, os
    from femto.gsplat.gsplat_backend import runtime_environment,runtime_python
    output=Path(output); load_model_metadata(output)
    with (output/'viewer_stdout.log').open('a',encoding='utf-8') as log:
        return subprocess.Popen([str(runtime_python()),'-m','femto.gsplat.scene_single_gs_viewer',str(output),'--lang',language],
            cwd=REPO_ROOT,env=runtime_environment(),stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)


if __name__=='__main__':
    from femto.paths import workspace_runtime_environment
    os.environ.update(workspace_runtime_environment())
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('output',nargs='?',type=Path)
    parser.add_argument('--lang',choices=['zh','en'],default='zh'); args=parser.parse_args()
    SingleViewer(args.output or Path(MARKER.read_text(encoding='utf-8').strip()),args.lang).run()
