"""Four-mode offline Depth Debug Viewer for an immutable derived quality bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODES = ('Original Depth', 'Invalid Mask', 'Filled Depth', 'Confidence Map')
GREEN, YELLOW, RED = (0,220,0), (255,220,0), (230,30,30)


class DepthDebugData:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.report = json.loads((self.directory/'depth_quality.json').read_text(encoding='utf-8'))
        with Image.open(self.directory/'original_depth_mm.png') as image:
            self.original = np.asarray(image).copy()
        with Image.open(self.directory/'filled_depth_mm.png') as image:
            self.filled = np.asarray(image).copy()
        with Image.open(self.directory/'invalid_mask.png') as image:
            self.invalid = np.asarray(image)>0
        self.confidence = np.load(self.directory/'confidence.npy',allow_pickle=False)
        if self.original.ndim != 2 or self.original.size == 0 or any(
                data.shape != self.original.shape for data in (self.filled,self.invalid,self.confidence)):
            raise ValueError('Debug bundle has inconsistent pixel grids')
        if not np.isin(self.confidence,np.array([0,.3,1],np.float32)).all():
            raise ValueError('Debug bundle has invalid confidence')

    def render(self, mode: str) -> Image.Image:
        if mode not in MODES:
            raise ValueError('Unknown debug mode')
        original = ~self.invalid
        repaired = self.confidence == np.float32(.3)
        rgb = np.full((*self.original.shape,3),RED,np.uint8)
        rgb[original] = GREEN
        if mode in ('Filled Depth','Confidence Map'):
            rgb[repaired] = YELLOW
        if mode in ('Original Depth','Filled Depth'):
            depth = self.original if mode == 'Original Depth' else self.filled
            low, high = self.report['active_range_mm']
            intensity = .3+.7*np.clip((depth.astype(float)-low)/(high-low),0,1)
            known = original if mode == 'Original Depth' else self.confidence>0
            rgb[known] = np.rint(rgb[known]*intensity[known,None]).astype(np.uint8)
        return Image.fromarray(rgb)

    def describe_pixel(self, x: int, y: int) -> str:
        original = int(self.original[y,x])
        filled = int(self.filled[y,x])
        delta = '无原始深度真值' if self.invalid[y,x] else f'差值 {filled-original:+d} mm'
        return f'像素 ({x}, {y}) | Original {original} mm | Filled {filled} mm | confidence {self.confidence[y,x]:.1f} | {delta}'

    def export_overview(self, path: Path):
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
        overview = Image.new('RGB',(1280,840),(22,28,34))
        draw = ImageDraw.Draw(overview)
        for index, mode in enumerate(MODES):
            x, y = (index%2)*640, (index//2)*400
            picture = self.render(mode)
            picture.thumbnail((640,360),Image.Resampling.NEAREST)
            overview.paste(picture,(x,y+32))
            draw.text((x+12,y+8),f'{index+1}. {mode}',font=font,fill='white')
        draw.text((12,803),'Green: measured | Yellow: repaired (0.3) | Red: unknown (0) | Depth: brightness encodes mm',font=font,fill='white')
        with Path(path).open('xb') as stream:
            overview.save(stream,format='PNG')


class DepthDebugViewer:
    def __init__(self, root, directory: Path):
        import tkinter as tk
        from tkinter import ttk
        self.root, self.data = root, DepthDebugData(directory)
        root.title('Depth Debug Viewer · 深度真实性检查')
        root.geometry('1200x820')
        bar = ttk.Frame(root); bar.pack(fill='x')
        self.mode = tk.StringVar(value=MODES[0])
        chooser = ttk.Combobox(bar,textvariable=self.mode,values=MODES,state='readonly',width=22)
        chooser.pack(side='left',padx=8,pady=8)
        chooser.bind('<<ComboboxSelected>>',lambda _:self.refresh())
        ttk.Button(bar,text='适应窗口',command=self.fit).pack(side='left')
        ttk.Button(bar,text='1:1像素',command=lambda:self.zoom_to(1)).pack(side='left')
        ttk.Button(bar,text='放大',command=lambda:self.zoom_to(self.zoom*1.5)).pack(side='left')
        ttk.Button(bar,text='缩小',command=lambda:self.zoom_to(self.zoom/1.5)).pack(side='left')
        ttk.Label(root,text='绿色：真实Depth  |  黄色：修复Depth（0.3）  |  红色：未知（0）  |  深度模式亮度表示毫米值；数字键1–4切换').pack(fill='x')
        report = self.data.report
        check = report.get('depth_error_check',{}).get('after_rejection',{}).get('original_valid_comparison',{})
        ttk.Label(root,text=f"{report.get('processing_mode','quality')} | invalid {report['original_invalid_rate']:.6%} → {report['filled_invalid_rate']:.6%} | 修复 {report['repaired_pixels']} | 原有效点MAE {check.get('mean_absolute_error_mm','未评估')} mm | 不能据此认定补点几何真值").pack(fill='x')
        area = ttk.Frame(root); area.pack(fill='both',expand=True)
        self.canvas = tk.Canvas(area,background='#161c22',highlightthickness=0)
        sy = ttk.Scrollbar(area,orient='vertical',command=self.canvas.yview)
        sx = ttk.Scrollbar(area,orient='horizontal',command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=sy.set,xscrollcommand=sx.set)
        self.canvas.grid(row=0,column=0,sticky='nsew'); sy.grid(row=0,column=1,sticky='ns'); sx.grid(row=1,column=0,sticky='ew')
        area.rowconfigure(0,weight=1); area.columnconfigure(0,weight=1)
        self.status = tk.StringVar(value=str(self.data.directory))
        ttk.Label(root,textvariable=self.status).pack(fill='x')
        self.zoom = .5
        self.canvas.bind('<Motion>',self.inspect)
        self.canvas.bind('<MouseWheel>',lambda e:self.zoom_to(self.zoom*(1.25 if e.delta>0 else .8)))
        for i, mode in enumerate(MODES,1):
            root.bind(str(i),lambda e,m=mode:self.set_mode(m))
        root.after_idle(self.fit)

    def set_mode(self, mode):
        self.mode.set(mode); self.refresh()

    def fit(self):
        self.root.update_idletasks()
        h,w = self.data.original.shape
        self.zoom_to(min(self.canvas.winfo_width()/w,self.canvas.winfo_height()/h))

    def zoom_to(self, zoom):
        self.zoom = min(8,max(.1,zoom)); self.refresh()

    def refresh(self):
        from PIL import ImageTk
        source = self.data.render(self.mode.get())
        width,height = max(1,round(source.width*self.zoom)),max(1,round(source.height*self.zoom))
        self.photo = ImageTk.PhotoImage(source.resize((width,height),Image.Resampling.NEAREST))
        self.canvas.delete('all')
        self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.canvas.configure(scrollregion=(0,0,width,height))

    def inspect(self, event):
        x,y = int(self.canvas.canvasx(event.x)/self.zoom),int(self.canvas.canvasy(event.y)/self.zoom)
        h,w = self.data.original.shape
        if 0<=x<w and 0<=y<h:
            self.status.set(self.data.describe_pixel(x,y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',type=Path,help='Directory containing depth_quality.json and PNG/NPY files')
    parser.add_argument('--export',type=Path,help='Export four panels without opening a window')
    args = parser.parse_args()
    if args.export:
        if not args.directory:
            parser.error('--export requires directory')
        DepthDebugData(args.directory).export_overview(args.export)
        return
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk(); root.withdraw()
    directory = args.directory or filedialog.askdirectory(title='选择包含 depth_quality.json 的派生目录')
    if not directory:
        root.destroy(); return
    DepthDebugViewer(root,Path(directory)); root.deiconify(); root.mainloop()


if __name__ == '__main__':
    main()
