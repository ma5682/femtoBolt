"""Display derivatives in memory; originals are never rewritten."""
import json
import struct
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtCore import QRect


def depth_heatmap(path, low_mm=250, high_mm=2210):
    raw = QImage(str(path))
    if raw.isNull():
        raise ValueError("无法读取深度图像")
    raw = raw.convertToFormat(QImage.Format_Grayscale16)
    source = bytes(raw.constBits())
    stride = raw.bytesPerLine()
    sample = max(1, raw.width() // 960)
    width, height = raw.width() // sample, raw.height() // sample
    output = bytearray(width*height*3)
    palette = []
    for i in range(256):
        color = QColor.fromHsvF((1-i/255)*0.67, .83, .96)
        palette.append((color.red(), color.green(), color.blue()))
    for y in range(height):
        values = struct.unpack_from("<"+"H"*raw.width(), source, y*sample*stride)
        for x in range(width):
            mm = values[x*sample]
            color = (10, 15, 20) if mm == 0 else palette[min(255, max(0, int((mm-low_mm)/(high_mm-low_mm)*255)))]
            at = (y*width+x)*3
            output[at:at+3] = bytes(color)
    return QImage(bytes(output), width, height, width*3, QImage.Format_RGB888).copy()


def region_image(data):
    """Nearest-neighbour RGB remap into the exported region's undistorted pixels."""
    region = data.region["region"]
    calibration = data.dataset / "calibration"
    intrinsic = json.loads((calibration / "color_intrinsic.json").read_text(encoding="utf-8"))
    distortion = json.loads((calibration / "distortion.json").read_text(encoding="utf-8"))["rgb"]
    if "BROWN_CONRADY" not in distortion.get("model", ""):
        raise ValueError("当前回放不支持该 RGB 畸变模型")
    k = region["view_reference"]["undistorted_intrinsic"]
    original = QImage(str(data.rgb_path)).convertToFormat(QImage.Format_RGB888)
    if original.isNull():
        raise ValueError("无法读取参考 RGB")
    full_w, full_h = region["view_reference"]["image_size"]
    if (original.width(), original.height()) != (full_w, full_h):
        raise ValueError("参考 RGB 尺寸与选区记录不一致")
    # Half-resolution is a display-only derivative in the same pixel coordinate frame.
    sample = 2
    w, h = full_w // sample, full_h // sample
    raw = bytes(original.constBits())
    stride = original.bytesPerLine()
    output = bytearray(w*h*3)
    d = distortion
    for y in range(h):
        yn = (y*sample-k[1][2])/k[1][1]
        yy = yn*yn
        for x in range(w):
            xn = (x*sample-k[0][2])/k[0][0]
            r2 = xn*xn+yy
            radial = (1+d["k1"]*r2+d["k2"]*r2*r2+d["k3"]*r2*r2*r2)/(1+d["k4"]*r2+d["k5"]*r2*r2+d["k6"]*r2*r2*r2)
            xd = xn*radial+2*d["p1"]*xn*yn+d["p2"]*(r2+2*xn*xn)
            yd = yn*radial+d["p1"]*(r2+2*yy)+2*d["p2"]*xn*yn
            sx = round(xd*intrinsic["fx"]+intrinsic["cx"])
            sy = round(yd*intrinsic["fy"]+intrinsic["cy"])
            if 0 <= sx < full_w and 0 <= sy < full_h:
                at = (y*w+x)*3
                source_at = sy*stride+sx*3
                output[at:at+3] = raw[source_at:source_at+3]
    base = QImage(bytes(output), w, h, w*3, QImage.Format_RGB888).copy()
    selected = base.copy()
    p = QPainter(selected)
    runs = region["selection_support"]["class_runs"]
    offset = 0
    for cls, count in zip(runs[::2], runs[1::2]):
        end = offset+count
        if cls:
            color = QColor(22, 197, 125, 75) if cls == 1 else QColor(193, 95, 248, 220)
            start = offset
            while start < end:
                row, col = divmod(start, full_w)
                stop = min(end, (row+1)*full_w)
                if row % sample == 0:
                    p.fillRect(QRect(col//sample, row//sample, max(1, (stop-start+sample-1)//sample), 1), color)
                start = stop
        offset = end
    p.end()
    if offset != full_w*full_h:
        raise ValueError("选区支持分类长度不正确")
    return base, selected
