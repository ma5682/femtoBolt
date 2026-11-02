"""Read-only stage diagnosis; never permits stage models to enter planning."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTextBrowser, QVBoxLayout


STATUS = {'reference_support_met':'绑定参考图中的目标像素均有一致支持，仍需核对模型与边界。',
          'needs_review_or_acquisition':'存在需检查或补采的位置。',
          'cannot_determine':'当前证据不足，无法判断目标范围是否满足要求。'}


def show_checkpoint(parent, checkpoint):
    dialog=QDialog(parent)
    dialog.setWindowTitle('阶段采集检查 · 可继续同会话补采')
    dialog.resize(760,760)
    layout=QVBoxLayout(dialog)
    feedback=checkpoint.get('feedback',{})
    text=[STATUS.get(feedback.get('status'),'检查状态未知'),
          '本结果只检查绑定参考图中的目标范围；阶段模型尚未确认为最终规划模型。']
    if 'supported_pixels' in feedback:
        text.append(f"当前阶段：一致支持 {feedback['supported_pixels']} 像素，"
                    f"仍待检查 {feedback['pending_pixels']} 像素；原框未映射像素另列。")
    text.append('检查图图例：绿色为网格与参考深度一致；紫色为网格或参考深度缺失；'
                '橙色为观测缺失、超范围、冲突或无源像素。同色可能有不同原因，详见任务说明。')
    comparison=feedback.get('comparison')
    if comparison:
        text.append(f"相对上一阶段：改善 {comparison['improved_pixels']} 像素，退化 "
                    f"{comparison['degraded_pixels']} 像素，仍待检查 {comparison['remaining_pending_pixels']} 像素。")
    elif feedback.get('comparison_unavailable_reason')=='previous_diagnosis_unavailable':
        text.append('上一阶段缺少可比较的像素诊断，本轮不计算改善、退化或无收益次数。')
    for task in feedback.get('tasks',[]):
        bounds=task['reference_pixel_bounds']
        frame='原始扫描框' if task.get('pixel_frame')=='raw_aligned_rgb' else '参考图'
        text.append(f"\n{task['reason']}（{task['pixel_count']} 像素）\n{task['suggestion']}\n"
                    f"{frame}范围：左上 {bounds[:2]}，右下 {bounds[2:]}；仅表示该原因的包围范围。")
    if feedback.get('stop_reason'):
        text.append('\n多轮未观察到参考像素支持改善，建议停止重复补采并检查原因。')
    browser=QTextBrowser()
    browser.setPlainText('\n'.join(text));layout.addWidget(browser)
    image=QPixmap(checkpoint.get('preview_path',''))
    if not image.isNull():
        label=QLabel();label.setAlignment(Qt.AlignCenter)
        label.setPixmap(image.scaled(700,320,Qt.KeepAspectRatio,Qt.SmoothTransformation))
        label.setAccessibleName('原扫描范围内的缺口与待检查位置颜色图')
        layout.addWidget(label)
    buttons=QDialogButtonBox(QDialogButtonBox.Close)
    buttons.button(QDialogButtonBox.Close).setText('关闭')
    buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
    dialog.exec()
