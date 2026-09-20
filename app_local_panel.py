"""Local workflow UI. Existing tools remain in the second notebook tab."""
from pathlib import Path
import json
import queue
import subprocess
import sys
import threading
import time
import uuid
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


TEXT = {
 'relocalizing': ('正在尝试重定位 · 剩余 {seconds:.1f} 秒；保持在已接纳区域附近。当前坐标未知，不接纳补采。','Relocalizing · {seconds:.1f} s remaining. Stay near accepted views. No current pose or admission.'),
 'relocalized': ('已通过同会话关键帧重定位；请重新停稳并请求补采。','Relocalized to a same-session keyframe; hold still and request a new view.'),
 'reason_anchor_disagreement': ('帧间与关键帧位姿冲突','Frame/keyframe pose disagreement'),
 'reason_relocalization_timeout': ('重定位窗口超时','Relocalization timed out'),
 'reason_relocalization_no_feature_support': ('已接纳视角中没有足够的特征和深度支持','Accepted views lack sufficient feature/depth support'),
 'reason_relocalization_ambiguous': ('多个关键帧候选位姿不一致','Keyframe pose hypotheses disagree'),
 'reason_relocalization_seed_disagreement': ('特征初值与稠密配准结果不一致','Feature seed and dense registration disagree'),
 'reason_relocalization_pending': ('已通过一帧，等待连续新鲜帧确认','One frame passed; waiting for consecutive fresh confirmation'),
 'reason_replay_end_while_relocalizing': ('回放结束时仍未恢复可信位姿','Replay ended without a trusted recovered pose'),
 'reason_pose_correction_recheck': ('位姿修正后需重新停稳','Recheck stillness after pose correction'),
 'title': ('局部三维采集与区域标记', 'Local capture and surface annotation'),
 'tab': ('局部移动补采', 'Local capture'), 'legacy': ('固定采集与高级功能', 'Fixed capture & advanced'),
 'origin': ('01 建立原点', '01 Set origin'), 'capture': ('02 移动补采', '02 Capture'),
 'review': ('03 检查模型', '03 Review'), 'mark': ('04 区域标记', '04 Annotate'),
 'start': ('建立原点并开始', 'Set origin and start'), 'add': ('停稳补采一帧', 'Capture a stable view'),
 'restart': ('重新建立原点并开始', 'New origin and restart'),
 'capture_lost': ('02 跟踪已中断', '02 Tracking interrupted'),
 'pause': ('暂停补采', 'Pause keyframes'), 'resume': ('继续补采', 'Resume keyframes'),
 'finish': ('结束并检查模型', 'Finish and reconstruct'), 'cancel': ('取消本次采集', 'Cancel session'),
 'view': ('打开模型与标记工具', 'Open model & annotation tools'),
 'fixed_view': ('固定 RGB 画面选区', 'Fixed RGB annotation'),
 'history': ('历史记录', 'History'),
 'outside_local_workspace': ('估计位姿超出原点周围 30 mm 或相对初始朝向总旋转 10°，未接纳补采。请缓慢返回范围内。', 'Estimated pose exceeds 30 mm from origin or 10 degrees total rotation from initial orientation. View not accepted; return slowly.'),
 'initial': ('初始点云', 'Initial cloud'), 'final': ('融合点云', 'Fused cloud'),
 'live': ('累计预览', 'Accumulated'), 'fit': ('复位视角', 'Fit view'),
 'highlight': ('高亮本次新增采样点', 'Highlight new samples'),
 'live_wait': ('等待第一帧有效补采数据', 'Waiting for first accepted frame'),
 'live_stats': ('预览已更新 {frames}/{accepted} 帧 · {points} 点 · 本次新增 {added} 个采样体素',
                'Preview {frames}/{accepted} views · {points} points · {added} new sample voxels'),
 'live_limit': ('已达预览点数上限；最终重建仍使用全部接纳帧。', 'Preview point budget reached; final reconstruction uses all accepted views.'),
 'live_error': ('累计预览不可用；跟踪和原始采集继续。', 'Accumulated preview unavailable; tracking and raw capture continue.'),
 'live_note': ('采样预览，非最终模型；新增采样体素不等于补洞数量。', 'Sample preview, not final geometry; new voxels are not a hole-fill count.'),
 'preview': ('当前 RGB', 'Live RGB'), 'depth': ('当前深度', 'Live depth'),
 'cloud': ('打开实时三维点云', 'Open live 3D cloud'),
 'waiting': ('等待建立原点；请保持相机和目标稳定。', 'Keep camera and target still to establish origin.'),
 'tracking': ('跟踪正常；缓慢移动，停稳后请求补采。候选帧提示不代表降噪通过。', 'Tracking. Move slowly, hold still, then request a view. Candidate feedback does not approve denoising.'),
 'uncertain': ('当前位姿暂不可信：{reason}。请立即停稳，保持与上一视角重叠；正在短间隔内重试，此帧不参与融合。',
               'Pose uncertain: {reason}. Hold still and keep the previous view in sight. Retrying within the short tracking window; this frame is excluded.'),
 'waiting_reason': ('尚未建立原点：{reason}。请保持目标稳定并检查深度画面。',
                    'Origin not set: {reason}. Keep the target still and check the depth view.'),
 'paused_tracking': ('补采已暂停，原点保持不变，位姿仍在跟踪；点击“继续补采”恢复接纳。',
                     'Keyframes paused; the fixed origin and live tracking are retained. Resume keyframes to accept views.'),
 'stale': ('位姿更新已过期，已暂停补采接纳。请停稳并保持原视野；等待后台确认跟踪状态。',
           'Pose update is stale; accepting views is blocked. Hold still with the same view while tracking status is checked.'),
 'finishing': ('正在结束采集并检查本次数据，请等待。', 'Ending capture and checking this session…'),
 'cancelling': ('正在取消本次任务，保留原始证据；尚未完成的结果不会发布。', 'Cancelling this task; original evidence is retained and unfinished results are not published.'),
 'comparison': ('同源对照', 'A/B/C comparison'),
 'lost_detail': ('跟踪已中断：{reason}。\n已接纳 {n} 帧，原始数据和预览保留，本次未生成最终模型。请重新开始；未启用恢复或未在恢复窗口内获得可信定位。',
                 'Tracking interrupted: {reason}.\n{n} accepted views, raw data and preview retained; no final model was generated. Start a new session; recovery was disabled or no trusted pose was recovered within its window.'),
 'preceding_reason': ('；此前：{reason}', '; preceding failure: {reason}'),
 'reason_pair_quality': ('RGB-D 配准未通过质量检查', 'RGB-D registration failed quality checks'),
 'reason_cycle_consistency': ('前后向配准结果不一致', 'forward and reverse registration disagree'),
 'reason_fitness': ('可匹配表面不足', 'insufficient matching surface'),
 'reason_rmse': ('配准残差不合格', 'registration residual failed the gate'),
 'reason_translation_step': ('估计单步位移超出门限', 'estimated translation step exceeds the limit'),
 'reason_rotation_step': ('估计单步转角超出门限', 'estimated rotation step exceeds the limit'),
 'reason_tracking_gap': ('超过 2 秒未获得可信位姿', 'no trusted pose for over 2 seconds'),
 'reason_no_frames': ('超过 2 秒未收到 RGB-D 输入', 'no RGB-D input for over 2 seconds'),
 'reason_stale_frame': ('待处理帧已过期', 'queued frame is too old'),
 'reason_processing_stale': ('位姿处理超过 2 秒时效门限', 'pose processing exceeded the 2-second age limit'),
 'reason_low_depth': ('有效深度不足', 'insufficient valid depth'),
 'reason_blur': ('画面清晰度不足', 'insufficient image sharpness'),
 'reason_rgb_depth_sync': ('彩色与深度时间戳不同步', 'color and depth timestamps are not synchronized'),
 'reason_nonmonotonic_timestamp': ('帧时间戳未递增', 'frame timestamp did not increase'),
 'reason_forward_pose': ('正向位姿求解失败', 'forward pose estimation failed'),
 'reason_reverse_pose': ('反向位姿求解失败', 'reverse pose estimation failed'),
 'reason_tracking_lost': ('未能保持可信的连续位姿', 'trusted continuous tracking could not be maintained'),
 'reason_raw_write_stale': ('原始数据保存完成时当前帧已过期', 'frame expired before its raw write completed'),
 'reason_reverse_translation_step': ('反向配准平移步长超限', 'reverse registration translation exceeded the limit'),
 'reason_reverse_rotation_step': ('反向配准旋转步长超限', 'reverse registration rotation exceeded the limit'),
 'reason_unknown': ('未记录具体原因，请查看本次报告', 'specific cause unavailable; check the session report'),
 'lost': ('跟踪丢失；重新开始采集。已有证据已保留。', 'Tracking lost. Start a new session; evidence retained.'),
 'processing': ('正在生成局部模型，请等待。', 'Reconstructing local model…'),
 'loading_review': ('模型已生成，正在后台准备对比点云。', 'Model ready; loading comparison clouds in the background.'),
 'review_load_error': ('模型已生成，但对比点云加载失败；可打开模型或报告。', 'Model ready, but comparison clouds failed to load. Open model or report.'),
 'ready': ('模型已生成。对比初始 / 融合点云后进入标记。', 'Model ready. Compare initial/fused clouds, then annotate.'),
 'cancelled': ('本次采集已取消，原始证据保留。', 'Session cancelled; evidence retained.'),
 'duplicate': ('视角变化不足，未增加关键帧。', 'View is redundant; no keyframe added.'),
 'selection_quality': ('三维重叠或深度一致性不足，未接纳。请保留稳定背景、减少遮挡并停稳后补采。',
                       'Insufficient 3D overlap or depth consistency. Keep stable background visible, reduce occlusion and hold still.'),
 'moving_wait': ('正在移动；停稳后补采。', 'Moving; hold still before capturing a view.'),
 'waiting_still': ('等待停稳补采请求。单帧接纳与五帧稳定审核分别记录。',
                   'Waiting for a stable-view request. Single-view admission and the five-frame audit are recorded separately.'),
 'request_timeout': ('补采请求超时；请停稳后重试。', 'Capture timed out; hold still and retry.'),
 'session_limit': ('已到 5 分钟会话上限，正在结束。', 'Five-minute session limit reached; finishing.'),
 'need_three_views': ('至少需要 3 个有效且不同的视角，请重新采集。', 'At least 3 distinct valid views required; start again.'),
 'need_more_views': ('至少接纳 3 个不同视角后才能结束建模；当前会话保持，请继续停稳补采。', 'Accept at least 3 distinct views before reconstructing. This session remains open; capture more stable views.'),
 'origin_timeout': ('15 秒内未能建立原点，请调整目标和距离。', 'No valid origin within 15 seconds; adjust scene and distance.'),
 'tracking_lost': ('跟踪丢失，本次不发布成功模型；请重新采集。', 'Tracking lost; no successful model published. Restart.'),
 'final_pose_invalid': ('结束时位姿无效，本次不发布模型。', 'Final pose invalid; no model published.'),
 'calibration_changed': ('标定文件发生变化，本次不发布模型。', 'Calibration changed; no model published.'),
 'camera_error': ('相机读取发生错误，采集已中断；原始证据已保留。请检查设备连接并重启相机。',
                  'Camera read failed; capture interrupted and evidence retained. Check the connection and restart the camera.'),
 'pose': ('相对初始原点 · 估计值', 'Relative to initial origin · estimates'),
 'unit': ('XYZ / mm；俯仰 Rx、偏航 Ry、滚转 Rz / °\n补采范围：距原点 ≤30 mm；相对初始朝向总旋转 ≤10°', 'XYZ / mm; pitch Rx, yaw Ry, roll Rz / °\nCapture: radius ≤30 mm; total rotation from W0 ≤10°'),
 'note': ('补采后更新累计点云；结束后生成最终模型。\n原点固定；坐标仅用于研发可视化。', 'Accumulated cloud after each capture; final model after finishing.\nFixed origin; R&D coordinates only.'),
 'none': ('暂无当前任务模型', 'No current-session model'),
 'report': ('查看本次报告', 'Open session report'),
 'rejected': ('错误 / 拒绝原因：', 'Error / rejection: '),
 'count': ('已接纳 {n} 帧 · 建模至少 3 帧', '{n} views · minimum 3 to reconstruct'),
 'age': ('更新距今 {age:.1f}s · 处理延迟 {latency:.0f}ms', 'Age {age:.1f}s · processing {latency:.0f}ms'),
 'age_lost': ('跟踪已中断 · 原因见主视图上方', 'Tracking interrupted · see banner'),
 'age_failed': ('本次任务失败 · 原因见主视图上方', 'Session failed · see banner'),
 'age_ended': ('采集已结束 · 末次记录', 'Capture ended · last recorded pose'),
 'age_waiting': ('等待有效位姿', 'Waiting for a valid pose'),
 'age_processing': ('采集已停止 · 正在处理', 'Capture stopped · processing'),
}


def read_review_clouds(result, include_initial=True):
    """Read bounded display copies off the Tk thread; never touch source geometry."""
    from scene_scale_validation import read_geometry_safe
    clouds = {}
    names = [('final', 'scene_fused_points_m.ply')]
    if include_initial:
        names.insert(0, ('initial', 'initial_observed_points_m.ply'))
    for key, name in names:
        cloud = read_geometry_safe(Path(result)/'reconstruction'/name, mesh=False)
        step = max(1, (len(cloud.points) + 34999)//35000)
        points, colors = np.asarray(cloud.points)[::step].copy(), np.asarray(cloud.colors)[::step].copy()
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('Empty or invalid review cloud')
        clouds[key] = (points, colors)
    return clouds


class LocalPanel:
    def __init__(self, app, parent):
        self.app = app
        self.frame = tk.Frame(parent, bg='#0b121a')
        self.frame.pack(fill='both', expand=True)
        self.active = False; self.finishing = False; self.paused = False
        self.cancelling = False
        self.cancel_event = threading.Event()
        self.session_id = None; self.source = None; self.result = None
        self.record = None; self.count = 0; self.state = 'waiting'
        self.failure = None
        self.quality = None
        self.stability = None
        self.view = 'preview'; self.clouds = {}; self.yaw = 0.; self.zoom = 1.
        self.pitch = 0.; self.preview_mailbox = None; self.preview_snapshot = None
        self.preview_error = False; self.notice = None; self.notice_until = 0.
        self._review_mailbox = None
        self.highlight = tk.BooleanVar(value=True)
        self.labels = []
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)
        title = tk.Label(self.frame, bg='#0b121a', fg='#e7eef5', font=('Microsoft YaHei UI',19,'bold'))
        title.grid(row=0,column=0,columnspan=2,sticky='w',padx=18,pady=12)
        self.labels.append((title,'title'))
        steps = tk.Frame(self.frame,bg='#0b121a'); steps.grid(row=1,column=0,columnspan=2,sticky='ew',padx=18)
        self.steps=[]
        for i,key in enumerate(('origin','capture','review','mark')):
            steps.columnconfigure(i,weight=1)
            label=tk.Label(steps,bg='#172635',fg='#91a9bd',padx=12,pady=10)
            label.grid(row=0,column=i,sticky='ew',padx=3,pady=6)
            self.labels.append((label,key)); self.steps.append(label)
        main=tk.Frame(self.frame,bg='#101c28');main.grid(row=2,column=0,sticky='nsew',padx=(18,8),pady=8)
        toolbar=tk.Frame(main,bg='#101c28');toolbar.pack(fill='x',padx=8,pady=8)
        self.buttons={}
        history_actions=tk.Frame(self.frame,bg='#0b121a')
        history_actions.grid(row=0,column=1,sticky='e',padx=18,pady=12)
        self.button(history_actions,'comparison',self.open_comparison).pack(side='left',padx=3)
        self.button(history_actions,'history',self.open_history).pack(side='left',padx=3)
        for key in ('preview','depth','initial','live','final','fit'):
            self.button(toolbar,key,lambda k=key:self.set_view(k)).pack(side='left',padx=3)
        preview_options=tk.Frame(main,bg='#101c28');preview_options.pack(fill='x',padx=8)
        self.highlight_button=tk.Checkbutton(preview_options,variable=self.highlight,command=self.draw_cloud,
            bg='#101c28',fg='#67dbc0',selectcolor='#233444',activebackground='#101c28',activeforeground='#edf5fa')
        self.highlight_button.pack(anchor='w')
        self.preview_status=tk.StringVar(value='')
        tk.Label(preview_options,textvariable=self.preview_status,bg='#101c28',fg='#9fb4c7',
                 anchor='w',justify='left',wraplength=600).pack(fill='x')
        # Reserve status space before the expanding image. A large PhotoImage
        # previously squeezed the last-packed error hint off the bottom.
        self.hint=tk.StringVar()
        self.hint_label=tk.Label(main,textvariable=self.hint,bg='#142738',fg='#d6e6ef',
            anchor='w',justify='left',wraplength=600,padx=12,pady=10)
        self.hint_label.pack(fill='x')
        self.stability_text=tk.StringVar()
        self.stability_label=tk.Label(main,textvariable=self.stability_text,bg='#142738',fg='#a4dccf',
            anchor='w',justify='left',wraplength=600,padx=12,pady=4)
        self.stability_label.pack(fill='x')
        main.bind('<Configure>',lambda e:(self.hint_label.configure(wraplength=max(180,e.width-28)),
                                       self.stability_label.configure(wraplength=max(180,e.width-28))))
        self.preview=tk.Label(main,bg='#07121c',fg='#b2c4d5',text='RGB-D',anchor='center',width=1,height=1)
        self.preview.pack(fill='both',expand=True)
        self.quality_frame=tk.Frame(main,bg='#142738')
        quality_scroll=ttk.Scrollbar(self.quality_frame,orient='vertical')
        quality_scroll.pack(side='right',fill='y')
        self.quality_text=tk.Text(self.quality_frame,height=8,wrap='word',bg='#142738',fg='#deedf4',
            relief='flat',padx=10,pady=6,state='disabled',yscrollcommand=quality_scroll.set)
        self.quality_text.pack(fill='both',expand=True)
        quality_scroll.configure(command=self.quality_text.yview)
        self.preview.bind('<Configure>',lambda e:self.draw_cloud())
        self.preview.bind('<ButtonPress-1>',lambda e:setattr(self,'drag_position',(e.x,e.y)))
        self.preview.bind('<B1-Motion>',self.rotate)
        self.preview.bind('<MouseWheel>',self.wheel)
        side=tk.Frame(self.frame,bg='#111e2b',width=315,padx=14,pady=14)
        side.grid(row=2,column=1,sticky='ns',padx=(0,18),pady=8)
        pose_title=tk.Label(side,bg='#111e2b',fg='#b8d2e2');pose_title.pack(anchor='w');self.labels.append((pose_title,'pose'))
        self.pose=tk.StringVar(value='X —     Y —     Z —\nRx —    Ry —    Rz —')
        tk.Label(side,textvariable=self.pose,bg='#111e2b',fg='#edf6fa',font=('Consolas',13),justify='left',pady=14).pack(anchor='w')
        unit=tk.Label(side,bg='#111e2b',fg='#9fb4c7',wraplength=280,justify='left');unit.pack(anchor='w');self.labels.append((unit,'unit'))
        self.age=tk.StringVar(value='—');tk.Label(side,textvariable=self.age,bg='#111e2b',fg='#9fb4c7').pack(anchor='w',pady=4)
        self.counter=tk.StringVar();tk.Label(side,textvariable=self.counter,bg='#111e2b',fg='#6ee1c0').pack(anchor='w',pady=6)
        actions=tk.Frame(side,bg='#111e2b');actions.pack(fill='x')
        actions.columnconfigure(0,weight=1,uniform='actions')
        actions.columnconfigure(1,weight=1,uniform='actions')
        # Pair the short actions so the report remains fully visible at 720 px.
        for key,command,row,col,span in [('start',self.start,0,0,2),
            ('add',lambda:self.send('local_request'),1,0,2),('pause',self.pause,2,0,1),
            ('cancel',lambda:self.finish(False),2,1,1),('finish',lambda:self.finish(True),3,0,2),
            ('view',self.open_model,4,0,2),('cloud',app.toggle_point_cloud_preview,5,0,1),
            ('report',self.open_report,5,1,1)]:
            self.button(actions,key,command).grid(row=row,column=col,columnspan=span,sticky='ew',padx=2,pady=4)
        note=tk.Label(side,bg='#111e2b',fg='#9fb4c7',wraplength=280,justify='left');note.pack(anchor='w',pady=6);self.labels.append((note,'note'))
        self.refresh(); self.tick()

    def text(self,key,**values):
        return TEXT.get(key,(key,key))[0 if self.app.language=='zh' else 1].format(**values)

    def button(self,parent,key,command):
        b=tk.Button(parent,command=command,bg='#1e4559' if key in ('start','finish','view') else '#233444',fg='#edf5fa',activebackground='#285870',activeforeground='white',relief='flat',padx=10,pady=6)
        self.buttons[key]=b;return b

    def refresh(self):
        for label,key in self.labels:label.configure(text=self.text(key))
        for key,b in self.buttons.items():b.configure(text=self.text('resume' if key=='pause' and self.paused else key))
        self.buttons['view'].configure(text=self.text('fixed_view'))
        if self.quality is not None:
            from local_processing import format_quality
            self.quality_frame.pack(fill='x',before=self.preview)
            self.quality_text.configure(state='normal')
            self.quality_text.delete('1.0','end')
            self.quality_text.insert('end',format_quality(self.quality,self.app.language))
            self.quality_text.configure(state='disabled')
        else:
            self.quality_frame.pack_forget()
        self.refresh_status()
        self.highlight_button.configure(text=self.text('highlight'))
        busy=self.app.capturing or self.app.processing or self.app.photo_saving
        self.buttons['start'].configure(state='normal' if self.app.camera_ready and not busy and not self.active else 'disabled')
        for key in ('finish','cancel','pause'):
            allowed = (self.active and not self.cancelling) if key=='cancel' else (self.active and not self.finishing and self.state!='lost')
            if key=='finish':allowed = allowed and self.count>=3 and self.state=='tracking'
            self.buttons[key].configure(state='normal' if allowed else 'disabled')
        fresh = (self.record and self.record['state']=='tracking' and
                 time.monotonic()-self.record.get('received_monotonic',0)<=2.)
        fresh = fresh and self.record.get('local_workspace',{}).get('within_limit',True)
        self.buttons['add'].configure(state='normal' if self.active and not self.finishing and not self.paused and fresh else 'disabled')
        self.buttons['view'].configure(state='normal' if self.result and not self.active else 'disabled')
        self.buttons['comparison'].configure(state='normal' if self.result and not self.active else 'disabled')
        self.buttons['final'].configure(state='normal' if self.result and not self.active and 'final' in self.clouds else 'disabled')
        self.buttons['initial'].configure(state='normal' if 'initial' in self.clouds else 'disabled')
        self.buttons['live'].configure(state='normal' if 'live' in self.clouds else 'disabled')
        self.buttons['fit'].configure(state='normal' if self.view in self.clouds else 'disabled')
        if self.preview_error:
            self.preview_status.set(self.text('live_error'))
        elif self.preview_snapshot is not None and self.view in ('initial','live'):
            data=self.preview_snapshot
            self.preview_status.set(self.text('live_stats',frames=data['accepted_preview_frames'],accepted=self.count,
                points=data['point_count'],added=data['added_samples'])+'\n'+self.text('live_limit' if data['budget_limited'] else 'live_note'))
        else:
            self.preview_status.set('')
        self.buttons['report'].configure(state='normal' if self.source and (not self.active or self.quality) else 'disabled')
        self.buttons['cloud'].configure(state='normal' if self.app.camera_ready else 'disabled')
        self.counter.set(self.text('count',n=self.count))
        lost = self.state in ('lost','tracking_lost','camera_error')
        stage=(1 if lost else 0 if self.state=='waiting' else 3 if self.state=='mark' else
               2 if self.result and not self.active else 1 if self.active else -1)
        if lost:
            self.steps[1].configure(text=self.text('capture_lost'))
            self.buttons['start'].configure(text=self.text('restart'))
        for i,label in enumerate(self.steps):
            label.configure(bg=('#77312f' if lost else '#175e73') if i==stage else '#172635',
                            fg='#effaff' if i==stage else '#91a9bd')

    def reason_text(self, reason, metrics=None):
        if reason=='pair_quality' and metrics and metrics.get('rejected_checks'):
            reason=metrics['rejected_checks'][0]
        key='reason_'+str(reason)
        return self.text(key) if key in TEXT else str(reason or self.text('reason_unknown'))

    def refresh_status(self):
        record=self.record or {}
        from local_stability_feedback import format_stability
        stability=self.stability
        if self.active and record.get('received_monotonic') is not None and time.monotonic()-record['received_monotonic']>2.:
            stability=dict(state='reset',reason='stale_frame',count=0,target=5)
        self.stability_text.set(format_stability(stability,self.app.language) if stability is not None and self.active and not self.finishing else '')
        color='#142738'
        if self.state in ('lost','tracking_lost'):
            context=(self.failure or {}).get('termination') or record.get('loss_context') or {}
            reason=context.get('reason') or record.get('rejection_reason') or 'tracking_lost'
            message=self.reason_text(reason,context.get('current_pair_failure') or record.get('pair_failure'))
            previous=context.get('preceding_rejection')
            if previous and reason in ('tracking_gap','no_frames','stale_frame','processing_stale'):
                message+=self.text('preceding_reason',reason=self.reason_text(previous['reason'],previous.get('pair_failure')))
            message=self.text('lost_detail',reason=message,n=self.count);color='#622e2c'
        elif self.state=='camera_error':
            message=self.text('camera_error');color='#622e2c'
        elif self.active and self.cancelling:
            message=self.text('cancelling')
        elif self.state in ('processing','loading_review'):
            message=self.text(self.state)
        elif self.active and self.finishing:
            message=self.text('finishing')
        elif self.active and record.get('received_monotonic') is not None and time.monotonic()-record['received_monotonic']>2.:
            message=self.text('stale');color='#5b4527'
        elif self.state=='relocalizing':
            remaining=record.get('continuity',{}).get('remaining_window_s') or 0.
            updated=record.get('status_updated_monotonic',time.monotonic())
            message=self.text('relocalizing',seconds=max(0.,remaining-(time.monotonic()-updated)));color='#5b4527'
        elif self.state=='uncertain':
            message=self.text('uncertain' if self.count else 'waiting_reason',
                reason=self.reason_text(record.get('rejection_reason'),record.get('pair_failure')))
            color='#5b4527'
        elif self.active and record.get('local_workspace',{}).get('within_limit') is False:
            message=self.text('outside_local_workspace');color='#5b4527'
        elif self.notice and time.monotonic()<self.notice_until:
            message=self.text(self.notice)
        elif self.active and self.paused:
            message=self.text('paused_tracking')
        elif self.active and record.get('selection', {}).get('state') in ('moving', 'waiting_still'):
            message=self.text('moving_wait' if record['selection']['state']=='moving' else 'waiting_still')
        else:
            message=self.text(self.state)
            if self.failure is not None:color='#622e2c'
        self.hint.set(message)
        if self.active and not self.finishing:
            from local_tracking_diagnostics import quality_text
            quality_hint = quality_text(record, self.app.language)
            if quality_hint:
                self.hint.set(message+'\n'+quality_hint)
        if self.active and not self.finishing:
            from local_tracking_diagnostics import feedback_text
            self.hint.set(self.hint.get()+'\n'+feedback_text(record,self.count,time.monotonic(),True,self.app.language))
        self.hint_label.configure(bg=color)

    def send(self,command,payload=None):
        message=dict(session_id=self.session_id,value=payload)
        if command=='local_start':message['cancel_event']=self.cancel_event
        self.app.command_queue.put((command,message))

    def start(self):
        if self.active or self.app.processing or self.app.capturing or self.app.photo_saving or not self.app.camera_ready:return
        self.active=True;self.finishing=False;self.paused=False;self.session_id=uuid.uuid4().hex
        self.cancelling=False
        self.cancel_event=threading.Event()
        self.result=None;self.source=None;self.clouds={};self.record=None;self.count=0;self.state='waiting';self.failure=None
        self.quality=None
        self.stability=None
        self.preview_mailbox=None;self.preview_snapshot=None;self.preview_error=False
        self._review_mailbox=None
        self.notice=None;self.pitch=0.;self.yaw=0.;self.zoom=1.
        if hasattr(self,'center'):del self.center
        self.app.capturing=True;self.app.local_busy=True
        self.set_view('live');self.pose.set('X —     Y —     Z —\nRx —    Ry —    Rz —')
        self.send('local_start');self.app._set_action_state();self.refresh()

    def pause(self):
        if not self.active or self.finishing or self.state=='lost':return
        self.paused=not self.paused;self.stability=dict(state='paused' if self.paused else 'idle',count=0,target=5)
        self.send('local_pause',self.paused);self.refresh()

    def finish(self,reconstruct):
        if not self.active or self.cancelling or (reconstruct and self.finishing):return
        if reconstruct and self.count<3:
            self.notice='need_more_views';self.notice_until=time.monotonic()+3.
            self.refresh();return
        self.finishing=True;self.cancelling=not reconstruct
        if not reconstruct:self.cancel_event.set()
        self.send('local_finish',reconstruct);self.refresh()

    def event(self,kind,payload):
        if not self.active or not self.session_id or payload.get('session_id')!=self.session_id:return
        self.source=Path(payload['source']) if payload.get('source') else self.source
        if kind=='local_started':
            self.preview_mailbox=payload.get('preview_mailbox')
        elif kind=='local_pose':
            previous_state=self.state
            self.record=payload['record'];self.count=payload['accepted'];self.state=self.record['state']
            if self.state=='tracking' and (previous_state in ('uncertain','relocalizing') or self.record.get('selected_for_fusion')):self.notice=None
            self.stability=self.record.get('candidate_stability')
            if self.record.get('continuity',{}).get('relocalized'):
                self.notice='relocalized';self.notice_until=time.monotonic()+5.
            if self.state!='tracking':self.pose.set('X —     Y —     Z —\nRx —    Ry —    Rz —')
        elif kind=='local_stability':
            self.stability=payload['stability']
        elif kind=='local_notice':
            self.notice=payload['key'];self.notice_until=time.monotonic()+3.
        elif kind=='local_processing':
            self.state='processing';self.app.processing=True;self.app.capturing=False
            if payload.get('quality') is not None:self.quality=payload['quality']
        elif kind in ('local_finished','local_cancelled','local_error'):
            self.active=False;self.finishing=False;self.cancelling=False;self.app.local_busy=False
            self.notice=None
            self.app.processing=False;self.app.capturing=False
            self.state='ready' if kind=='local_finished' else 'cancelled' if kind=='local_cancelled' else payload.get('error','lost')
            self.failure=payload.get('failure',{}) if kind=='local_error' else None
            if payload.get('quality') is not None:self.quality=payload['quality']
            if kind=='local_error':self.pose.set('X —     Y —     Z —\nRx —    Ry —    Rz —')
            if kind=='local_finished':
                if payload.get('quality') is not None:self.quality=payload['quality']
                self.result=Path(payload['result']);self.app.latest_result=self.result;self.app.latest_moving_result=self.result
                self.app.latest_dataset=self.source
                self.load_clouds()
        self.refresh()

    def tick(self):
        if not self.frame.winfo_exists():return
        self.consume_review()
        self.consume_preview()
        if self.notice and time.monotonic()>=self.notice_until:
            self.notice=None
        self.refresh_status()
        r=self.record
        tracking_live = self.active and not self.finishing and self.state!='processing'
        if self.failure is not None:
            self.age.set(self.text('age_failed'))
        elif self.state in ('lost','tracking_lost','camera_error'):
            self.age.set(self.text('age_lost'))
        elif self.finishing or self.state in ('processing','loading_review'):
            self.age.set(self.text('age_processing'))
        elif not tracking_live and r:
            self.age.set(self.text('age_ended'))
        elif not r or r.get('received_monotonic') is None:
            self.age.set(self.text('age_waiting'))
        if r and r.get('received_monotonic') is not None:
            age=max(0,time.monotonic()-r['received_monotonic'])
            if tracking_live and self.state not in ('lost','tracking_lost','camera_error'):
                self.age.set(self.text('age',age=age,latency=r.get('latency_ms',0)))
            if self.failure is None and r['state']=='tracking' and self.state not in ('lost','tracking_lost','camera_error') and (not tracking_live or age<=2):
                xyz=r['xyz_mm'];ang=r['pitch_yaw_roll_deg']
                self.pose.set('X {:+.1f}  Y {:+.1f}  Z {:+.1f}\n'.format(*xyz)+('Rx {:+.1f} Ry {:+.1f} Rz {:+.1f}'.format(*ang) if ang else 'Rx —  Ry —  Rz —'))
            else:
                self.pose.set('X —     Y —     Z —\nRx —    Ry —    Rz —')
                if tracking_live and age>2:
                    self.buttons['add'].configure(state='disabled')
        else:self.pose.set('X —     Y —     Z —\nRx —    Ry —    Rz —')
        self.app.root.after(150,self.tick)

    def set_view(self,view):
        if view=='fit':
            self.fit_cloud();return
        self.view=view
        if view in self.clouds:self.draw_cloud()
        elif view=='live':self.preview.configure(image='',text=self.text('live_wait'))
        self.refresh()

    def consume_preview(self):
        if self.preview_mailbox is None:return
        try:data=self.preview_mailbox.get_nowait()
        except queue.Empty:return
        if data.get('session_id')!=self.session_id:return
        if data['status']=='error':
            self.preview_error=True;self.refresh();return
        self.preview_error=False
        first=self.preview_snapshot is None
        self.preview_snapshot=data
        self.clouds['live']=(data['points'],data['colors'].astype(float)/255.)
        # A completed model's initial reference is kept at final reconstruction resolution.
        if self.result is None:
            self.clouds['initial']=(data['initial_points'],data['initial_colors'].astype(float)/255.)
        if self.result is None and (first or not hasattr(self,'center')):self.fit_cloud()
        elif self.view in ('live','initial'):self.draw_cloud()
        self.refresh()

    def fit_cloud(self):
        key='final' if 'final' in self.clouds else 'live' if 'live' in self.clouds else self.view
        if key not in self.clouds or not len(self.clouds[key][0]):return
        p=self.clouds[key][0]
        self.center=(p.min(axis=0)+p.max(axis=0))*.5
        self.span=max(float(np.ptp(p,axis=0).max()),.01)
        self.yaw=0.;self.pitch=0.;self.zoom=1.;self.draw_cloud()

    def load_clouds(self):
        mailbox=self._review_mailbox=queue.Queue(maxsize=1)
        session,result=self.session_id,self.result
        reuse_initial='initial' in self.clouds
        self.state='loading_review';self.app.processing=True
        def worker():
            from local_processing import result_quality
            quality=result_quality(result)
            try:
                clouds=(read_review_clouds(result,include_initial=False) if reuse_initial
                        else read_review_clouds(result))
                data=dict(clouds=clouds,error=None)
            except Exception as error:
                data=dict(clouds=None,error=str(error))
            mailbox.put_nowait(dict(session_id=session,result=result,quality=quality,**data))
        threading.Thread(target=worker,name='FemtoLocalReview',daemon=True).start()

    def consume_review(self):
        if self._review_mailbox is None:return
        try:data=self._review_mailbox.get_nowait()
        except queue.Empty:return
        self._review_mailbox=None
        if data['session_id']!=self.session_id or data['result']!=self.result or self.active:return
        if data.get('quality') is not None:self.quality=data['quality']
        self.app.processing=False
        if data['error'] is None:
            self.clouds.update(data['clouds']);self.state='ready'
            self.view='final';self.fit_cloud()
        else:
            self.state='review_load_error'
        self.app._set_action_state();self.refresh()

    def rotate(self,event):
        # Local workflow keeps the initial axis orientation. Free orbit is in
        # the advanced model viewer only; zoom/fit remain available here.
        return

    def wheel(self,event):
        self.zoom=float(np.clip(self.zoom*(1.1 if event.delta>0 else 1/1.1),.25,6));self.draw_cloud()

    def draw_cloud(self):
        if self.view not in self.clouds or not hasattr(self,'center'):return
        points,colors=self.clouds[self.view];p=points-self.center
        c,s=np.cos(self.yaw),np.sin(self.yaw)
        p=p@np.array([[c,0,-s],[0,1,0],[s,0,c]])
        c,s=np.cos(self.pitch),np.sin(self.pitch)
        p=p@np.array([[1,0,0],[0,c,s],[0,-s,c]])
        width=max(100,self.preview.winfo_width());height=max(100,self.preview.winfo_height())
        xy=p[:,:2]*min(width,height)*.82*self.zoom/self.span+[width/2,height/2]
        image=Image.new('RGB',(width,height),'#07121c');draw=ImageDraw.Draw(image)
        cols=np.clip(colors*255,0,255).astype(np.uint8) if len(colors)==len(p) else np.tile([140,190,200],(len(p),1))
        if self.view=='live' and self.preview_snapshot is not None and self.highlight.get():
            sources=self.preview_snapshot['sources']
            if self.preview_snapshot['accepted_preview_frames']>1:
                cols[sources==self.preview_snapshot['frame_id']]=[56,213,181]
        for i in np.argsort(p[:,2])[::-1]:
            x,y=xy[i]
            if 0<=x<width and 0<=y<height:draw.ellipse((x,y,x+2,y+2),fill=tuple(cols[i]))
        self.photo=ImageTk.PhotoImage(image);self.preview.configure(image=self.photo,text='')

    def open_model(self):
        if not self.result:return
        self.state='mark';self.refresh()
        subprocess.Popen([sys.executable,str(Path(__file__).parent/'fixed_rgb_viewer.py'),str(self.result/'reconstruction'),'--lang',self.app.language],cwd=Path(__file__).parent)

    def open_history(self):
        from app_local_history import LocalHistoryWindow
        from femto_control_app import DATASETS_ROOT,OUTPUTS_ROOT
        old=getattr(self,'history_window',None)
        if old is not None and not old.closed:
            old.window.lift();old.refresh();return
        self.history_window=LocalHistoryWindow(self.app.root,DATASETS_ROOT,OUTPUTS_ROOT,self.app.language)

    def open_comparison(self):
        if not self.result or self.active:return
        from app_local_comparison import ComparisonWindow
        from femto_control_app import OUTPUTS_ROOT
        self.comparison_window=ComparisonWindow(self.app.root,self.result,OUTPUTS_ROOT,self.app.language)

    def open_report(self):
        import os
        choices=[]
        if self.result:choices.extend([self.result/'processing_report.html',self.result/'result_quality.html'])
        if self.quality and self.quality.get('report_path'):
            choices.append(Path(self.quality['report_path']).with_suffix('.html'))
        if self.source:choices.extend([self.source/'derived/local_failure.json',self.source/'derived/local_capture_report.json'])
        for path in choices:
            if path.is_file():os.startfile(path);return
