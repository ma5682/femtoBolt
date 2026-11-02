"""Independent data maintenance window; close the workstation before writes."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication,QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,
    QLineEdit,QPushButton,QTreeWidget,QTreeWidgetItem,QCheckBox,QTextBrowser,QFileDialog,QStyledItemDelegate)

from femto import data_backup as storage
from femto.annotation.fixed_rgb_geometry import GeometryTaskCancelled
from femto.app.qt_common import TaskRunner
from femto.paths import REPO_ROOT
from femto.pipeline.data_maintenance import default_roots,inspect_references,backup_workspace
from femto.pipeline.disk_inventory import inspect_disk

ROOT_LABELS=dict(datasets='采集数据',outputs='模型与方案',photos='原始照片',models='模型资源')

class RootDirectoryDelegate(QStyledItemDelegate):
    def createEditor(self,parent,option,index):
        return None if index.column()==0 else super().createEditor(parent,option,index)


class DataMaintenanceDialog(QDialog):
    def __init__(self,workspace,parent=None):
        super().__init__(parent)
        self.workspace=Path(workspace).resolve();self._closing=False;self._loaded_path=None;self._cancellable=True;self._manifest_sha=None
        self.setWindowTitle('数据备份与恢复');self.resize(940,640)
        layout=QVBoxLayout(self)
        self.notice=QLabel('开始前请关闭工作台和数据处理工具。');self.notice.setWordWrap(True);layout.addWidget(self.notice)
        line=QHBoxLayout();self.operation=QComboBox()
        self.operation.setMinimumWidth(180)
        arrow=(Path(__file__).parent/'assets/maintenance_chevron.svg').as_posix()
        self.operation.setStyleSheet('QComboBox::down-arrow { image: url("'+arrow+'"); width: 10px; height: 6px; }')
        for label,key in [('备份数据','backup'),('检查备份','verify'),('恢复数据','restore'),
                          ('继续中断恢复','recover'),('中止中断恢复','abort')]:self.operation.addItem(label,key)
        line.addWidget(self.operation)
        self.path=QLineEdit();self.path.setPlaceholderText('选择备份位置或恢复记录');line.addWidget(self.path,1)
        self.browse=QPushButton('选择…');self.browse.clicked.connect(self.choose_path);line.addWidget(self.browse);layout.addLayout(line)
        self.roots=QTreeWidget();self.roots.setHeaderLabels(['数据根','目录']);self.roots.setColumnWidth(0,140)
        self.roots.setRootIsDecorated(False);layout.addWidget(self.roots,1)
        self.roots.setItemDelegate(RootDirectoryDelegate(self.roots))
        extra=QHBoxLayout();self.add_root=QPushButton('添加外部数据目录');self.add_root.clicked.connect(self.add_external)
        extra.addWidget(self.add_root)
        self.inspect=QPushButton('检查关联资产');self.inspect.clicked.connect(self.inspect_data);extra.addWidget(self.inspect)
        self.disk_preview=QPushButton('磁盘与清理预览');self.disk_preview.clicked.connect(self.preview_disk);extra.addWidget(self.disk_preview)
        extra.addStretch();layout.addLayout(extra)
        self.allow_missing=QCheckBox('记录已有缺失并备份其余数据')
        self.allow_missing.setToolTip('仅允许记录备份前已经缺失的资产；此备份无法补回这些文件。现存外部资产仍须纳入数据目录。')
        layout.addWidget(self.allow_missing)
        self.status=QLabel('选择操作和位置。');self.status.setWordWrap(True);self.status.setTextFormat(Qt.PlainText);layout.addWidget(self.status)
        self.detail_toggle=QPushButton('查看详情');self.detail_toggle.setCheckable(True);layout.addWidget(self.detail_toggle)
        self.details=QTextBrowser();self.details.setVisible(False);self.details.setMaximumHeight(170)
        self.detail_toggle.toggled.connect(self.details.setVisible);layout.addWidget(self.details)
        actions=QHBoxLayout();actions.addStretch()
        self.run=QPushButton('开始备份');self.run.clicked.connect(self.execute);actions.addWidget(self.run)
        self.cancel=QPushButton('取消操作');self.cancel.setEnabled(False);self.cancel.clicked.connect(self.request_cancel);actions.addWidget(self.cancel)
        self.close_button=QPushButton('关闭');self.close_button.clicked.connect(self.close);actions.addWidget(self.close_button);layout.addLayout(actions)
        self.runner=TaskRunner(self);self.runner.progress.connect(self.status.setText);self.runner.finished.connect(self.finished_job)
        self.operation.currentIndexChanged.connect(self.mode_changed);self.path.textEdited.connect(self.path_changed)
        self.mode_changed()

    def roots_value(self):
        return {self.roots.topLevelItem(i).data(0,Qt.UserRole):Path(self.roots.topLevelItem(i).text(1)).resolve()
                for i in range(self.roots.topLevelItemCount())}

    def set_roots(self,roots):
        self.roots.clear()
        for name,path in roots.items():
            item=QTreeWidgetItem([ROOT_LABELS.get(name,'外部数据 · '+name),str(path)])
            item.setData(0,Qt.UserRole,name);item.setFlags(item.flags()|Qt.ItemIsEditable)
            item.setToolTip(1,'双击编辑目录；恢复目标必须不存在，原目录不会被覆盖。')
            self.roots.addTopLevelItem(item)

    def mode_changed(self):
        key=self.operation.currentData();self._loaded_path=None;self._manifest_sha=None
        self.set_roots(default_roots(self.workspace) if key=='backup' else {})
        self.add_root.setVisible(key=='backup');self.allow_missing.setVisible(key=='backup')
        self.disk_preview.setVisible(key=='backup')
        self.inspect.setText('检查关联资产' if key=='backup' else '读取备份' if key in ('restore','verify') else '读取恢复记录')
        self.run.setText(dict(backup='开始备份',verify='检查备份',restore='恢复到空目录',recover='继续恢复',abort='中止并保留暂存')[key])
        self.sync_controls()

    def path_changed(self):
        self._loaded_path=None
        self._manifest_sha=None
        if self.operation.currentData()!='backup':self.roots.clear()
        self.sync_controls()

    def sync_controls(self):
        busy=self.runner.busy;key=self.operation.currentData()
        for widget in (self.operation,self.path,self.browse,self.roots,self.add_root,self.inspect,self.disk_preview,self.allow_missing):widget.setEnabled(not busy)
        self.cancel.setEnabled(busy and self._cancellable and not self.runner.cancelling)
        self.run.setEnabled(not busy and (key in ('backup','verify') or self._loaded_path==self.path.text()))

    def choose_path(self):
        key=self.operation.currentData()
        if key in ('recover','abort'):
            value,_=QFileDialog.getOpenFileName(self,'选择恢复记录',str(self.workspace/'.data_restores'),'恢复记录 (*.json)')
        else:
            value=QFileDialog.getExistingDirectory(self,'选择备份目录的上级位置' if key=='backup' else '选择备份目录',str(self.workspace))
            if value and key=='backup':value=str(Path(value)/('backup_'+datetime.now().strftime('%Y%m%d_%H%M%S')))
        if value:
            self.path.setText(value);self.path_changed()
            if key!='backup':self.inspect_data()

    def add_external(self):
        value=QFileDialog.getExistingDirectory(self,'选择关联资产所在的数据目录',str(self.workspace))
        if value:
            roots=self.roots_value();number=1
            while 'external_'+str(number) in roots:number+=1
            roots['external_'+str(number)]=Path(value);self.set_roots(roots)

    def job(self,function,success,*,cancellable=True):
        if self.runner.busy:return
        self._cancellable=cancellable
        completed=[None]
        def work(cancel,progress):
            def check():
                if cancel.is_set():raise GeometryTaskCancelled('操作已取消')
            value=function(check,progress);completed[0]=value;return value
        def failure(error):
            value=completed[0]
            if isinstance(error,GeometryTaskCancelled) and isinstance(value,dict) and value.get('state') in ('backed_up','restored','aborted'):
                success(value);self.status.setText(self.status.text()+' 取消到达前操作已完成，已发布结果仍保留。')
            else:
                message='未完成：'+str(error)
                if storage.pending_restores(self.workspace):message+=' 请先继续或中止未完成的数据恢复。'
                self.status.setText(message);self.details.setPlainText(str(error))
        self.runner.start(work,success,failure);self.sync_controls()

    def inspect_data(self):
        key=self.operation.currentData();path=self.path.text();roots=self.roots_value()
        if key=='backup':
            def inspect(check,progress):
                progress('检查历史和关联资产')
                value=inspect_references(self.workspace,check=check)
                value['outside']=[p for p in value['references'] if Path(p).exists() and not any(Path(p).is_relative_to(root) for root in roots.values())]
                return value
            def show(value):
                self.status.setText('关联资产%d项；原已缺失%d项；另有%d项需添加数据目录。' % (len(value['references']),len(value['missing']),len(value['outside'])))
                self.details.setPlainText(json.dumps(value,ensure_ascii=False,indent=2))
            self.job(inspect,show);return
        if not path:self.status.setText('请先选择备份目录或恢复记录。');return
        def read(check,progress):
            progress('核对备份与记录')
            if key in ('verify','restore'):
                manifest=storage.verify(path,check=check)
                targets={n:self.workspace/n if n in ('outputs','datasets','photos','models') else Path(v['origin']) for n,v in manifest['roots'].items()}
                return dict(targets={n:str(p) for n,p in targets.items()},audit=manifest.get('reference_audit',{}),manifest_sha=manifest['_verified_sha256'])
            state=json.loads(Path(path).read_text('utf-8'))
            return dict(targets=state['destinations'],audit=dict(journal_state=state['state']))
        def show(value):
            self.set_roots(value['targets']);self._loaded_path=path
            self._manifest_sha=value.get('manifest_sha')
            missing=len(value['audit'].get('unavailable',[]))
            self.status.setText(('备份校验通过。' if key in ('verify','restore') else '已读取恢复记录。')+
                               (' 原数据已有%d项缺失，此备份无法补回。'%missing if missing else '')+
                               (' 恢复目标必须是空位置（目录尚不存在）。' if key=='restore' else ''))
            self.details.setPlainText(json.dumps(value,ensure_ascii=False,indent=2));self.sync_controls()
        self.job(read,show)

    def preview_disk(self):
        roots={self.roots.topLevelItem(i).data(0,Qt.UserRole):Path(self.roots.topLevelItem(i).text(1)).absolute()
               for i in range(self.roots.topLevelItemCount())}
        def show(value):
            total=value['logical_bytes']/1024**3
            free='；'.join('可用 %.1f GiB%s'%(v['free_bytes']/1024**3,'（空间偏低）' if v['low_space'] else '') for v in value['volumes'])
            review=value['cleanup']['recovery_review']['files']
            self.status.setText(('统计完成' if value['scan_complete'] else '统计不完整，请查看详情')+
                '：已统计文件 %.2f GiB；%s。%d项暂存需核对，未删除文件。'%(total,free or '未能读取磁盘容量',review))
            self.details.setPlainText(json.dumps(value,ensure_ascii=False,indent=2))
            self.detail_toggle.setChecked(True)
        self.job(lambda check,progress:inspect_disk(roots,check=check,progress=progress),show)

    def execute(self):
        key=self.operation.currentData();text=self.path.text().strip();roots=self.roots_value();allow=self.allow_missing.isChecked()
        expected_manifest_sha=self._manifest_sha
        if not text:self.status.setText('请先选择操作位置。');return
        if key=='verify':self.inspect_data();return
        def work(check,progress):
            if key=='backup':
                value=backup_workspace(self.workspace,text,roots=roots,check=check,progress=progress,allow_existing_missing=allow)
                audit=value['manifest']['reference_audit']
                return dict(state='backed_up',destination=value['destination'],unavailable_reference_count=len(audit['unavailable']))
            if key=='restore':return storage.restore_new(text,roots,workspace=self.workspace,check=check,
                                                       expected_manifest_sha=expected_manifest_sha)
            if key=='recover':return storage.recover_restore(text,roots,workspace=self.workspace,check=check)
            return storage.abort_restore(text,roots,workspace=self.workspace)
        def show(value):
            missing=value.get('unavailable_reference_count',0)
            label=dict(backed_up='备份已保存并校验。',restored='恢复完成。',aborted='恢复已中止，暂存文件保留。')[value['state']]
            if missing:label+=' 原数据已有%d项关联缺失，未补造文件。'%missing
            if value.get('original_locations') is False:label+=' 原来源路径未重写，文件迁移仍需单独核验。'
            self.status.setText(label);self.details.setPlainText(json.dumps(value,ensure_ascii=False,indent=2))
        self.job(work,show,cancellable=key!='abort')

    def request_cancel(self):
        if not self._cancellable:
            self.status.setText('正在核对并保留恢复现场，完成后即可关闭。');return
        self.runner.cancel();self.cancel.setEnabled(False);self.status.setText('正在取消；请等待文件发布或回滚结束。')

    def finished_job(self):
        self.sync_controls()
        if self._closing:self.close()

    def closeEvent(self,event):
        if self.runner.busy:
            self._closing=True;self.request_cancel();event.ignore()
        else:event.accept()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=REPO_ROOT);args=parser.parse_args()
    app=QApplication(sys.argv[:1])
    from femto.app.qt_main import apply_theme,_logs
    _logs()
    apply_theme(app)
    dialog=DataMaintenanceDialog(args.workspace);dialog.show()
    return app.exec()


if __name__=='__main__':raise SystemExit(main())
