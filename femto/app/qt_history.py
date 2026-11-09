"""Qt history views over the existing capture and planning catalogues.

Browsing never updates the active capture, latest-result marker or source assets.
Opening a record delegates to the Qt workspace through ``planning_requested``.
"""
from __future__ import annotations

from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import sys
import time

from PySide6.QtCore import Qt, QTimer, Signal, QSize, QRectF, QEvent
from PySide6.QtGui import QImageReader, QPixmap, QColor, QPainter, QPen, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTabWidget, QComboBox, QCheckBox,
    QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QStyledItemDelegate, QStyle, QMenu,
)

from femto.app.qt_common import TaskRunner, open_path
from femto.app.qt_icons import line_icon
from femto.paths import REPO_ROOT
from femto.history_store import HistoryStore
from femto.pipeline.local_history import existing_actions, scan_history, set_records_deleted
from femto.planning.catalog import local_asset, scan_saved_versions, scan_capture_exports
from femto.planning.planning_io import TaskControl


_STATES = {
    "model": "模型已生成", "failed": "采集失败", "lost": "跟踪丢失",
    "recorded": "原始采集", "incomplete": "文件待检查", "unpublished": "结果未发布",
}
_INVALID = {"损坏", "文件丢失"}


class RecordBadgeDelegate(QStyledItemDelegate):
    """Paint compact catalogue badges while retaining full evidence text."""
    def paint(self, painter, option, index):
        if index.column() not in (1, 3):
            return super().paint(painter, option, index)
        row = index.siblingAtColumn(0).data(Qt.UserRole) or {}
        text = index.data() or ''
        color = '#87b8d4'
        if index.column() == 1:
            text = {'区域工程 / 方案':'区域工程', '三维表面轨迹':'表面轨迹'}.get(text,text)
        elif row.get('file_status') in _INVALID or any(word in text for word in ('损坏','失败','丢失','缺失')):
            text, color = '文件待检查', '#f3b957'
        elif '待工作台重算' in text:
            text, color = '指纹一致 · 待重算', '#f3b957'
        elif row.get('file_status') == '轨迹资产指纹一致':
            text, color = '指纹一致 · 只读', '#7bcde8'
        elif '指纹一致' in text:
            text, color = '资产指纹一致', '#6ed4c4'
        else:
            color = '#f3b957'
        # Let the native view draw row selection/hover before the badge.
        style_option = type(option)(option)
        self.initStyleOption(style_option,index)
        style_option.text = '';style_option.icon = QIcon()
        option.widget.style().drawControl(QStyle.CE_ItemViewItem,style_option,painter,option.widget)
        painter.save();painter.setRenderHint(QPainter.Antialiasing)
        bounds = QRectF(option.rect).adjusted(10,12,-10,-12)
        width = min(bounds.width(),option.fontMetrics.horizontalAdvance(text)+24)
        bounds.setWidth(width)
        painter.setPen(QPen(QColor(color), .7));painter.setBrush(QColor('#103349'))
        painter.drawRoundedRect(bounds,5,5)
        painter.setPen(QColor(color))
        text = option.fontMetrics.elidedText(text,Qt.ElideRight,max(0,int(width)-20))
        painter.drawText(bounds,Qt.AlignCenter,text);painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option,index);size.setHeight(max(54,size.height()))
        return size


class RowActionDelegate(QStyledItemDelegate):
    """Draw lightweight per-row actions without creating thousands of widgets."""
    def __init__(self, dialog, tree):
        super().__init__(tree);self.dialog=dialog;self.tree=tree

    def paint(self,painter,option,index):
        super().paint(painter,option,index)
        idle=not (self.dialog.busy or self.dialog.opening)
        painter.save();painter.setRenderHint(QPainter.Antialiasing)
        bounds=QRectF(option.rect).adjusted(7,10,-7,-10)
        main=QRectF(bounds);main.setWidth(bounds.width()-34)
        painter.setPen(QPen(QColor('#2c718f' if idle else '#294453'),1))
        painter.setBrush(QColor('#10394e'));painter.drawRoundedRect(main,4,4)
        painter.setPen(QColor('#a7e4fa' if idle else '#657c8c'))
        painter.drawText(main,Qt.AlignCenter,'查看')
        cx=bounds.right()-12;cy=bounds.center().y()
        for delta in (-5,0,5):painter.drawEllipse(QRectF(cx+delta-1,cy-1,2,2))
        painter.restore()

    def editorEvent(self,event,model,option,index):
        if event.type()==QEvent.MouseButtonRelease and event.button()==Qt.LeftButton:
            if self.dialog.busy or self.dialog.opening:return True
            item=self.tree.itemFromIndex(index);self.tree.setCurrentItem(item)
            if event.position().x()>=option.rect.right()-38:
                menu=self.dialog.row_menu(self.tree,item)
                menu.exec(self.tree.viewport().mapToGlobal(option.rect.bottomRight()))
            else:self.dialog.open_selected()
            return True
        return False

    def sizeHint(self,option,index):
        return QSize(112,54)


def launch_data_maintenance(root):
    import subprocess
    return subprocess.Popen([sys.executable,'-m','femto.app.qt_data_maintenance','--workspace',str(root)],
                            cwd=REPO_ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))


class _ReadControl(TaskControl):
    def __init__(self, stage, cancel):
        super().__init__(stage)
        self.cancel_event = cancel

    def is_set(self):
        return self.cancel_event.is_set() or super().is_set()


def read_report(path: str | Path) -> tuple[str, bool]:
    """Read a bounded, local report without starting a browser or legacy UI."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("报告已移动或删除，请刷新记录。")
    if source.stat().st_size > 10 * 1024**2:
        raise ValueError("报告超过 10 MiB，请从文件夹中查看完整文件。")
    text = source.read_text(encoding="utf-8-sig")
    if source.suffix.lower() == ".json":
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    return text, source.suffix.lower() in {".html", ".htm"}


def reference_image(row: dict) -> Path | None:
    """Locate the first recorded RGB file; refuse paths outside its dataset."""
    candidates = []
    if row.get("result"):
        report = Path(row["result"]) / "reconstruction" / "reconstruction_report.json"
        try:
            data = json.loads(report.read_text(encoding="utf-8-sig"))
            if data.get("source_dataset"):
                candidates.append(Path(data["source_dataset"]))
        except (OSError, ValueError, TypeError):
            pass
    if row.get("source"):
        candidates.append(Path(row["source"]))
    for dataset in candidates:
        try:
            with (dataset / "frames.jsonl").open(encoding="utf-8-sig") as stream:
                for _ in range(10):
                    line = stream.readline(1024 * 1024)
                    if not line:
                        break
                    if line.strip():
                        frame = json.loads(line)
                        image = local_asset(dataset, frame.get("rgb"))
                        if image:
                            return image
                        break
        except (OSError, ValueError, AttributeError):
            continue
    return None


def record_target(row: dict) -> Path | None:
    """Return the exact saved record to reopen, never a similarly named file."""
    if row.get("file_status") in _INVALID:
        return None
    if row.get("kind") == "区域工程 / 方案":
        target = row.get("project")
    elif row.get("kind") == "区域":
        from femto.planning.trajectory_history import region_manifest
        try:
            target = region_manifest(row)
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return None
    elif row.get("kind") in {"三维表面轨迹", "区域＋路径"}:
        target = row.get("target") or row.get("source")
    else:
        return None
    try:
        path = Path(target) if target else None
        return path if path and path.is_file() else None
    except (OSError, TypeError, ValueError):
        # A damaged index can contain a non-path value or an invalid path;
        # malformed records must stay non-openable instead of breaking selection.
        return None


class ReportDialog(QDialog):
    def __init__(self, parent=None, *, title="记录详情", text="", html=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(880, 620)
        self.setMinimumSize(540, 360)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(False)
        self.browser.setOpenLinks(False)
        self.browser.setHtml(text) if html else self.browser.setPlainText(text)
        layout.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ImagePreviewDialog(QDialog):
    def __init__(self, path: str | Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("记录画面 · " + Path(path).name)
        self.resize(1024, 720)
        layout = QVBoxLayout(self)
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError("图片无法读取：" + reader.errorString())
        self.original = QPixmap.fromImage(image)
        self.picture = QLabel(self)
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setMinimumSize(1, 1)
        layout.addWidget(self.picture, 1)
        caption = QLabel(str(Path(path)), self)
        caption.setWordWrap(True)
        caption.setProperty("role", "secondary")
        layout.addWidget(caption)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "original"):
            self.picture.setPixmap(self.original.scaled(
                self.picture.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))


class HistoryDialog(QDialog):
    """Sessions and saved plans, using the same evidence gates as the old GUI."""

    planning_requested = Signal(str)
    cancel_open_requested = Signal()

    def __init__(self, parent=None, root: str | Path | None = None):
        super().__init__(parent)
        requested = Path(root or REPO_ROOT).resolve()
        self.setObjectName('historyDialog')
        focus = requested if requested.name == "reconstruction" else None
        # A reconstruction path is accepted by callers that open history for a
        # specific result. Keep that result's project root instead of falling
        # back to the installation repository; the latter made temporary or
        # imported projects silently scan the wrong datasets/outputs folders.
        self.root = requested
        if focus is not None and focus.parent.parent.name == "outputs":
            self.root = focus.parent.parent.parent
        self.datasets = self.root / "datasets"
        self.outputs = self.root / "outputs"
        self.store = HistoryStore(self.outputs)
        self._collections = {}
        self._search_queries = {0: "", 1: ""}
        self._search_tab = 0
        self._mutating = False
        self.reconstruction = focus
        self.sessions: list[dict] = []
        self.records: list[dict] = []
        self.deleted_stack: list[list[dict]] = []
        self.busy = False
        self.opening = False
        self._open_cancelling = False
        self._closed = False
        self._closing_requested = False
        self.runner = TaskRunner(self)
        self.runner.finished.connect(self._finished)
        self._children: list[QDialog] = []
        self.setWindowTitle("历史记录 · 三维采集与规划")
        self.resize(1120, 730)
        self.setMinimumSize(760, 500)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        heading = QHBoxLayout()
        title = QLabel("历史记录", self)
        title.setObjectName('historyTitle')
        emblem = QLabel();emblem.setPixmap(line_icon('history', True).pixmap(42,42))
        heading.addWidget(emblem)
        title_column = QVBoxLayout();title_column.setSpacing(3)
        title_column.addWidget(title)
        subtitle = QLabel('查看采集记录、区域与规划状态，管理数据备份与恢复。')
        subtitle.setProperty('role','secondary');title_column.addWidget(subtitle)
        heading.addLayout(title_column)
        heading.addStretch()
        self.maintenance_button=QPushButton('数据备份与恢复',self)
        self.maintenance_button.setToolTip('打开独立维护窗口；开始备份或恢复前请关闭工作台和数据处理工具。')
        self.maintenance_button.clicked.connect(self.open_maintenance)
        heading.addWidget(self.maintenance_button)
        self.refresh_button = QPushButton("刷新", self)
        self.refresh_button.clicked.connect(self.refresh)
        heading.addWidget(self.refresh_button)
        layout.addLayout(heading)
        self.search = QLineEdit(self)
        self.search.setObjectName('historySearch')
        self.search.addAction(line_icon('search'), QLineEdit.LeadingPosition)
        self.search.setPlaceholderText("搜索采集记录、区域名称或规划状态")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.removed = QCheckBox('已移除记录', self)
        self.removed.toggled.connect(self.refresh)
        layout.addWidget(self.removed)
        self.scope=QComboBox()
        self.scope.addItem('当前采集的区域与轨迹','capture')
        self.scope.addItem('全部已保存方案版本','all')
        self.scope.currentIndexChanged.connect(lambda *_:self._tab_changed(self.tabs.currentIndex()))
        self.scope.setVisible(False)
        layout.addWidget(self.scope)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName('historyTabs')
        self.session_tree = self._tree(["采集时间", "状态", "帧数", "采集记录"])
        self.record_tree = self._tree(["保存时间", "类型", "名称", "核验状态"])
        self.record_tree.setItemDelegate(RecordBadgeDelegate(self.record_tree))
        self.record_tree.setTextElideMode(Qt.ElideRight)
        record_header = self.record_tree.header()
        for column, width in ((0,185),(1,145),(3,205)):
            record_header.setSectionResizeMode(column,QHeaderView.Interactive)
            self.record_tree.setColumnWidth(column,width)
        record_header.setStretchLastSection(False)
        record_header.setSectionResizeMode(2,QHeaderView.Stretch)
        self.tabs.addTab(self.session_tree, "采集记录")
        self.tabs.addTab(self.record_tree, "区域与轨迹")
        self.tabs.setTabIcon(0,line_icon('cube'));self.tabs.setTabIcon(1,line_icon('document'))
        self.tabs.currentChanged.connect(self._tab_changed)
        self.session_tree.itemSelectionChanged.connect(self._selection_changed)
        self.record_tree.itemSelectionChanged.connect(self._selection_changed)
        self.session_tree.itemDoubleClicked.connect(lambda item,column: self.open_selected() if column!=4 else None)
        self.record_tree.itemDoubleClicked.connect(lambda item,column: self.open_selected() if column!=4 else None)
        layout.addWidget(self.tabs, 1)
        self.details = QLabel("选择记录后查看来源、报告或继续规划。", self)
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setProperty("role", "secondary")
        self.details.setTextFormat(Qt.PlainText)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.preview_button = QPushButton("查看画面", self)
        self.preview_button.clicked.connect(self.preview_selected)
        actions.addWidget(self.preview_button)
        self.report_button = QPushButton('查看报告', self)
        self.report_button.clicked.connect(self.report_selected)
        actions.addWidget(self.report_button)
        self.folder_button = QPushButton('打开文件夹', self)
        self.folder_button.clicked.connect(self.open_folder)
        actions.addWidget(self.folder_button)
        self.source_toggle = QPushButton('来源详情', self)
        self.source_toggle.setCheckable(True)
        self.source_toggle.toggled.connect(self._selection_changed)
        actions.addWidget(self.source_toggle)
        actions.addStretch()
        layout.addLayout(actions)
        footer = QHBoxLayout()
        self.hide_button = QPushButton("从列表移除", self)
        self.undo_button = QPushButton("撤销移除", self)
        self.hide_button.clicked.connect(self.hide_selected)
        self.undo_button.clicked.connect(self.undo_hide)
        footer.addWidget(self.hide_button)
        footer.addWidget(self.undo_button)
        self.purge_button = QPushButton('永久删除', self)
        self.purge_button.setIcon(line_icon('trash'))
        self.purge_button.setProperty('tone','danger')
        self.purge_button.setToolTip('永久删除所选记录的文件，无法恢复。')
        self.purge_button.clicked.connect(self.purge_selected)
        footer.addWidget(self.purge_button)
        footer.addStretch()
        self.cancel_button = QPushButton("取消读取", self)
        self.cancel_button.clicked.connect(self.cancel_read)
        footer.addWidget(self.cancel_button)
        self.open_button = QPushButton("打开规划", self)
        self.open_button.setProperty("role", "primary")
        self.open_button.setMinimumHeight(48)
        self.open_button.clicked.connect(self.open_selected)
        footer.addWidget(self.open_button)
        layout.addLayout(footer)
        self.status = QLabel("", self)
        self.status.setWordWrap(True)
        self.status.setProperty("role", "secondary")
        layout.addWidget(self.status)
        self._selection_changed()
        for button, name in ((self.maintenance_button,'database'),(self.refresh_button,'refresh'),
                (self.preview_button,'eye'),(self.report_button,'document'),(self.folder_button,'folder'),
                (self.source_toggle,'info'),(self.hide_button,'trash'),(self.undo_button,'undo'),
                (self.cancel_button,'close'),(self.open_button,'arrow')):
            button.setIcon(line_icon(name));button.setIconSize(QSize(20,20))
        QTimer.singleShot(0, self.refresh)

    def _tree(self, headings):
        tree = QTreeWidget(self)
        tree.setObjectName('historyTable')
        tree.setColumnCount(len(headings)+1)
        tree.setHeaderLabels([*headings,'操作'])
        tree.setItemDelegateForColumn(4,RowActionDelegate(self,tree))
        tree.setContextMenuPolicy(Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(lambda pos:self._row_context_menu(tree,pos))
        tree.setRootIsDecorated(False)
        tree.setAlternatingRowColors(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tree.setUniformRowHeights(True)
        tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tree.header().setSectionResizeMode(len(headings)-1, QHeaderView.ResizeMode.Stretch)
        tree.header().setStretchLastSection(False)
        tree.header().setSectionResizeMode(4,QHeaderView.Fixed)
        tree.setColumnWidth(4,112)
        tree.setMinimumHeight(160)
        return tree

    def _start(self, label, work, done, *, mutating=False):
        if self.busy or self.opening or self._closed:
            return
        self.busy = True
        self._mutating = mutating
        self.status.setText(label + "…")
        self._selection_changed()

        def run(cancel, progress):
            control = _ReadControl(label, cancel)
            value = work(control)
            control.check()
            return value

        def complete(value):
            self.busy = False
            self._mutating = False
            if not self._closing_requested:
                done(value)
            self._selection_changed()

        def failed(error):
            self.busy = False
            self._mutating = False
            self._selection_changed()
            self.status.setText(("操作未完成，当前列表保留：" if mutating else
                                 "读取未完成，当前列表保留：") + str(error))

        self.runner.start(run, complete, failed)

    def _finished(self):
        if self._closing_requested:
            self.close()

    def refresh(self):
        if self._closed or self.opening:
            return
        if self.tabs.currentIndex() == 1:
            if self.scope.currentData()=='all':
                self._load_exports(self.outputs/'saved_projects')
                return
            selected = self._session_reconstruction(self._selected_session())
            reconstruction = selected or self.reconstruction
            if reconstruction:
                self._load_exports(reconstruction)
            else:
                self._tab_changed(1)
            return
        key = (self.current() or {}).get("key")
        removed = self.removed.isChecked()

        def read(control):
            from femto.dataset.task_context import read_task
            rows = scan_history(self.datasets, self.outputs, cancel=control, include_hidden=True)
            enriched = []
            for row in rows:
                control.check()
                try:
                    task = read_task(row['result']) if row.get('result') else None
                except (OSError, ValueError, KeyError, TypeError):
                    task = None
                if task:
                    row = dict(row, task_context=task,
                               name=task['name'])
                enriched.append(row)
            self.store.replace('captures', enriched, search_text=self._search_text, check=control.check)
            return self.store.query('captures', hidden_only=removed, check=control.check)['rows']

        def done(rows):
            self._collections[self.session_tree] = ('captures', removed)
            self.sessions = rows
            self.session_tree.clear()
            selection = None
            for row in rows:
                item = QTreeWidgetItem([
                    datetime.fromtimestamp(row["time"]).strftime("%Y-%m-%d %H:%M:%S"),
                    _STATES.get(row["state"], row["state"]),
                    str(row["count"]) if row.get("count") is not None else "—", row["name"],
                ])
                item.setData(0, Qt.ItemDataRole.UserRole, row)
                item.setIcon(1,line_icon('shield' if row['state']=='model' else 'warning'))
                item.setData(0, Qt.ItemDataRole.UserRole + 1, self.store.identity(row))
                self.session_tree.addTopLevelItem(item)
                if row["key"] == key:
                    selection = item
            if rows:
                self.session_tree.setCurrentItem(selection or self.session_tree.topLevelItem(0))
            self._filter()
            self.status.setText(f"共 {len(rows)} 条" + ("已移除记录" if removed else "采集记录"))
            self._selection_changed()
            if self.tabs.currentIndex() == 1:
                self._tab_changed(1)

        self._start("读取采集历史", read, done)

    @staticmethod
    def _search_text(row):
        if 'kind' not in row:
            return ' '.join((datetime.fromtimestamp(row['time']).strftime('%Y-%m-%d %H:%M:%S'),
                             _STATES.get(row['state'], row['state']), str(row.get('count', '')),
                             row.get('name', '')))
        return ' '.join(str(row.get(key) or '') for key in
                        ('time', 'kind', 'name', 'status', 'project_id', 'revision_id', 'parent_revision_id'))

    def current(self):
        tree = self.session_tree if self.tabs.currentIndex() == 0 else self.record_tree
        item = tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item and not item.isHidden() else None

    def _selected_session(self):
        """Return the session selection even while the exports tab is visible."""
        item = self.session_tree.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item and not item.isHidden() else None

    @staticmethod
    def _session_reconstruction(row):
        """Resolve the exports root belonging to one capture-history row."""
        result = row.get("result") if row else None
        if not result:
            return None
        return Path(result) / "reconstruction"

    def _clear_exports(self, message):
        """Drop exports that belong to a different or unavailable session."""
        self.reconstruction = None
        self.records = []
        self._collections.pop(self.record_tree, None)
        self.record_tree.clear()
        self.status.setText(message)
        self._selection_changed()

    def _tab_changed(self, index):
        """Load exports for the capture row that the user selected first.

        The two tabs share one history dialog, so switching to "区域与轨迹"
        should follow the session selection instead of reusing a previously
        displayed reconstruction (or waiting for the separate action button).
        """
        self.scope.setVisible(index==1)
        self.removed.setVisible(index==0)
        if index != self._search_tab:
            # Each list keeps its own query. A plan-only search must never
            # remove the capture selection that defines its catalogue.
            self._search_tab = index
            blocked = self.search.blockSignals(True)
            try:
                self.search.setText(self._search_queries[index])
            finally:
                self.search.blockSignals(blocked)
        self._filter()
        if index != 1:
            return
        if self.scope.currentData()=='all':
            if not self.busy:self._load_exports(self.outputs/'saved_projects')
            return
        session = self._selected_session()
        reconstruction = self._session_reconstruction(session)
        if reconstruction and reconstruction.is_dir():
            current = Path(self.reconstruction).resolve() if self.reconstruction else None
            if current != reconstruction.resolve() or not self.records:
                self._clear_exports("正在读取当前采集记录的区域与轨迹数据…")
                self._load_exports(reconstruction)
                return
            self._selection_changed()
            return
        # A failed or source-only capture has no region/trajectory catalogue.
        # Clear stale rows so the exports tab cannot appear to belong to the
        # previously selected session.
        self._clear_exports("当前采集记录没有可用的区域与轨迹数据。")

    def _filter(self, *_):
        index = self.tabs.currentIndex()
        self._search_queries[index] = self.search.text()
        query = self.search.text().strip().casefold()
        tree = self.session_tree if index == 0 else self.record_tree
        matches = None
        # Reads never wait for a writer. Keep the previous filter if the index
        # is unavailable or exceeds the UI budget, and expose the failure.
        deadline = time.monotonic() + .05
        def check():
            if time.monotonic() > deadline:
                raise TimeoutError('搜索超时，请缩短搜索条件后重试。')
        try:
            if tree in self._collections:
                collection, removed = self._collections[tree]
                matches = self.store.matching_identities(
                    collection, query, hidden_only=removed, check=check)
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as error:
            self.status.setText('搜索未完成，当前列表保留：' + str(error))
            return
        for position in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(position)
            if matches is not None:
                identity = item.data(0, Qt.ItemDataRole.UserRole + 1)
                if identity is None:
                    identity = self.store.identity(item.data(0, Qt.ItemDataRole.UserRole))
                    item.setData(0, Qt.ItemDataRole.UserRole + 1, identity)
                item.setHidden(identity not in matches)
            else:
                item.setHidden(bool(query and query not in " ".join(
                    item.text(col) for col in range(tree.columnCount())).casefold()))
        current = tree.currentItem()
        if current is None or current.isHidden():
            visible = next((tree.topLevelItem(i) for i in range(tree.topLevelItemCount())
                            if not tree.topLevelItem(i).isHidden()), None)
            tree.setCurrentItem(visible)
        self._selection_changed()

    def _selection_changed(self, *_):
        row = self.current()
        session = self.tabs.currentIndex() == 0
        idle = not (self.busy or self.opening or self._closing_requested)
        ready = bool(row and idle)
        for widget in (self.search, self.scope, self.tabs):
            widget.setEnabled(not self.opening and not self._closing_requested)
        actions = existing_actions(row) if row and session else {}
        valid = bool(ready and row.get("file_status") not in _INVALID)
        target = record_target(row) if valid and not session else None
        self.open_button.setEnabled(bool(ready and (actions.get("fixed") if session else target)))
        self.open_button.setText("正在核验所选记录…" if self.opening else "查看采集结果" if session else "只读查看所选版本")
        if self.open_button.isEnabled():
            open_hint = ""
        elif not idle:
            open_hint = "请等待当前读取或取消操作结束。"
        elif not row:
            open_hint = "请先从列表中选择一条记录；没有匹配结果时可调整搜索条件。"
        else:
            open_hint = ("需要有效参考图像、标定与来源记录；文件变化后请先刷新核对。" if session else
                         "记录缺失、损坏或没有可恢复的区域清单。可查看报告和文件位置。")
        self.open_button.setToolTip(open_hint)
        self.refresh_button.setEnabled(idle)
        self.removed.setEnabled(idle)
        cancelling = self._open_cancelling or self.runner.cancelling
        self.cancel_button.setText("正在取消…" if cancelling else "取消读取")
        self.cancel_button.setEnabled((self.busy or self.opening) and not cancelling)
        self.cancel_button.setVisible((self.busy or self.opening) and not self._mutating)
        self.hide_button.setEnabled(ready and session)
        self.hide_button.setText('恢复记录' if self.removed.isChecked() else '从列表移除')
        self.purge_button.setEnabled(ready and session and self.removed.isChecked())
        self.undo_button.setEnabled(bool(self.deleted_stack) and idle)
        for button in (self.preview_button,self.report_button,self.folder_button,self.source_toggle,
                       self.hide_button,self.undo_button,self.purge_button,self.open_button):
            button.hide()
        self.preview_button.setEnabled(bool(valid and (reference_image(row) if session else row.get("diagram"))))
        self.report_button.setEnabled(ready)
        self.folder_button.setEnabled(ready)
        self.source_toggle.setEnabled(bool(row))
        self.source_toggle.setText('收起来源' if self.source_toggle.isChecked() else '来源详情')
        if not row:
            self.details.setText("未找到匹配记录。" if self.search.text() else "暂无记录。" if not self.busy else "")
            self.details.setToolTip('')
            return
        if session:
            text = "模型：" + (row.get("result") or "未生成") + "\n采集：" + (row.get("source") or "未记录")
            if not actions.get("fixed"):
                reasons = {
                    "origin_unverified": "参考位置来源未通过核对。",
                    "invalid_calibration": "参考标定无效。",
                    "source_budget": "来源超出历史预检预算。",
                    "unpublished_result": "结果尚未成功发布，不能进入规划。",
                    "missing_source_files": "参考图像或来源文件缺失。",
                }
                text += "\n" + reasons.get(row.get("fixed_reason"), "参考图像、模型或来源文件不可用。")
            if row.get("error"):
                text += "\n读取提示：" + str(row["error"])
        else:
            text = row.get("status", "") + "\n位置：" + str(row.get("folder", ""))
            if row.get('revision_id'):
                text += '\n版本：' + str(row['revision_id'])
            if row.get('parent_revision_id'):
                text += '\n上一版本：' + str(row['parent_revision_id'])
            if row.get("note"):
                text += "\n备注：" + row["note"]
        self.details.setToolTip(text)
        if not self.source_toggle.isChecked():
            text = '\n'.join(line for line in text.splitlines()
                             if not line.startswith(('模型：', '采集：', '位置：', '版本：', '上一版本：')))
        self.details.setText(text.strip())

    def show_exports(self):
        row = self._selected_session()
        reconstruction = self._session_reconstruction(row)
        if reconstruction and not self.busy:
            if self.tabs.currentIndex() != 1:
                self.tabs.setCurrentIndex(1)
            else:
                self._load_exports(reconstruction)

    def _load_exports(self, reconstruction):
        reconstruction = Path(reconstruction).resolve()
        collection = 'exports:' + json.dumps(self.store.location(reconstruction), ensure_ascii=False)
        selected_item = self.record_tree.currentItem()
        previous = selected_item.data(0, Qt.ItemDataRole.UserRole) if selected_item else None
        def identity(row):
            return tuple(str(row.get(key, '')) for key in
                         ('project', 'source', 'folder', 'revision_id', 'kind', 'name'))
        previous_identity = identity(previous) if previous else None

        def done(rows):
            # The user can change the session while a large catalogue is being
            # verified. Never let the slower result overwrite the newly
            # selected session, including when the capture tab is now visible.
            selected = (self.outputs/'saved_projects' if self.scope.currentData()=='all'
                        else self._session_reconstruction(self._selected_session()))
            selected = selected.resolve() if selected else None
            if selected != reconstruction:
                viewing_exports = self.tabs.currentIndex() == 1
                message = ("正在切换到当前采集记录的区域与轨迹数据…" if selected else
                           "当前采集记录没有可用的区域与轨迹数据。")
                self._clear_exports(message if viewing_exports else f"共 {len(self.sessions)} 条采集记录")
                if selected is not None and viewing_exports:
                    self._load_exports(selected)
                return
            self.reconstruction = reconstruction
            self._collections[self.record_tree] = (collection, False)
            self.records = rows
            self.record_tree.clear()
            selection = None
            for row in rows:
                version=row.get('revision_id')
                name=row.get('name','')+((' · 版本 '+str(version)[:8]) if version else '')
                item = QTreeWidgetItem([row.get("time", ""), row["kind"], name, row.get("status", "")])
                item.setData(0, Qt.ItemDataRole.UserRole, row)
                item.setData(0, Qt.ItemDataRole.UserRole + 1, self.store.identity(row))
                item.setToolTip(3, row.get("status", ""))
                item.setToolTip(1,row['kind'])
                item.setToolTip(2,name)
                item.setIcon(2,line_icon('document' if row['kind']=='区域工程 / 方案' else 'cube'))
                self.record_tree.addTopLevelItem(item)
                if previous_identity is not None and identity(row) == previous_identity:
                    selection = item
            if rows:
                self.record_tree.setCurrentItem(selection or self.record_tree.topLevelItem(0))
            self._filter()
            if self.tabs.currentIndex() == 1:
                self.status.setText(f"共 {len(rows)} 条区域与轨迹记录")
            else:
                self.status.setText(f"共 {len(self.sessions)} 条采集记录")
            self._selection_changed()

        def read(control):
            rows = (scan_saved_versions(self.outputs, check=control.check)
                    if reconstruction == self.outputs/'saved_projects'
                    else scan_capture_exports(reconstruction, check=control.check))
            self.store.replace(collection, rows, search_text=self._search_text, check=control.check)
            return self.store.query(collection, check=control.check)['rows']
        self._start("核验区域与轨迹", read, done)

    def open_maintenance(self):
        try:launch_data_maintenance(self.root)
        except OSError as error:self.status.setText('无法打开数据维护：'+str(error))

    def show_all_plans(self):
        self.scope.setCurrentIndex(1)
        self.tabs.setCurrentIndex(1)

    def open_selected(self):
        row = self.current()
        if not row or self.busy or self.opening or self._closed:
            return
        if self.tabs.currentIndex() == 0:
            if not existing_actions(row).get("fixed"):
                self.status.setText("参考文件已变化或不可用，请刷新后检查记录。")
                self._selection_changed()
                return
            target = Path(row["result"]) / "reconstruction"
        else:
            target = record_target(row)
            if not target:
                self.status.setText("所选记录不可恢复，请检查核验状态。")
                self._selection_changed()
                return
        self.planning_requested.emit(str(target))

    def _show_dialog(self, dialog):
        self._children.append(dialog)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.destroyed.connect(lambda *_: self._children.remove(dialog) if dialog in self._children else None)
        dialog.show()

    def preview_selected(self):
        row = self.current()
        if not row or self.busy or row.get("file_status") in _INVALID:
            return
        path = reference_image(row) if self.tabs.currentIndex() == 0 else row.get("diagram")
        try:
            if not path:
                raise ValueError("此记录未保存预览图片。")
            self._show_dialog(ImagePreviewDialog(path, self))
        except (OSError, ValueError) as error:
            self.status.setText("无法显示画面：" + str(error))

    def report_selected(self):
        row = self.current()
        if not row or self.busy:
            return
        if self.tabs.currentIndex() == 1 and row.get("kind") == "三维表面轨迹" and row.get("file_status") not in _INVALID:
            from femto.planning.trajectory_history import prepare_report
            self._start("核验规划报告", lambda control: prepare_report(row, control),
                lambda text: self._show_dialog(ReportDialog(self, title="规划检查", text=text)))
            return
        path = row.get("report") if self.tabs.currentIndex() == 0 else row.get("source")
        try:
            text, html = read_report(path)
            if row.get("status"):
                text = row["status"] + "\n\n" + text
            self._show_dialog(ReportDialog(self, title="记录报告", text=text, html=html))
        except (OSError, ValueError, TypeError) as error:
            self.status.setText("无法读取报告：" + str(error))

    def open_folder(self):
        row = self.current()
        if not row or self.busy:
            return
        path = row.get("result") or row.get("source") if self.tabs.currentIndex() == 0 else row.get("folder")
        try:
            if not path or not Path(path).is_dir():
                raise FileNotFoundError("记录目录已移动或删除。")
            open_path(path)
        except (OSError, ValueError) as error:
            self.status.setText("无法打开文件夹：" + str(error))

    def hide_selected(self):
        row = self.current()
        if not row or self.busy or self.opening or self.tabs.currentIndex() != 0:
            return
        restoring = self.removed.isChecked()
        if not restoring:
            answer = QMessageBox.question(self, "从列表移除", "从历史列表移除此记录？\n\n" + row["name"] +
                "\n\n原始采集、模型和标记文件会保留，可在“已移除记录”中恢复。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        def done(_):
            if not restoring:
                self.deleted_stack.append([dict(row)])
            self.refresh()
        self._start('恢复记录' if restoring else '移除记录',
                    lambda control: set_records_deleted([row], self.datasets, self.outputs,
                        not restoring, check=control.check), done, mutating=True)

    def _row_context_menu(self,tree,position):
        item=tree.currentItem() if position.x()<0 or position.y()<0 else tree.itemAt(position)
        if item is not None and not self.busy and not self.opening:
            anchor=tree.visualItemRect(item).center() if position.x()<0 else position
            self.row_menu(tree,item).exec(tree.viewport().mapToGlobal(anchor))

    def row_menu(self,tree,item):
        tree.setCurrentItem(item)
        self._selection_changed()
        menu=QMenu(self)
        for button in (self.open_button,self.preview_button,self.report_button,self.folder_button):
            action=menu.addAction(button.icon(),button.text())
            action.setEnabled(button.isEnabled());action.triggered.connect(button.click)
        details=menu.addAction(line_icon('info'),'来源详情')
        details.triggered.connect(self.show_row_details)
        if tree is self.session_tree:
            menu.addSeparator()
            for button in ([self.hide_button,self.purge_button] if self.removed.isChecked() else [self.hide_button]):
                action=menu.addAction(button.icon(),button.text())
                action.setEnabled(button.isEnabled());action.triggered.connect(button.click)
        return menu

    def show_row_details(self):
        from femto.app.qt_common import show_text
        self.source_toggle.setChecked(True);self._selection_changed()
        show_text(self,'来源详情',self.details.text())
        self.source_toggle.setChecked(False);self._selection_changed()

    def purge_selected(self):
        row=self.current()
        if not row or self.busy or self.opening or self.tabs.currentIndex()!=0 or not self.removed.isChecked():return
        owner=self.parentWidget()
        target=Path(row['key']).resolve()
        if owner is not None and hasattr(owner,'capture'):
            pages=(owner.planning,owner.review)
            blocked=(not owner.capture.can_leave() or owner.capture.session.finishing
                     or owner.capture.render_task.busy or any(page.busy or page.quality.runner.busy for page in pages)
                     or owner.execution.busy or not owner.execution.can_leave())
            sources=[owner.capture.session.result,owner.capture.session.source]
            for page in pages:
                sources.extend([page.source,getattr(page.model,'capture_source',None)])
            if blocked or any(source and Path(source).resolve().is_relative_to(target) for source in sources):
                self.status.setText('此记录正在使用或任务尚未结束，请先切换记录并结束后台任务。')
                return
        answer=QMessageBox.question(self,'永久删除',
            '是否永久删除这条记录及其文件？此操作不可恢复。\n\n'+row['name'],
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes:return
        from femto.pipeline.history_purge import execute_purge
        lease=getattr(owner,'data_lease',None)
        if lease is not None:
            def verify_lease():
                self.runner.finished.disconnect(verify_lease)
                if lease.stream is None:
                    owner.setEnabled(False)
                    QMessageBox.critical(self,'数据维护中','工作台数据锁未能恢复，请结束其他维护后重启工作台。')
            self.runner.finished.connect(verify_lease)
        def done(_):
            self.deleted_stack=[rows for rows in self.deleted_stack if all(item['key']!=row['key'] for item in rows)]
            self.refresh()
        def work(control):
            return execute_purge(self.root,target,check=control.check,workstation_lease=lease)
        self._start('正在永久删除',work,done,mutating=True)

    def undo_hide(self):
        if self.busy or self.opening or not self.deleted_stack:
            return
        rows = self.deleted_stack[-1]
        def done(_):
            self.deleted_stack.pop()
            self.refresh()
        self._start('恢复记录', lambda control: set_records_deleted(
            rows, self.datasets, self.outputs, False, check=control.check), done, mutating=True)

    def set_opening(self, value, message=''):
        """Track the owning workspace's read separately from catalogue reads."""
        self.opening = value
        self._open_cancelling = False
        if message:
            self.status.setText(message)
        self._selection_changed()
        if not value and self._closing_requested:
            self.close()

    def cancel_read(self):
        if self._mutating:
            return
        if self._open_cancelling or self.runner.cancelling:
            return
        if self.opening:
            self._open_cancelling = True
            self.cancel_open_requested.emit()
            self.status.setText('正在取消所选记录的读取；原记录保留…')
            self._selection_changed()
        elif self.busy:
            self.runner.cancel()
            self.status.setText("正在取消读取，请等待当前处理结束；当前列表保留。")
            self._selection_changed()

    def closeEvent(self, event):
        if self.runner.busy or self.opening:
            self._closing_requested = True
            self.cancel_read()
            event.ignore()
            return
        self._closed = True
        self.runner.shutdown()
        super().closeEvent(event)

    def reject(self):
        if self.runner.busy or self.opening:
            self._closing_requested = True
            self.cancel_read()
            return
        self._closed = True
        self.runner.shutdown()
        super().reject()


class AdvancedToolsDialog(QDialog):
    """Read-only diagnostics rendered with the same Qt workbench style."""

    def __init__(self, parent=None, root: str | Path | None = None):
        super().__init__(parent)
        self.root = Path(root or REPO_ROOT).resolve()
        self.setWindowTitle("高级工具 · 运行与数据检查")
        self.resize(840, 610)
        layout = QVBoxLayout(self)
        title = QLabel("运行环境与设备接口", self)
        title.setProperty("role", "panelTitle")
        layout.addWidget(title)
        self.report = QTextBrowser(self)
        self.report.setOpenExternalLinks(False)
        layout.addWidget(self.report, 1)
        actions = QHBoxLayout()
        refresh = QPushButton("刷新环境", self)
        refresh.clicked.connect(self.refresh)
        actions.addWidget(refresh)
        maintenance=QPushButton('数据备份与恢复',self)
        maintenance.setToolTip('打开独立维护窗口；开始备份或恢复前需关闭工作台和数据处理工具。')
        maintenance.clicked.connect(self.open_maintenance);actions.addWidget(maintenance)
        for label, folder in (("采集数据", "datasets"), ("重建结果", "outputs"), ("日志", "Log")):
            button = QPushButton(label, self)
            button.clicked.connect(lambda _=False, name=folder: self.open_folder(name))
            actions.addWidget(button)
        close = QPushButton("关闭", self)
        close.clicked.connect(self.close)
        actions.addWidget(close)
        layout.addLayout(actions)
        self.refresh()

    def refresh(self):
        lines = ["界面：PySide6 / Qt", "Python：" + sys.version.split()[0],
                 "解释器：" + sys.executable, "项目：" + str(self.root), "", "已安装组件"]
        for name in ("PySide6", "numpy", "open3d", "opencv-python", "pyorbbecsdk"):
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                version = "未找到分发信息（本地 SDK 可能通过 DLL 或源码加载）" if name == "pyorbbecsdk" else "未安装"
            lines.append(f"{name}：{version}")
        lines.extend(["", "设备状态请以采集页的连接与参考位置检查为准。",
            "本页仅读取环境信息，不打开相机、不发送机械臂或超声设备指令。",
            "规划记录中的点法线、深度与覆盖估算尚不构成设备执行指令。"])
        self.report.setPlainText("\n".join(lines))

    def open_folder(self, folder):
        path = self.root / folder
        if not path.is_dir():
            self.report.append("\n目录尚未创建：" + str(path))
            return
        try:
            open_path(path)
        except (OSError, ValueError) as error:
            self.report.append("\n无法打开目录：" + str(error))

    def open_maintenance(self):
        try:
            launch_data_maintenance(self.root)
        except OSError as error:
            self.report.append('\n无法打开数据维护：'+str(error))
