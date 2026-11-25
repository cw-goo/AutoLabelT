"""Curated video model picker for Grounding DINO and SAM."""

from html import escape

from PyQt6 import QtCore, QtGui, QtWidgets

from autolabel.workbench.views.labeling.utils.qt import new_icon

from .models.registry import get_model
from .models.downloader import model_download_state


SQUARE_MODEL_IDS = ("grounding_dino", "grounding_dino_base", "sam3", "sam3_1")


class ModelTree(QtWidgets.QTreeWidget):
    def drawBranches(self, painter, rect, index):
        # Keep provider expand arrows, without native hover/selection painting
        # in the indentation area before a model row.
        if index.parent().isValid() or not self.model().hasChildren(index):
            return
        x = rect.right() - self.indentation() / 2 + 1
        y = rect.center().y()
        if self.isExpanded(index):
            points = [(x - 3, y - 1), (x, y + 2), (x + 3, y - 1)]
        else:
            points = [(x - 1, y - 3), (x + 2, y), (x - 1, y + 3)]
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.ColorRole.Text), 1.2))
        painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(*point) for point in points]))
        painter.restore()


class ModelSourceLink(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.url = None
        self.setOpenExternalLinks(True)
        self.setWordWrap(True)

    def set_source(self, url):
        self.url = url
        self._render()

    def _render(self):
        if self.url:
            dark = self.window().palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
            color = "#58c5c1" if dark else "#087f88"
            self.setText(f'<a href="{escape(self.url, quote=True)}" style="color:{color};">查看模型来源与许可证 ↗</a>')
        else:
            self.setText("本地权重的来源及许可证由用户自行确认")

    def clear(self):
        self.url = None
        super().clear()

    def changeEvent(self, event):
        super().changeEvent(event)
        if getattr(self, "url", None) and event.type() in (
                QtCore.QEvent.Type.StyleChange, QtCore.QEvent.Type.PaletteChange):
            self._render()


class ModelDetailsPanel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modelInfoPanel")
        body = QtWidgets.QVBoxLayout(self)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(14)
        self.title = QtWidgets.QLabel("选择一个模型")
        self.title.setObjectName("modelInfoTitle")
        self.title.setWordWrap(True)
        body.addWidget(self.title)
        badges = QtWidgets.QHBoxLayout()
        self.task = QtWidgets.QLabel("视频检测")
        self.task.setObjectName("modelTaskBadge")
        self.state = QtWidgets.QLabel("待选择")
        self.state.setObjectName("modelStateBadge")
        badges.addWidget(self.task)
        badges.addWidget(self.state)
        badges.addStretch()
        body.addLayout(badges)
        self.description = QtWidgets.QLabel("从左侧浏览模型，查看适用任务与使用要求。")
        self.description.setWordWrap(True)
        body.addWidget(self.description)
        divider = QtWidgets.QFrame()
        divider.setObjectName("modelInfoDivider")
        divider.setFixedHeight(1)
        body.addWidget(divider)
        facts = QtWidgets.QFormLayout()
        facts.setSpacing(12)
        facts.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.facts = {}
        for key in ("模型来源", "许可证", "版本", "检测类别"):
            label = QtWidgets.QLabel("—")
            label.setWordWrap(True)
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.facts[key] = label
            facts.addRow(key, label)
        body.addLayout(facts)
        body.addStretch()
        self.guidance = QtWidgets.QLabel("选择模型只更新设置，下载和加载测试由你主动开始。")
        self.guidance.setObjectName("modelUsageHint")
        self.guidance.setWordWrap(True)
        body.addWidget(self.guidance)
        self.source_link = ModelSourceLink()
        body.addWidget(self.source_link)

    def show_model(self, model_id):
        spec = get_model(model_id)
        self.title.setText(spec.title)
        self.task.setText(spec.task)
        self.state.setText({"missing": "未下载", "downloaded": "权重已下载",
                            "ready": "已通过加载测试"}[model_download_state(model_id)])
        self.description.setText(spec.description)
        categories = ("COCO 已有类别" if spec.backend == "ultralytics_coco" else
                      "person（类别 ID 0）" if model_id == "local_yolo" else "自定义文本类别")
        for key, value in (("模型来源", spec.source), ("许可证", spec.license),
                           ("版本", spec.revision), ("检测类别", categories)):
            self.facts[key].setText(value if len(value) <= 24 else value[:16] + "…")
            self.facts[key].setToolTip(value)
        self.source_link.set_source(spec.source_url)
        self.guidance.setText(
            "需要 Hugging Face 权重授权；分割结果会转换为检测框，生成后请人工复核。"
            if model_id in ("sam3", "sam3_1") else
            "首次加载需要确认模型代码来源。在模型加载窗口勾选信任后，再进行测试。"
            if model_id == "florence2" else
            "设置已有类别后下载并测试模型；COCO 预训练权重不支持新增文本类别。"
            if spec.backend == "ultralytics_coco" else
            "可输入 person、phone 等类别，用英文逗号分隔。选择后下载并测试模型。")

    def show_unavailable(self, title="选择一个模型", description=None):
        self.title.setText(title)
        self.task.setText("模型目录")
        self.state.setText("仅图片标注" if description else "待选择")
        self.description.setText(description or "从左侧浏览模型，查看适用任务与使用要求。")
        for label in self.facts.values():
            label.setText("—")
            label.setToolTip("")
        self.source_link.clear()
        self.guidance.setText("图片模型请从左侧图片标注入口加载。" if description else
                              "选择模型只更新设置，下载和加载测试由你主动开始。")


class ModelSquareDialog(QtWidgets.QDialog):
    def __init__(self, native_models=None, parent=None):
        super().__init__(parent)
        self.native_models = native_models
        self.selected_model_id = None
        self.setWindowTitle("模型广场 · 视频数据集")
        self.setObjectName("modelSquareDialog")
        self.resize(980, 660)
        self.setMinimumSize(860, 580)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(14)
        heading = QtWidgets.QLabel("模型广场")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        hint = QtWidgets.QLabel("Grounding DINO · SAM 系列")
        hint.setWordWrap(True)
        hint.setProperty("tone", "muted")
        layout.addWidget(hint)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("搜索厂商或模型")
        self.search.textChanged.connect(self.filter_models)
        filters = QtWidgets.QHBoxLayout()
        filters.addWidget(self.search, 1)
        layout.addLayout(filters)
        panes = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        panes.setChildrenCollapsible(False)
        self.tree = ModelTree()
        self.tree.setObjectName("modelSquareTree")
        self.tree.setHeaderLabels(["厂商 / 模型", "视频数据集"])
        self.tree.setMinimumWidth(320)
        self.tree.setIndentation(18)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept_selection())
        panes.addWidget(self.tree)
        self.info = ModelDetailsPanel()
        self.info.setMinimumWidth(350)
        self.details = self.info.description
        panes.addWidget(self.info)
        panes.setSizes([410, 490])
        layout.addWidget(panes, 1)
        self.result_count = QtWidgets.QLabel()
        self.result_count.setProperty("tone", "muted")
        layout.addWidget(self.result_count)
        actions = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        actions.rejected.connect(self.reject)
        self.choose_button = actions.addButton("选择模型", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.choose_button.setProperty("role", "primary")
        self.choose_button.setEnabled(False)
        self.info.show_unavailable()
        self.choose_button.clicked.connect(self.accept_selection)
        layout.addWidget(actions)
        self.refresh()

    def refresh(self):
        self.tree.clear()
        self.selected_model_id = None
        self.choose_button.setEnabled(False)
        groups = {}

        def add(provider, name, title, model_id):
            if provider not in groups:
                groups[provider] = QtWidgets.QTreeWidgetItem([provider])
                groups[provider].setIcon(0, QtGui.QIcon(new_icon(provider.lower(), "png")))
                self.tree.addTopLevelItem(groups[provider])
            item = QtWidgets.QTreeWidgetItem([title, "可用" if model_id else "仅图片标注"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, model_id)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, name)
            groups[provider].addChild(item)

        for model_id in SQUARE_MODEL_IDS:
            spec = get_model(model_id)
            add(spec.source, model_id, spec.title, model_id)
        for group in groups.values():
            group.setExpanded(True)
        self.filter_models(self.search.text())

    def filter_models(self, query):
        query = query.strip().casefold()
        count = 0
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            visible = False
            for row in range(group.childCount()):
                item = group.child(row)
                match = query in group.text(0).casefold() or query in item.text(0).casefold()
                item.setHidden(not match)
                count += int(match)
                visible |= match
            group.setHidden(not visible)
        self.result_count.setText(f"{count} 个模型")
        current = self.tree.currentItem()
        if current and (current.isHidden() or (current.parent() and current.parent().isHidden())):
            self.tree.setCurrentItem(None)

    def _selection_changed(self):
        item = self.tree.currentItem()
        model_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.choose_button.setEnabled(bool(model_id))
        if model_id:
            self.info.show_model(model_id)
        elif item and item.parent():
            self.info.show_unavailable(item.text(0), "此模型可在图片自动标注中使用，尚未接入视频数据集检测。")
        else:
            self.info.show_unavailable()

    def accept_selection(self):
        item = self.tree.currentItem()
        model_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        if model_id in SQUARE_MODEL_IDS:
            self.selected_model_id = model_id
            self.accept()

