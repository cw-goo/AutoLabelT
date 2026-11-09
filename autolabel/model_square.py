"""Provider/model picker backed by the image workbench's model catalogue."""

import re

from PyQt6 import QtCore, QtGui, QtWidgets

from autolabel.workbench.views.labeling.utils.qt import new_icon

from .models.registry import MODEL_REGISTRY, get_model


def video_model_id(name):
    name = name.split("-r", 1)[0]
    match = re.fullmatch(r"(yolov8|yolo11|yolo26)([nsmlx])", name)
    if match:
        return f"{match[1]}_{match[2]}"
    return {"yolov8s_worldv2": "yolo_world", "sam3_vit_h": "sam3"}.get(name)


class ModelSquareDialog(QtWidgets.QDialog):
    def __init__(self, native_models=None, parent=None):
        super().__init__(parent)
        self.native_models = native_models
        self.selected_model_id = None
        self.setWindowTitle("模型广场 · 视频数据集")
        self.setObjectName("modelSquareDialog")
        self.resize(740, 620)
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel("模型广场")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        hint = QtWidgets.QLabel("按厂商选择模型。模型目录与图片自动标注同步；支持视频数据集的模型可直接选择。")
        hint.setWordWrap(True)
        hint.setProperty("tone", "muted")
        layout.addWidget(hint)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("搜索厂商或模型")
        self.search.textChanged.connect(self.filter_models)
        layout.addWidget(self.search)
        self.only_video = QtWidgets.QCheckBox("仅显示可用于视频数据集的模型")
        self.only_video.setChecked(True)
        layout.addWidget(self.only_video)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setObjectName("modelSquareTree")
        self.tree.setHeaderLabels(["厂商 / 模型", "视频数据集"])
        self.tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tree.itemSelectionChanged.connect(self._selection_changed)
        self.tree.itemDoubleClicked.connect(lambda *_: self.accept_selection())
        self.only_video.toggled.connect(lambda _: self.filter_models(self.search.text()))
        layout.addWidget(self.tree, 1)
        self.details = QtWidgets.QLabel("选择一个可用于视频数据集的模型。")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        actions = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        actions.rejected.connect(self.reject)
        self.choose_button = actions.addButton("选择模型", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.choose_button.setProperty("role", "primary")
        self.choose_button.setEnabled(False)
        self.choose_button.clicked.connect(self.accept_selection)
        layout.addWidget(actions)
        self.refresh()

    def refresh(self):
        self.tree.clear()
        self.selected_model_id = None
        self.choose_button.setEnabled(False)
        groups, represented = {}, set()

        def add(provider, name, title, model_id):
            if provider not in groups:
                groups[provider] = QtWidgets.QTreeWidgetItem([provider])
                groups[provider].setIcon(0, QtGui.QIcon(new_icon(provider.lower(), "png")))
                self.tree.addTopLevelItem(groups[provider])
            item = QtWidgets.QTreeWidgetItem([title, "可用" if model_id else "仅图片标注"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, model_id)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, name)
            if not model_id:
                item.setToolTip(0, "此模型尚未接入视频抽帧检测，可在图片自动标注中使用。")
            else:
                represented.add(model_id)
            groups[provider].addChild(item)

        if self.native_models is not None:
            for provider, models in self.native_models.init_model_data().items():
                for name, details in models.items():
                    if name in ("load_custom_model", "import_local_weight"):
                        continue
                    add(provider, name, details["display_name"], video_model_id(name))
        # Some video backends (e.g. Transformers) have their own supported variants.
        for model_id, spec in MODEL_REGISTRY.items():
            if model_id == "local_yolo" or model_id in represented:
                continue
            provider = spec.source.split(" / ", 1)[0]
            add(provider, model_id, spec.title, model_id)
        for group in groups.values():
            group.setExpanded(True)
        self.filter_models(self.search.text())

    def filter_models(self, query):
        query = query.strip().casefold()
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            visible = False
            for row in range(group.childCount()):
                item = group.child(row)
                match = query in group.text(0).casefold() or query in item.text(0).casefold()
                if self.only_video.isChecked():
                    match &= bool(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
                item.setHidden(not match)
                visible |= match
            group.setHidden(not visible)

    def _selection_changed(self):
        item = self.tree.currentItem()
        model_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        self.choose_button.setEnabled(bool(model_id))
        if model_id:
            spec = get_model(model_id)
            self.details.setText(f"{spec.title} · {spec.task}\n{spec.description}")
        elif item and item.parent():
            self.details.setText("此模型可在图片自动标注中使用，尚未接入视频数据集检测。")

    def accept_selection(self):
        item = self.tree.currentItem()
        model_id = item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None
        if model_id:
            self.selected_model_id = model_id
            self.accept()

