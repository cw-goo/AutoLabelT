"""Video dataset controls docked beside AutoLabel's native canvas."""

from dataclasses import replace
from pathlib import Path
import time

from PyQt6 import QtCore, QtWidgets

from autolabel.workbench.views.labeling.annotation_io import describe_file_error

from .core import ProjectConfig, run_pipeline, video_info
from .models.downloader import model_download_state, model_weight_path, release_session_model
from .models.registry import MODEL_REGISTRY, get_model
from .model_square import ModelSquareDialog
from .models.worker import ModelWorker
from .models.adapters import parse_prompts
from .annotation_bridge import prepare_review_dataset, prepare_review_image, sync_review_to_yolo


PARAMETER_LABELS = {
    "confidence": "置信度",
    "image_size": "推理尺寸",
    "device": "运行设备",
    "min_pixel_height": "主目标最小高度",
    "idle_jump": "空镜跳帧",
    "step_single": "单目标跳帧",
    "step_small_group": "2–3 目标跳帧",
    "step_large_group": "4+ 目标跳帧",
    "companion_min_height": "伴随标注最小高度",
    "small_target_ratio": "小目标占比",
    "reject_single_below": "单目标弱特征高度",
}

PARAMETER_GROUPS = (
    ("检测设置", ("confidence", "image_size", "device")),
    ("画面筛选", ("min_pixel_height", "companion_min_height",
               "small_target_ratio", "reject_single_below")),
    ("动态抽帧", ("idle_jump", "step_single", "step_small_group",
               "step_large_group")),
)


class ParameterDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("处理参数 · AutoLabel")
        self.setObjectName("parameterDialog")
        self.resize(620, 660)
        self.controls = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)
        heading = QtWidgets.QLabel("处理参数")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        hint = QtWidgets.QLabel("按已验证的运行基线调整，保存后应用于下次构建。")
        hint.setObjectName("dialogHint")
        layout.addWidget(hint)
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("parameterScroll")
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        body.setObjectName("parameterBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(2, 4, 2, 4)
        body_layout.setSpacing(8)
        for group_title, keys in PARAMETER_GROUPS:
            section = QtWidgets.QLabel(group_title)
            section.setObjectName("parameterSectionTitle")
            body_layout.addWidget(section)
            grid = QtWidgets.QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(10)
            for index, key in enumerate(keys):
                value = getattr(config, key)
                if key == "device":
                    control = QtWidgets.QComboBox()
                    for label, data in (("自动选择", "auto"), ("CPU", "cpu"),
                                        ("GPU 0", "0"), ("GPU 1", "1")):
                        control.addItem(label, data)
                    if control.findData(value) < 0:
                        control.addItem(str(value), str(value))
                    control.setCurrentIndex(control.findData(value))
                elif isinstance(value, float):
                    control = QtWidgets.QDoubleSpinBox()
                    control.setRange(0, 1)
                    control.setDecimals(2)
                    control.setSingleStep(0.05)
                    control.setValue(value)
                else:
                    control = QtWidgets.QSpinBox()
                    control.setRange(1, 100000)
                    control.setValue(value)
                    control.setSuffix(" px" if "height" in key else
                                      " 帧" if "jump" in key or "step" in key else "")
                self.controls[key] = control
                field = QtWidgets.QWidget()
                field_layout = QtWidgets.QVBoxLayout(field)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.setSpacing(5)
                label = QtWidgets.QLabel(PARAMETER_LABELS[key])
                label.setObjectName("parameterFieldLabel")
                field_layout.addWidget(label)
                if isinstance(control, QtWidgets.QComboBox):
                    field_layout.addWidget(control)
                else:
                    control.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
                    stepper = QtWidgets.QFrame()
                    stepper.setObjectName("parameterStepper")
                    stepper_layout = QtWidgets.QHBoxLayout(stepper)
                    stepper_layout.setContentsMargins(4, 3, 3, 3)
                    stepper_layout.setSpacing(3)
                    stepper_layout.addWidget(control, 1)
                    decrease = QtWidgets.QPushButton("−")
                    decrease.setObjectName("parameterStepButton")
                    decrease.setFixedSize(30, 30)
                    decrease.setAccessibleName(f"调小{PARAMETER_LABELS[key]}")
                    decrease.clicked.connect(control.stepDown)
                    increase = QtWidgets.QPushButton("+")
                    increase.setObjectName("parameterStepButton")
                    increase.setFixedSize(30, 30)
                    increase.setAccessibleName(f"调大{PARAMETER_LABELS[key]}")
                    increase.clicked.connect(control.stepUp)
                    stepper_layout.addWidget(decrease)
                    stepper_layout.addWidget(increase)
                    field_layout.addWidget(stepper)
                grid.addWidget(field, index // 2, index % 2)
            body_layout.addLayout(grid)
            body_layout.addSpacing(5)
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll)
        footer = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("恢复默认")
        reset.clicked.connect(self.restore_defaults)
        footer.addWidget(reset)
        footer.addStretch()
        cancel = QtWidgets.QPushButton("取消")
        cancel.clicked.connect(self.reject)
        apply = QtWidgets.QPushButton("应用参数")
        apply.setProperty("role", "primary")
        apply.clicked.connect(self.accept)
        footer.addWidget(cancel)
        footer.addWidget(apply)
        layout.addLayout(footer)

    def values(self):
        return {key: control.currentData() if isinstance(control, QtWidgets.QComboBox)
                else control.value() for key, control in self.controls.items()}

    def restore_defaults(self):
        defaults = ProjectConfig()
        for key, control in self.controls.items():
            value = getattr(defaults, key)
            if isinstance(control, QtWidgets.QComboBox):
                control.setCurrentIndex(control.findData(value))
            else:
                control.setValue(value)


class ModelLoadDialog(QtWidgets.QDialog):
    def __init__(self, dock):
        super().__init__(dock)
        self.dock = dock
        self.setWindowTitle("模型加载 · AutoLabel")
        self.setObjectName("modelLoadDialog")
        self.resize(900, 580)
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(24, 22, 24, 20)
        root_layout.setSpacing(14)
        heading = QtWidgets.QLabel("模型中心")
        heading.setObjectName("dialogHeading")
        root_layout.addWidget(heading)
        hint = QtWidgets.QLabel("为图片标注或视频构建选择模型。两种任务分别保存自己的模型选择。")
        hint.setObjectName("dialogHint")
        root_layout.addWidget(hint)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("modelTaskTabs")
        root_layout.addWidget(self.tabs, 1)
        self.image_page = QtWidgets.QWidget()
        self.tabs.addTab(self.image_page, "图片标注")
        self._build_image_page()
        video_page = QtWidgets.QWidget()
        self.tabs.addTab(video_page, "视频数据集")
        layout = QtWidgets.QVBoxLayout(video_page)
        layout.setContentsMargins(4, 14, 4, 4)
        layout.setSpacing(12)
        video_hint = QtWidgets.QLabel("视频模型用于自动抽帧和生成 YOLO 标签；云端权重仅在本次运行期间保留。")
        video_hint.setProperty("tone", "muted")
        layout.addWidget(video_hint)
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(12)
        local_card = QtWidgets.QFrame()
        local_card.setProperty("modelCard", True)
        local_layout = QtWidgets.QVBoxLayout(local_card)
        local_layout.setContentsMargins(17, 17, 17, 17)
        local_title = QtWidgets.QLabel("加载本地权重")
        local_title.setObjectName("modelColumnTitle")
        local_layout.addWidget(local_title)
        local_hint = QtWidgets.QLabel("选择本地 YOLO .pt 权重，用于视频检测与抽帧。")
        local_hint.setWordWrap(True)
        local_layout.addWidget(local_hint)
        local_layout.addStretch()
        dock.weight_button.setMinimumHeight(42)
        local_layout.addWidget(dock.weight_button)
        local_layout.addWidget(dock.weight_label)
        columns.addWidget(local_card, 1)

        cloud_card = QtWidgets.QFrame()
        cloud_card.setProperty("modelCard", True)
        cloud_layout = QtWidgets.QVBoxLayout(cloud_card)
        cloud_layout.setContentsMargins(17, 17, 17, 17)
        cloud_title = QtWidgets.QLabel("模型广场")
        cloud_title.setObjectName("modelColumnTitle")
        cloud_layout.addWidget(cloud_title)
        cloud_hint = QtWidgets.QLabel("与图片自动标注共用厂商和模型目录，选择后下载并测试加载。")
        cloud_hint.setWordWrap(True)
        cloud_layout.addWidget(cloud_hint)
        cloud_layout.addStretch()
        self.square_button = QtWidgets.QPushButton("打开模型广场")
        self.square_button.setProperty("role", "primary")
        self.square_button.setMinimumHeight(42)
        self.square_button.clicked.connect(self.open_model_square)
        cloud_layout.addWidget(self.square_button)
        self.square_selection = QtWidgets.QLabel()
        self.square_selection.setWordWrap(True)
        self.square_selection.setProperty("tone", "muted")
        cloud_layout.addWidget(self.square_selection)
        self.model_square = ModelSquareDialog(self.native_models, self)
        columns.addWidget(cloud_card, 1)
        layout.addLayout(columns, 1)
        self.details = QtWidgets.QLabel()
        self.details.setObjectName("modelDetails")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.source_link = QtWidgets.QLabel()
        self.source_link.setOpenExternalLinks(True)
        layout.addWidget(self.source_link)
        prompt_row = QtWidgets.QHBoxLayout()
        prompt_label = QtWidgets.QLabel("检测类别  *")
        prompt_row.addWidget(prompt_label)
        prompt_row.addWidget(dock.prompts, 1)
        layout.addLayout(prompt_row)
        self.prompt_hint = QtWidgets.QLabel()
        self.prompt_hint.setProperty("tone", "muted")
        layout.addWidget(self.prompt_hint)
        layout.addWidget(dock.trust_code)
        actions = QtWidgets.QHBoxLayout()
        for button in (dock.download_button, dock.test_button, dock.runtime_button,
                       dock.model_cancel_button):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        layout.addWidget(dock.model_progress)
        layout.addWidget(dock.model_status)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        confirm = QtWidgets.QPushButton("使用此模型")
        confirm.setProperty("role", "primary")
        confirm.clicked.connect(self.confirm_selection)
        footer.addWidget(confirm)
        layout.addLayout(footer)
        self.select_model(dock.config.model_id)

    def _build_image_page(self):
        layout = QtWidgets.QVBoxLayout(self.image_page)
        layout.setContentsMargins(4, 14, 4, 4)
        layout.setSpacing(10)
        hint = QtWidgets.QLabel("使用 AutoLabel 原生模型标注图片；可选预训练模型、本地 YOLO 权重或自定义 YAML 配置。")
        hint.setWordWrap(True)
        hint.setProperty("tone", "muted")
        layout.addWidget(hint)
        self.image_search = QtWidgets.QLineEdit()
        self.image_search.setPlaceholderText("搜索模型名称或来源")
        self.image_search.textChanged.connect(self._filter_image_models)
        layout.addWidget(self.image_search)
        self.image_models = QtWidgets.QTreeWidget()
        self.image_models.setObjectName("imageModelTree")
        self.image_models.setHeaderHidden(True)
        self.image_models.itemDoubleClicked.connect(lambda *_: self.load_image_model())
        layout.addWidget(self.image_models, 1)
        self.image_status = QtWidgets.QLabel("选择模型后点击加载。")
        self.image_status.setWordWrap(True)
        self.image_status.setProperty("tone", "muted")
        layout.addWidget(self.image_status)
        actions = QtWidgets.QHBoxLayout()
        local_weight = QtWidgets.QPushButton("导入本地 YOLO 权重")
        local_weight.clicked.connect(self.import_image_weight)
        actions.addWidget(local_weight)
        custom = QtWidgets.QPushButton("导入自定义 YAML")
        custom.clicked.connect(self.import_image_model)
        actions.addWidget(custom)
        actions.addStretch()
        load = QtWidgets.QPushButton("加载到图片标注")
        load.setProperty("role", "primary")
        load.clicked.connect(self.load_image_model)
        actions.addWidget(load)
        layout.addLayout(actions)
        view = getattr(getattr(self.dock.window, "labeling_widget", None), "view", None)
        self.native_models = getattr(view, "auto_labeling_widget", None)
        if self.native_models is None:
            self.image_search.setEnabled(False)
            load.setEnabled(False)
            local_weight.setEnabled(False)
            custom.setEnabled(False)
            self.image_status.setText("图片模型仅在集成工作台中可用。")
            return
        self.native_models.model_manager.new_model_status.connect(self.image_status.setText)
        self.refresh_image_models()

    def refresh_image_models(self):
        if self.native_models is None:
            return
        self.image_models.clear()
        for provider, models in self.native_models.init_model_data().items():
            group = QtWidgets.QTreeWidgetItem([provider])
            self.image_models.addTopLevelItem(group)
            for name, details in models.items():
                if name in ("load_custom_model", "import_local_weight"):
                    continue
                item = QtWidgets.QTreeWidgetItem([details["display_name"]])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (provider, name))
                group.addChild(item)
            group.setExpanded(provider == "Custom")
        self._filter_image_models(self.image_search.text())

    def _filter_image_models(self, query):
        query = query.strip().casefold()
        for index in range(self.image_models.topLevelItemCount()):
            group = self.image_models.topLevelItem(index)
            matches = 0
            for child_index in range(group.childCount()):
                child = group.child(child_index)
                visible = query in group.text(0).casefold() or query in child.text(0).casefold()
                child.setHidden(not visible)
                matches += visible
            group.setHidden(matches == 0)
            if query and matches:
                group.setExpanded(True)

    def load_image_model(self):
        if self.native_models is None:
            return
        item = self.image_models.currentItem()
        if item is None or item.data(0, QtCore.Qt.ItemDataRole.UserRole) is None:
            self.image_status.setText("请先选择一个图片模型。")
            return
        provider, name = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if self.native_models.isHidden():
            self.dock.window.labeling_widget.view.toggle_auto_labeling_widget()
        self.image_status.setText(f"正在加载 {item.text(0)}…")
        self.native_models.on_model_selected(provider, name)

    def import_image_model(self):
        if self.native_models is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 AutoLabel 模型配置", "", "YAML 配置 (*.yaml *.yml)")
        if path:
            self.image_status.setText(f"正在加载 {Path(path).name}…")
        if path and self.native_models.load_custom_model_config(path):
            self.refresh_image_models()

    def import_image_weight(self):
        if self.native_models is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "导入本地 YOLO 权重", "", "YOLO 权重 (*.pt)")
        if not path:
            return
        try:
            self.image_status.setText(f"正在加载 {Path(path).name}…")
            if self.native_models.prompt_and_import_local_weight(path, self):
                self.refresh_image_models()
        except ValueError as error:
            self.image_status.setText(str(error))

    def confirm_selection(self):
        try:
            current = self.dock._current_config()
        except ValueError as error:
            self.dock._error(str(error))
            self.dock.prompts.setFocus()
            return
        self.dock.model_summary.setText(
            f"{get_model(current.model_id).title} · {current.prompt_labels}")
        self.hide()

    def open_model_square(self):
        if not self.dock.model_combo.isEnabled():
            return
        self.model_square.refresh()
        if self.model_square.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            model_id = self.model_square.selected_model_id
            if model_id:
                self.dock.model_combo.setCurrentIndex(self.dock.model_combo.findData(model_id))

    def select_model(self, model_id):
        self.dock.weight_button.setProperty("selected", model_id == "local_yolo")
        self.dock.weight_button.style().unpolish(self.dock.weight_button)
        self.dock.weight_button.style().polish(self.dock.weight_button)
        if model_id != "local_yolo":
            self.square_selection.setText(get_model(model_id).title)
        else:
            self.square_selection.setText("按厂商浏览可用模型")
        spec = get_model(model_id)
        self.details.setText(f"{spec.description}  ·  {spec.source}  ·  {spec.license}  ·  {spec.revision}")
        self.source_link.setText(
            f'<a href="{spec.source_url}">查看模型来源与许可证 ↗</a>' if spec.source_url else
            "本地权重的来源及许可证由用户自行确认")
        self.prompt_hint.setText(
            "当前本地 YOLO 基线只检测 person（类别 ID 0）。" if model_id == "local_yolo"
            else "COCO 预训练仅支持已有类别；开放词汇模型可输入新类别。"
            if model_id not in ("sam3", "sam3_1") else
            "SAM 3 使用文字概念生成分割结果，再转换为 YOLO 框；需授权和人工复核。")


class VideoPipelineWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(dict)
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def report(event):
                if event.get("image"):
                    prepare_review_image(self.config.output_dir, event["image"])
                self.progress.emit(event)

            result = run_pipeline(self.config, report,
                                  lambda: self._cancel)
            # Each saved sample was already converted and checked by report().
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(describe_file_error(error))


class SyncWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = root

    def run(self):
        try:
            self.completed.emit(sync_review_to_yolo(self.root, self.isInterruptionRequested))
        except Exception as error:
            self.failed.emit(describe_file_error(error))


class ReviewPrepareWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = root

    def run(self):
        try:
            self.completed.emit(prepare_review_dataset(self.root, self.isInterruptionRequested))
        except Exception as error:
            self.failed.emit(describe_file_error(error))


class VideoDatasetDock(QtWidgets.QDockWidget):
    def __init__(self, window):
        super().__init__("视频数据集", window)
        self.setObjectName("videoDatasetDock")
        title_bar = QtWidgets.QWidget(self)
        title_bar.setFixedHeight(0)
        self.setTitleBarWidget(title_bar)
        self.setMinimumWidth(340)
        self.window = window
        self.config = ProjectConfig()
        self.local_model_path = self.config.model_path
        self.pipeline_worker = None
        self.model_worker = None
        self.model_started = 0.0
        self.model_phase = ""
        self.model_timer = QtCore.QTimer(self)
        self.model_timer.setInterval(500)
        self.model_timer.timeout.connect(self._model_tick)
        self.sync_worker = None
        self.prepare_worker = None
        self.live_review_dir = None
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("videoDockScroll")
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        content.setObjectName("videoDockContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(14)

        header = QtWidgets.QFrame()
        header.setObjectName("workflowHeader")
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(3)
        title = QtWidgets.QLabel("视频数据集")
        title.setObjectName("workflowTitle")
        subtitle = QtWidgets.QLabel("从视频到可复核的标注数据")
        subtitle.setObjectName("workflowSubtitle")
        steps = QtWidgets.QLabel("视频  →  检测  →  抽帧  →  复核")
        steps.setObjectName("workflowSteps")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addSpacing(5)
        header_layout.addWidget(steps)
        layout.addWidget(header)

        project = QtWidgets.QGroupBox("项目")
        project_form = QtWidgets.QFormLayout(project)
        self.video_button = QtWidgets.QPushButton("选择视频")
        self.video_button.clicked.connect(self.choose_video)
        self.video_label = QtWidgets.QLabel("未选择")
        self.video_label.setProperty("tone", "muted")
        self.video_label.setWordWrap(True)
        project_form.addRow(self.video_button, self.video_label)
        self.output_button = QtWidgets.QPushButton("输出目录")
        self.output_button.clicked.connect(self.choose_output)
        self.output_label = QtWidgets.QLabel("未选择")
        self.output_label.setProperty("tone", "muted")
        self.output_label.setWordWrap(True)
        project_form.addRow(self.output_button, self.output_label)
        layout.addWidget(project)

        model_card = QtWidgets.QGroupBox("检测模型")
        model_layout = QtWidgets.QVBoxLayout(model_card)
        self.model_combo = QtWidgets.QComboBox()
        for model_id, spec in MODEL_REGISTRY.items():
            self.model_combo.addItem(spec.title, model_id)
        self.model_combo.currentIndexChanged.connect(self.model_changed)
        self.model_combo.setParent(self)
        self.model_source = QtWidgets.QLabel()
        self.model_source.setWordWrap(True)
        self.weight_button = QtWidgets.QPushButton("选择本地权重")
        self.weight_button.clicked.connect(self.choose_weight)
        self.weight_label = QtWidgets.QLabel("staffv2.pt")
        self.weight_label.setProperty("tone", "muted")
        self.weight_label.setWordWrap(True)
        self.prompts = QtWidgets.QLineEdit("person")
        self.prompts.setPlaceholderText("类别：person, phone")
        self.trust_code = QtWidgets.QCheckBox("信任 Florence-2 固定版本模型代码")
        self.download_button = QtWidgets.QPushButton("下载")
        self.download_button.setProperty("role", "primary")
        self.download_button.clicked.connect(self.download_model)
        self.test_button = QtWidgets.QPushButton("测试模型")
        self.test_button.clicked.connect(self.test_model)
        self.runtime_button = QtWidgets.QPushButton("安装依赖")
        self.runtime_button.clicked.connect(self.install_runtime)
        self.model_cancel_button = QtWidgets.QPushButton("取消模型任务")
        self.model_cancel_button.setProperty("role", "danger")
        self.model_cancel_button.clicked.connect(self.cancel_model_task)
        self.model_cancel_button.setEnabled(False)
        self.model_progress = QtWidgets.QProgressBar()
        self.model_progress.setVisible(False)
        self.model_status = QtWidgets.QLabel()
        self.model_status.setProperty("tone", "muted")
        self.model_status.setWordWrap(True)
        self.model_dialog = ModelLoadDialog(self)
        self.model_button = QtWidgets.QPushButton("模型加载")
        self.model_button.setProperty("role", "primary")
        self.model_button.clicked.connect(self.open_model_dialog)
        model_layout.addWidget(self.model_button)
        self.model_summary = QtWidgets.QLabel()
        self.model_summary.setProperty("tone", "muted")
        self.model_summary.setWordWrap(True)
        model_layout.addWidget(self.model_summary)
        layout.addWidget(model_card)

        parameter_card = QtWidgets.QGroupBox("处理参数")
        parameter_layout = QtWidgets.QVBoxLayout(parameter_card)
        self.parameter_button = QtWidgets.QPushButton("调整参数")
        self.parameter_button.clicked.connect(self.edit_parameters)
        parameter_layout.addWidget(self.parameter_button)
        self.parameter_summary = QtWidgets.QLabel()
        self.parameter_summary.setProperty("tone", "muted")
        self.parameter_summary.setWordWrap(True)
        parameter_layout.addWidget(self.parameter_summary)
        configuration_actions = QtWidgets.QHBoxLayout()
        save_configuration = QtWidgets.QPushButton("保存配置")
        save_configuration.clicked.connect(self.save_config)
        load_configuration = QtWidgets.QPushButton("加载配置")
        load_configuration.clicked.connect(self.load_config)
        configuration_actions.addWidget(save_configuration)
        configuration_actions.addWidget(load_configuration)
        parameter_layout.addLayout(configuration_actions)
        layout.addWidget(parameter_card)

        run_card = QtWidgets.QGroupBox("构建与复核")
        run_layout = QtWidgets.QVBoxLayout(run_card)
        run_actions = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始构建")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QtWidgets.QPushButton("取消")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.clicked.connect(self.cancel)
        run_actions.addWidget(self.start_button)
        run_actions.addWidget(self.cancel_button)
        run_layout.addLayout(run_actions)
        self.progress_bar = QtWidgets.QProgressBar()
        run_layout.addWidget(self.progress_bar)
        self.stats = QtWidgets.QLabel("等待视频")
        self.stats.setProperty("tone", "muted")
        run_layout.addWidget(self.stats)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setMaximumBlockCount(1000)
        self.log.setReadOnly(True)
        self.log.setFixedHeight(76)
        self.log.setPlaceholderText("运行记录")
        run_layout.addWidget(self.log)
        self.open_button = QtWidgets.QPushButton("在画布中复核")
        self.open_button.clicked.connect(self.open_review)
        self.sync_button = QtWidgets.QPushButton("同步更新标签")
        self.sync_button.clicked.connect(self.sync_review)
        review_actions = QtWidgets.QHBoxLayout()
        review_actions.addWidget(self.open_button)
        review_actions.addWidget(self.sync_button)
        run_layout.addLayout(review_actions)
        layout.addWidget(run_card)
        layout.addStretch(1)
        scroll.setWidget(content)
        self.setWidget(scroll)
        self._project_controls = (self.video_button, self.output_button,
                                  self.parameter_button, load_configuration,
                                  self.prompts, self.trust_code, self.open_button,
                                  self.sync_button, self.start_button)
        self.model_changed()

        self._update_parameter_summary()

    def has_running_workers(self):
        return any(worker and getattr(worker, "isRunning", lambda: False)()
                   for worker in (self.pipeline_worker, self.model_worker,
                                  self.sync_worker, self.prepare_worker))

    def _refresh_task_controls(self):
        enabled = not self.has_running_workers()
        for control in self._project_controls:
            control.setEnabled(enabled)
        self._set_model_selection_enabled(enabled)

    def open_model_dialog(self, task="video"):
        self.model_dialog.tabs.setCurrentWidget(
            self.model_dialog.image_page if task == "image" else
            self.model_dialog.tabs.widget(1))
        if task == "image":
            self.model_dialog.refresh_image_models()
        motion = getattr(self.window, "button_motion", None)
        if motion is not None:
            motion.attach(self.model_dialog)
        self.model_dialog.show()
        self.model_dialog.raise_()
        self.model_dialog.activateWindow()

    def _update_parameter_summary(self):
        self.parameter_summary.setText(
            f"主目标 ≥{self.config.min_pixel_height}px · "
            f"空镜/单目标/2–3/4+ 步长 "
            f"{self.config.idle_jump}/{self.config.step_single}/"
            f"{self.config.step_small_group}/{self.config.step_large_group}")

    def choose_video(self):
        if self.has_running_workers():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)")
        if not path:
            return
        try:
            info = video_info(path)
            self.config.video_path = path
            self.video_label.setText(
                f"{Path(path).name}\n{info['width']}×{info['height']} · "
                f"{info['fps']:.1f} FPS · {info['frames']} 帧")
            self.video_label.setToolTip(path)
        except Exception as error:
            self._error(str(error))

    def choose_output(self):
        if self.has_running_workers():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.config.output_dir = path
            self.output_label.setText(path)

    def choose_weight(self):
        if not self.model_combo.isEnabled():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.model_dialog, "加载本地权重", self.local_model_path, "YOLO 模型权重 (*.pt)")
        if path:
            try:
                if Path(path).suffix.lower() != ".pt":
                    raise ValueError("请选择 YOLO .pt 权重文件")
                with open(path, "rb") as stream:
                    stream.read(1)
            except (OSError, ValueError) as error:
                self._error(describe_file_error(error))
                return
            self.local_model_path = path
            self.config.model_path = path
            self.config.model_id = "local_yolo"
            self.model_combo.setCurrentIndex(self.model_combo.findData("local_yolo"))
            self.weight_label.setText(Path(path).name)
            self.weight_label.setToolTip(path)
            self.model_changed()

    def model_changed(self, *_):
        model_id = self.model_combo.currentData()
        spec = get_model(model_id)
        self.config.model_id = model_id
        if model_id == "local_yolo":
            self.config.model_path = self.local_model_path
        else:
            self.config.model_path = str(model_weight_path(model_id))
        self.model_source.setText(f"{spec.source} · {spec.license} · {spec.revision}")
        self.model_dialog.select_model(model_id)
        self.download_button.setVisible(model_id != "local_yolo")
        self.runtime_button.setVisible(model_id in (
            "yolo_world", "grounding_dino", "grounding_dino_base", "florence2",
            "sam3", "sam3_1")
            or model_id.startswith("yolo26_"))
        self.trust_code.setVisible(model_id == "florence2")
        self.weight_label.setText(Path(self.local_model_path).name)
        self.weight_label.setToolTip(self.local_model_path)
        if model_id == "local_yolo":
            status = "本地权重已选择 · 点击测试模型" if Path(self.config.model_path).is_file() else "未选择本地权重"
        else:
            status = {"missing": "未下载 · 点击下载", "downloaded": "本机已有 · 点击测试模型",
                      "ready": "本机已有 · 已通过加载测试"}[model_download_state(model_id)]
        self.model_status.setText(status)
        self.model_summary.setText(f"{spec.title} · {status}")

    def edit_parameters(self):
        if self.has_running_workers():
            return
        dialog = ParameterDialog(self.config, self)
        motion = getattr(self.window, "button_motion", None)
        if motion is not None:
            motion.attach(dialog)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.config = replace(self.config, **dialog.values())
            self._update_parameter_summary()

    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存配置", "autolabel.json", "JSON (*.json)")
        if path:
            try:
                self._current_config().save(path)
                self._log(f"已保存配置：{path}")
            except Exception as error:
                self._error(str(error))

    def load_config(self):
        if self.has_running_workers():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载配置", "", "JSON (*.json)")
        if not path:
            return
        try:
            loaded = ProjectConfig.load(path)
            get_model(loaded.model_id)
            self.config = loaded
            if loaded.model_id == "local_yolo":
                self.local_model_path = loaded.model_path
            self.model_combo.setCurrentIndex(self.model_combo.findData(loaded.model_id))
            self.prompts.setText(loaded.prompt_labels)
            self.trust_code.setChecked(loaded.trust_remote_code)
            self.video_label.setText(Path(loaded.video_path).name)
            self.output_label.setText(loaded.output_dir)
            self.model_changed()
            self._update_parameter_summary()
            self._log(f"已加载配置：{path}")
        except Exception as error:
            self._error(str(error))

    def _model_task(self, action):
        if self.has_running_workers():
            return
        try:
            current = self._current_config()
        except ValueError as error:
            self._error(str(error))
            return
        self.model_worker = ModelWorker(
            action, current.model_id, current.model_path,
            current.prompt_labels, current.device,
            current.image_size, current.confidence,
            current.trust_remote_code, self)
        self.model_worker.progress.connect(self._model_progress)
        self.model_worker.message.connect(self._log)
        self.model_worker.failed.connect(self._model_failed)
        self.model_worker.completed.connect(self._model_done)
        self.model_worker.finished.connect(self._refresh_task_controls)
        self._set_model_selection_enabled(False)
        for button in (self.download_button, self.test_button, self.runtime_button):
            button.setEnabled(False)
        self.model_cancel_button.setEnabled(True)
        self.model_started = time.monotonic()
        self.model_phase = {"download": "正在连接下载源", "test": "正在加载并测试模型",
                            "install_runtime": "正在安装依赖"}[action]
        self.model_progress.setVisible(True)
        self.model_progress.setRange(0, 0)
        self.model_progress.setFormat("处理中…")
        self._model_tick()
        self.model_timer.start()
        self.model_worker.start()
        self._refresh_task_controls()

    def download_model(self):
        if self.config.model_id == "local_yolo":
            return
        if model_download_state(self.config.model_id) != "missing":
            self.model_status.setText("本机已有模型文件 · 点击测试模型验证可用性")
            return
        spec = get_model(self.config.model_id)
        answer = QtWidgets.QMessageBox.question(
            self, "确认下载", f"{spec.title}\n来源：{spec.source_url}\n"
            f"许可：{spec.license}\n版本：{spec.revision}\n\n"
            "下载到本次临时目录？视频任务结束后自动删除。")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._model_task("download")

    def test_model(self):
        if not Path(self.config.model_path).exists():
            self._error("请先导入或下载模型")
            return
        if self.config.model_id == "florence2" and not self.trust_code.isChecked():
            self._error("Florence-2 需先确认并信任模型代码")
            return
        self._model_task("test")

    def install_runtime(self):
        answer = QtWidgets.QMessageBox.question(
            self, "安装依赖", "仅安装当前模型依赖，不自动升级 PyTorch/CUDA。继续？")
        if answer == QtWidgets.QMessageBox.StandardButton.Yes:
            self._model_task("install_runtime")

    def _model_progress(self, event):
        total = event.get("total", 0)
        downloaded = event.get("downloaded", 0)
        if event.get("unit") == "files":
            self.model_phase = f"正在下载文件 {downloaded}/{total or '?'} · 大文件期间进度可能暂不变化"
        else:
            amount = f"{downloaded / 1048576:.1f} MB"
            if total:
                amount += f" / {total / 1048576:.1f} MB"
            self.model_phase = f"正在下载 {event.get('file', '模型')} · {amount}"
        self.model_progress.setRange(0, 100 if total else 0)
        if total:
            self.model_progress.setValue(min(100, int(100 * downloaded / total)))
            self.model_progress.setFormat("%p%")
        self._model_tick()

    def _model_tick(self):
        elapsed = int(time.monotonic() - self.model_started)
        self.model_status.setText(f"{self.model_phase} · 已用 {elapsed} 秒")

    def _finish_model_task(self):
        self.model_timer.stop()
        self._set_model_selection_enabled(True)
        for button in (self.download_button, self.test_button, self.runtime_button):
            button.setEnabled(True)
        self.model_cancel_button.setEnabled(False)
        self.model_progress.setVisible(False)
        self._refresh_task_controls()

    def cancel_model_task(self):
        if self.model_worker and self.model_worker.isRunning():
            self.model_worker.cancel()
            self.model_phase = "正在取消模型任务"
            self._model_tick()

    def _model_failed(self, message):
        self._finish_model_task()
        self._error(message)

    def _model_done(self, result):
        self._finish_model_task()
        if result["action"] == "download":
            self.model_status.setText("下载完成 · 请测试模型")
        elif result["action"] == "test":
            self.model_status.setText("模型可用")
        else:
            self.model_status.setText("依赖已安装 · 请重启应用")
        self.model_summary.setText(f"{get_model(self.config.model_id).title} · {self.model_status.text()}")

    def _current_config(self):
        labels = parse_prompts(self.prompts.text().strip())
        if self.config.model_id == "local_yolo" and labels != ["person"]:
            raise ValueError("当前本地 YOLO 基线只检测 person；如需 person、phone 等自由类别，请选择开放词汇模型")
        return replace(self.config, prompt_labels=", ".join(labels),
                       trust_remote_code=self.trust_code.isChecked())

    def start(self):
        if self.has_running_workers():
            return
        try:
            config = self._current_config()
            config.validate()
            if not Path(config.video_path).is_file() or not Path(config.model_path).exists():
                raise ValueError("请先选择有效视频和模型")
            self.pipeline_worker = VideoPipelineWorker(config, self)
            self.pipeline_worker.progress.connect(self._progress)
            self.pipeline_worker.failed.connect(self._error)
            self.pipeline_worker.completed.connect(self._completed)
            self.pipeline_worker.finished.connect(self._pipeline_finished)
            self.pipeline_worker.finished.connect(self._refresh_task_controls)
            self.start_button.setEnabled(False)
            self._set_model_selection_enabled(False)
            self.progress_bar.setValue(0)
            self.live_review_dir = None
            self._log("开始处理视频")
            self.pipeline_worker.start()
            self._refresh_task_controls()
        except Exception as error:
            self._error(str(error))

    def _progress(self, event):
        self.progress_bar.setMaximum(event["total"])
        self.progress_bar.setValue(event["frame"])
        self.stats.setText(
            f"源帧 {event['frame']}/{event['total']} · 推理 {event['scanned']} · "
            f"保存 {event['saved']} · 淘汰 {event['discarded']}")
        self._log(event["reason"])
        if event.get("image"):
            self._show_live_image(event["image"])

    def _show_live_image(self, image):
        view = self.window.labeling_widget.view
        image = str(image)
        directory = str(Path(image).parent)
        if self.live_review_dir != directory:
            if view.dirty:
                self._log("当前标注未保存，已生成新帧；保存后可在图片列表中查看")
                return
            view.output_dir = None
            view.output_file = None
            view.import_image_folder(directory)
            self.live_review_dir = directory
            if image in view.fn_to_index and view.filename != image:
                view.load_file(image)
            return
        if image in view.fn_to_index:
            return
        item = view._create_file_list_item(image, str(Path(image).with_suffix(".json")))
        view.file_list_widget.addItem(item)
        view.fn_to_index[image] = view.file_list_widget.count() - 1
        view.toggle_actions(True)
        if view.dirty:
            self._log("新帧已加入图片列表；当前标注未保存，暂不切换画布")
            return
        view.load_file(image)

    def cancel(self):
        if self.pipeline_worker and self.pipeline_worker.isRunning():
            self.pipeline_worker.cancel()
            self._log("正在取消…")
        if self.model_worker and self.model_worker.isRunning():
            self.cancel_model_task()
        for worker in (self.sync_worker, self.prepare_worker):
            if worker and worker.isRunning():
                worker.requestInterruption()

    def _completed(self, result):
        status = "已取消" if result.get("cancelled") else "处理完成"
        self._log(f"{status}：保存 {result['saved']}，淘汰 {result['discarded']}")
        if result["saved"] and self.live_review_dir != str(Path(self.config.output_dir) / "images"):
            self.open_review()

    def _pipeline_finished(self):
        self.start_button.setEnabled(True)
        self._set_model_selection_enabled(True)
        model_id = self.pipeline_worker.config.model_id
        if model_id != "local_yolo":
            try:
                if release_session_model(model_id):
                    self._log("临时模型已删除；下次构建需重新下载")
                    self.model_changed()
            except OSError as error:
                self._log(f"临时模型清理失败：{error}")

    def _set_model_selection_enabled(self, enabled):
        for control in (self.model_combo, self.weight_button, self.model_dialog.square_button):
            control.setEnabled(enabled)

    def open_review(self):
        if self.has_running_workers():
            return
        images = Path(self.config.output_dir) / "images"
        image_files = list(images.glob("*.jpg")) if images.is_dir() else []
        if not image_files:
            self._error("当前项目没有可复核图片")
            return
        if any(not image.with_suffix(".json").is_file() for image in image_files):
            if self.prepare_worker and self.prepare_worker.isRunning():
                return
            self.prepare_worker = ReviewPrepareWorker(self.config.output_dir, self)
            self.prepare_worker.completed.connect(lambda _result: self._load_review_images())
            self.prepare_worker.failed.connect(self._error)
            self.prepare_worker.finished.connect(self._refresh_task_controls)
            self.prepare_worker.start()
            self._refresh_task_controls()
            self._log("正在准备原生标注文件…")
            return
        self._load_review_images()

    def _load_review_images(self):
        images = Path(self.config.output_dir) / "images"
        view = self.window.labeling_widget.view
        view.output_dir = None
        view.output_file = None
        view.import_image_folder(str(images))

    def sync_review(self):
        if self.has_running_workers():
            return
        view = self.window.labeling_widget.view
        if not view.may_continue():
            return
        self.sync_worker = SyncWorker(self.config.output_dir, self)
        self.sync_worker.completed.connect(self._sync_done)
        self.sync_worker.failed.connect(self._error)
        self.sync_worker.finished.connect(self._refresh_task_controls)
        self.sync_worker.start()
        self._refresh_task_controls()
        self._log("正在同步更新标签…")

    def _sync_done(self, result):
        self._log(f"YOLO 标签已更新：{result['labels']} 个文件")

    def _log(self, message):
        self.log.appendPlainText(str(message))

    def _error(self, message):
        if hasattr(self.window, "_close_timer") and self.window._close_timer.isActive():
            self._log(str(message))
            return
        self._log("错误：" + str(message))
        self.model_status.setText("错误：" + str(message))
        QtWidgets.QMessageBox.warning(self, "AutoLabel", str(message))

    def stop_workers(self):
        self.cancel()
