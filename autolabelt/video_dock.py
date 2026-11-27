"""Video dataset controls docked beside AutoLabel's native canvas."""

from dataclasses import replace
import math
from pathlib import Path
import time

from PyQt6 import QtCore, QtWidgets

from autolabelt.workbench.views.labeling.annotation_io import describe_file_error

from .branding import APP_NAME
from .core import DYNAMIC_JUMP_KEYS, ProjectConfig, video_info
from .video_batch import (TARGET_COUNT_BUCKETS, analyze_videos, plan_signature, recommend_sampling,
                          run_batch, video_paths)
from .video_widgets import SummaryLabel, WorkflowCard
from .models.downloader import model_download_state, model_weight_path
from .models.registry import MODEL_REGISTRY, get_model
from .model_square import ModelSourceLink, ModelSquareDialog, SQUARE_MODEL_IDS
from .models.worker import ModelWorker
from .models.adapters import parse_prompts
from .annotation_bridge import prepare_review_dataset, prepare_review_image, review_sync_location, sync_review_to_yolo


DEFAULT_VIDEO_DOCK_WIDTH = 332


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
    ("动态抽帧", DYNAMIC_JUMP_KEYS),
)


def analysis_detail(report):
    counts = report["target_counts"]
    estimates = report["estimated_frames"]
    lines = [f"{Path(report['path']).name} · {report['frames']:,} 帧 · 抽检 {report['probes']} 帧"]
    for key, label in TARGET_COUNT_BUCKETS:
        lines.append(f"{label} · 抽检 {counts[key]} 帧 · 全片估计 {estimates[key]:,} 帧")
    lines.append(f"符合画面筛选条件 {report['rate']:.1%}")
    if report["quota"]:
        lines.append(f"分配 {report['quota']} 张 · 初始间隔 {report['recommended_jump']} 帧")
    return "\n".join(lines)


class ParameterDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"处理参数 · {APP_NAME}")
        self.setObjectName("parameterDialog")
        self.resize(620, 704)
        self.controls = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)
        heading = QtWidgets.QLabel("处理参数")
        heading.setObjectName("dialogHeading")
        layout.addWidget(heading)
        hint = QtWidgets.QLabel("检查或调整参数后，点击“使用参数”确认。")
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
                if key == "idle_jump":
                    control.setToolTip("未检出合格目标或画面未通过筛选时，跳过的帧数。")
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
        apply = QtWidgets.QPushButton("使用参数")
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
        self.setWindowTitle(f"视频模型加载 · {APP_NAME}")
        self.setObjectName("modelLoadDialog")
        self.resize(940, 700)
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(24, 22, 24, 20)
        root_layout.setSpacing(14)
        heading = QtWidgets.QLabel("视频模型")
        heading.setObjectName("dialogHeading")
        root_layout.addWidget(heading)
        hint = QtWidgets.QLabel("选择视频检测模型，设置类别并验证加载；图片模型在左侧图片标注中选择。")
        hint.setObjectName("dialogHint")
        root_layout.addWidget(hint)
        view = getattr(getattr(dock.window, "labeling_widget", None), "view", None)
        self.native_models = getattr(view, "auto_labeling_widget", None)
        video_page = QtWidgets.QWidget()
        video_page.setObjectName("videoModelPage")
        root_layout.addWidget(video_page, 1)
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
        cloud_hint = QtWidgets.QLabel("浏览 Grounding DINO 与 SAM 系列模型。")
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
        layout.addLayout(columns)
        settings_card = QtWidgets.QFrame()
        settings_card.setProperty("modelCard", True)
        settings_layout = QtWidgets.QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(17, 15, 17, 15)
        settings_layout.setSpacing(10)
        self.settings_title = QtWidgets.QLabel()
        self.settings_title.setObjectName("modelColumnTitle")
        settings_layout.addWidget(self.settings_title)
        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        settings_layout.addWidget(self.details)
        self.spec_metadata = QtWidgets.QLabel()
        self.spec_metadata.setProperty("tone", "muted")
        self.spec_metadata.setWordWrap(True)
        settings_layout.addWidget(self.spec_metadata)
        self.source_link = ModelSourceLink()
        settings_layout.addWidget(self.source_link)
        prompt_row = QtWidgets.QHBoxLayout()
        prompt_label = QtWidgets.QLabel("检测类别  *")
        prompt_row.addWidget(prompt_label)
        prompt_row.addWidget(dock.prompts, 1)
        settings_layout.addLayout(prompt_row)
        self.prompt_hint = QtWidgets.QLabel()
        self.prompt_hint.setProperty("tone", "muted")
        self.prompt_hint.setWordWrap(True)
        settings_layout.addWidget(self.prompt_hint)
        settings_layout.addWidget(dock.trust_code)
        actions = QtWidgets.QHBoxLayout()
        for button in (dock.download_button, dock.test_button, dock.runtime_button,
                       dock.model_cancel_button):
            actions.addWidget(button)
        actions.addStretch()
        settings_layout.addLayout(actions)
        settings_layout.addWidget(dock.model_progress)
        settings_layout.addWidget(dock.model_status)
        layout.addWidget(settings_card, 1)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        confirm = QtWidgets.QPushButton("使用此模型")
        confirm.setProperty("role", "primary")
        confirm.clicked.connect(self.confirm_selection)
        footer.addWidget(confirm)
        layout.addLayout(footer)
        self.select_model(dock.config.model_id)

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
            if model_id in SQUARE_MODEL_IDS:
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
        self.settings_title.setText(f"{spec.title} · 模型设置")
        self.details.setText(spec.description)
        revision = spec.revision if len(spec.revision) <= 20 else spec.revision[:12] + "…"
        self.spec_metadata.setText(f"{spec.source}  ·  {spec.license}  ·  {revision}")
        self.spec_metadata.setToolTip(f"来源：{spec.source}\n许可证：{spec.license}\n版本：{spec.revision}")
        self.source_link.set_source(spec.source_url)
        self.prompt_hint.setText(
            "当前本地 YOLO 基线只检测 person（类别 ID 0）。" if model_id == "local_yolo"
            else "COCO 预训练仅支持已有类别；开放词汇模型可输入新类别。"
            if model_id not in ("sam3", "sam3_1") else
            "SAM 3 使用文字概念生成分割结果，再转换为 YOLO 框；需授权和人工复核。")


class VideoPipelineWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(dict)
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, config, parent=None, analyze_only=False, plan=None, refresh_analysis=False):
        super().__init__(parent)
        self.config = config
        self._cancel = False
        self.analyze_only = analyze_only
        self.plan = plan
        self.refresh_analysis = refresh_analysis

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def report(event):
                if event.get("image"):
                    prepare_review_image(self.config.output_dir, event["image"])
                if event.get("phase") == "plan":
                    event = {**event, "signature": plan_signature(self.config)}
                self.progress.emit(event)

            if self.analyze_only:
                plan = analyze_videos(self.config, report, lambda: self._cancel)
                result = {"analysis": True, "plan": plan, "cancelled": self._cancel,
                          "signature": plan_signature(self.config)}
            else:
                result = run_batch(self.config, report, lambda: self._cancel, plan=self.plan,
                                   refresh_analysis=self.refresh_analysis)
            # Each saved sample was already converted and checked by report().
            self.completed.emit(result)
        except Exception as error:
            self.failed.emit(describe_file_error(error))


class SyncWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, root, classes_file, parent=None, *, active_annotation=None):
        super().__init__(parent)
        self.root = root
        self.classes_file = classes_file
        self.active_annotation = active_annotation

    def run(self):
        try:
            self.completed.emit(sync_review_to_yolo(
                self.root, self.isInterruptionRequested, classes_file=self.classes_file,
                active_annotation=self.active_annotation,
            ))
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
        self.setMinimumWidth(318)
        self.window = window
        self.collapsed = False
        self._expanded_width = DEFAULT_VIDEO_DOCK_WIDTH
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
        self.analysis_plan = None
        self.analysis_signature = None
        self.confirmed_signature = None
        self._use_recommended_jumps = True
        content = QtWidgets.QWidget()
        content.setObjectName("videoDockContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(12)

        project = WorkflowCard("01", "素材队列", "添加 · 排序 · 输出")
        project_form = QtWidgets.QFormLayout()
        project_form.setContentsMargins(0, 0, 0, 0)
        project_form.setSpacing(8)
        project.body.addLayout(project_form)
        project_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.DontWrapRows)
        self.video_button = QtWidgets.QPushButton("添加视频")
        self.video_button.setToolTip("可以一次选择多个视频")
        self.video_button.clicked.connect(self.choose_video)
        self.video_label = SummaryLabel("未选择")
        project.set_caption_widget(self.video_label)
        self.video_queue = QtWidgets.QTreeWidget()
        self.video_queue.setObjectName("videoQueue")
        self.video_queue.setHeaderLabels(["视频 / 时长", "目标", "状态"])
        self.video_queue.setRootIsDecorated(False)
        self.video_queue.setMinimumHeight(62)
        self.video_queue.setMaximumHeight(108)
        self.video_queue.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
        self.video_queue.setAlternatingRowColors(True)
        self.video_queue.setUniformRowHeights(True)
        queue_header = self.video_queue.header()
        queue_header.setStretchLastSection(False)
        queue_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        queue_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        project_form.addRow(self.video_queue)
        queue_actions = QtWidgets.QHBoxLayout()
        queue_actions.setSpacing(7)
        self.queue_up = QtWidgets.QPushButton("上移")
        self.queue_down = QtWidgets.QPushButton("下移")
        self.queue_remove = QtWidgets.QPushButton("移除")
        self.queue_up.clicked.connect(lambda: self.move_video(-1))
        self.queue_down.clicked.connect(lambda: self.move_video(1))
        self.queue_remove.clicked.connect(self.remove_video)
        for button in (self.queue_up, self.queue_down, self.queue_remove):
            queue_actions.addWidget(button)
        queue_actions.insertWidget(0, self.video_button, 2)
        project_form.insertRow(0, queue_actions)
        self.output_button = QtWidgets.QPushButton("输出目录")
        self.output_button.clicked.connect(self.choose_output)
        self.output_label = SummaryLabel("未选择")
        project_form.addRow(self.output_button, self.output_label)
        project_form.itemAt(project_form.rowCount() - 1, QtWidgets.QFormLayout.ItemRole.FieldRole).setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(project, 3)

        model_card = WorkflowCard("02", "检测模型", "")
        model_layout = model_card.body
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
        self.prompts.textChanged.connect(self._invalidate_analysis)
        self.prompts.setPlaceholderText("类别：person, phone")
        self.trust_code = QtWidgets.QCheckBox("信任 Florence-2 固定版本模型代码")
        self.trust_code.toggled.connect(self._invalidate_analysis)
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
        model_row = QtWidgets.QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addStretch()
        self.model_button.setMinimumWidth(160)
        model_row.addWidget(self.model_button)
        model_row.addStretch()
        self.model_summary = SummaryLabel()
        self.model_summary.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        model_card.set_caption_widget(self.model_summary)
        model_layout.addLayout(model_row)
        layout.addWidget(model_card)

        parameter_card = WorkflowCard("03", "采样计划", "")
        parameter_layout = parameter_card.body
        self.parameter_button = QtWidgets.QPushButton("调整参数")
        self.parameter_button.clicked.connect(self.edit_parameters)
        parameter_row = QtWidgets.QHBoxLayout()
        parameter_row.setSpacing(8)
        parameter_row.addWidget(self.parameter_button)
        self.parameter_summary = SummaryLabel()
        parameter_row.addWidget(self.parameter_summary, 1)
        parameter_layout.addLayout(parameter_row)
        self.target_count = QtWidgets.QSpinBox()
        self.target_count.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.target_count.setRange(0, 1000000)
        self.target_count.setToolTip("输出目录最终希望保留的图片总数。填写 0 可按动态步长扫描。")
        self.target_count.valueChanged.connect(self._refresh_sampling_plan)
        self.sample_gap = QtWidgets.QSpinBox()
        self.sample_gap.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.sample_gap.setRange(1, 1000000)
        self.sample_gap.setValue(25)
        self.sample_gap.setSuffix(" 帧")
        self.sample_gap.setToolTip("指定图片数量时，同一视频保存图片之间至少间隔多少源帧。")
        self.sample_gap.valueChanged.connect(self._refresh_sampling_plan)
        target_form = QtWidgets.QHBoxLayout()
        target_form.setSpacing(4)
        target_form.addWidget(QtWidgets.QLabel("数据集图片数量"))
        target_form.addWidget(self.target_count, 1)
        target_form.addSpacing(4)
        target_form.addWidget(QtWidgets.QLabel("最小间隔"))
        target_form.addWidget(self.sample_gap, 1)
        parameter_layout.addLayout(target_form)
        self.analyze_button = QtWidgets.QPushButton("预分析并推荐")
        self.analyze_button.clicked.connect(self.analyze)
        self.analysis_summary = SummaryLabel("未预分析", lines=2)
        configuration_actions = QtWidgets.QHBoxLayout()
        configuration_actions.setSpacing(8)
        parameter_layout.addWidget(self.analyze_button)
        save_configuration = QtWidgets.QPushButton("保存配置")
        save_configuration.clicked.connect(self.save_config)
        load_configuration = QtWidgets.QPushButton("加载配置")
        load_configuration.clicked.connect(self.load_config)
        configuration_actions.addWidget(save_configuration)
        configuration_actions.addWidget(load_configuration)
        parameter_layout.addLayout(configuration_actions)
        parameter_layout.addSpacing(3)
        self.analysis_results = QtWidgets.QFrame()
        self.analysis_results.setObjectName("samplingAnalysisResults")
        self.analysis_results.setFixedHeight(118)
        result_layout = QtWidgets.QVBoxLayout(self.analysis_results)
        result_layout.setContentsMargins(5, 4, 5, 4)
        result_layout.setSpacing(4)
        result_layout.addWidget(self.analysis_summary, alignment=QtCore.Qt.AlignmentFlag.AlignTop)
        self.sampling_plan_table = QtWidgets.QTreeWidget()
        self.sampling_plan_table.setObjectName("samplingPlanTable")
        self.sampling_plan_table.setAccessibleName("预分析采样参数")
        self.sampling_plan_table.setHeaderLabels(["视频", "分配张数", "初始间隔"])
        self.sampling_plan_table.setRootIsDecorated(False)
        self.sampling_plan_table.setAlternatingRowColors(True)
        self.sampling_plan_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.sampling_plan_table.setUniformRowHeights(True)
        self.sampling_plan_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sampling_plan_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.sampling_plan_table.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                              QtWidgets.QSizePolicy.Policy.Ignored)
        header = self.sampling_plan_table.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        result_layout.addWidget(self.sampling_plan_table, 1)
        self.sampling_plan_table.hide()
        parameter_layout.addWidget(self.analysis_results)
        layout.addWidget(parameter_card)

        run_card = WorkflowCard("04", "构建与复核", "")
        run_layout = run_card.body
        run_actions = QtWidgets.QHBoxLayout()
        run_actions.setSpacing(8)
        self.start_button = QtWidgets.QPushButton("开始构建")
        self.start_button.setProperty("role", "primary")
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QtWidgets.QPushButton("取消")
        self.cancel_button.setProperty("role", "danger")
        self.cancel_button.clicked.connect(self.cancel)
        run_actions.addWidget(self.start_button)
        run_actions.addWidget(self.cancel_button)
        run_layout.addLayout(run_actions)
        self.batch_progress = QtWidgets.QProgressBar()
        self.batch_progress.setObjectName("videoBatchProgress")
        self.batch_progress.setRange(0, 1000)
        self.batch_progress.setValue(0)
        self.batch_progress.setAccessibleName("整批进度")
        self.batch_progress.setTextVisible(False)
        run_layout.addSpacing(4)
        run_layout.addWidget(self.batch_progress)
        self.stats = SummaryLabel("等待视频", lines=2)
        run_layout.addWidget(self.stats)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setMaximumBlockCount(1000)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(48)
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                               QtWidgets.QSizePolicy.Policy.Expanding)
        self.log.setPlaceholderText("运行记录")
        run_layout.addWidget(self.log, 1)
        self.open_button = QtWidgets.QPushButton("在画布中复核")
        self.open_button.clicked.connect(self.open_review)
        self.sync_button = QtWidgets.QPushButton("同步更新标签")
        self.sync_button.setToolTip("保存当前画布并将复核 JSON 写回所属数据集的 YOLO 标签")
        self.sync_button.clicked.connect(self.sync_review)
        review_actions = QtWidgets.QHBoxLayout()
        review_actions.setSpacing(8)
        review_actions.addWidget(self.open_button)
        review_actions.addWidget(self.sync_button)
        run_layout.addLayout(review_actions)
        layout.addWidget(run_card, 4)
        for card in (project, model_card, parameter_card, run_card):
            card.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                               QtWidgets.QSizePolicy.Policy.Preferred)
            card.body.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            if card is not run_card:
                card.body.addStretch(1)
        self.workflow_body = content
        self.sidebar_scroll = QtWidgets.QScrollArea()
        self.sidebar_scroll.setObjectName("videoDockScroll")
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.sidebar_scroll.setWidget(content)
        shell = QtWidgets.QWidget()
        shell.setObjectName("videoSidebarShell")
        shell_layout = QtWidgets.QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        self.sidebar_heading = QtWidgets.QLabel("视频数据集")
        self.sidebar_heading.setObjectName("sidebarHeading")
        self.sidebar_heading.setContentsMargins(12, 5, 10, 5)
        shell_layout.addWidget(self.sidebar_heading)
        shell_layout.addWidget(self.sidebar_scroll, 1)
        self.training_footer = QtWidgets.QWidget()
        footer_layout = QtWidgets.QVBoxLayout(self.training_footer)
        footer_layout.setContentsMargins(10, 8, 10, 10)
        self.training_button = QtWidgets.QPushButton("开始训练")
        self.training_button.setProperty("role", "primary")
        self.training_button.setObjectName("startTrainingButton")
        self.training_button.setMinimumHeight(36)
        self.training_button.clicked.connect(lambda: self.window.open_training_center())
        footer_layout.addWidget(self.training_button)
        shell_layout.addWidget(self.training_footer)
        self.setWidget(shell)
        self.workflow_cards = (project, model_card, parameter_card, run_card)
        project.setObjectName("workflowFirstCard")
        run_card.setObjectName("workflowLastCard")
        self.workflow_layout = layout
        for button in content.findChildren(QtWidgets.QPushButton):
            button.setMinimumHeight(28)
            button.setMaximumHeight(32)
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                 QtWidgets.QSizePolicy.Policy.Preferred)
        for control in (self.target_count, self.sample_gap):
            control.setMinimumWidth(50)
            control.setMinimumHeight(28)
            control.setMaximumHeight(32)
            control.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self._project_controls = (self.video_button, self.output_button,
                                  self.parameter_button, load_configuration,
                                  self.prompts, self.trust_code, self.open_button,
                                  self.sync_button, self.start_button, self.training_button,
                                  self.video_queue, self.queue_up, self.queue_down,
                                  self.queue_remove, self.target_count, self.sample_gap,
                                  self.analyze_button)
        self.model_changed()

        self._update_parameter_summary()

    def set_collapsed(self, collapsed):
        if self.collapsed == collapsed:
            return
        self.collapsed = collapsed
        if collapsed:
            self._expanded_width = self.width()
            self.hide()
        else:
            self.show()
            self.window.resizeDocks([self], [self._expanded_width], QtCore.Qt.Orientation.Horizontal)

    def has_running_workers(self):
        training = getattr(self.window, "training_window", None)
        if training and training.is_busy():
            return True
        return any(worker and getattr(worker, "isRunning", lambda: False)()
                   for worker in (self.pipeline_worker, self.model_worker,
                                  self.sync_worker, self.prepare_worker))

    def _refresh_task_controls(self):
        enabled = not self.has_running_workers()
        for control in self._project_controls:
            control.setEnabled(enabled)
        self.start_button.setEnabled(enabled and self.confirmed_signature is not None
                                     and bool(self.config.output_dir))
        if self.analysis_plan is None:
            hint = "请先预分析"
        elif self.confirmed_signature is None:
            hint = "请在“调整参数”中点击“使用参数”"
        elif not self.config.output_dir:
            hint = "请选择输出目录"
        else:
            hint = "使用已确认的参数开始构建"
        self.start_button.setToolTip(hint)
        training = getattr(self.window, "training_window", None)
        self.training_button.setEnabled(enabled or bool(training and training.is_busy()))
        self._set_model_selection_enabled(enabled)

    def open_model_dialog(self):
        motion = getattr(self.window, "button_motion", None)
        if motion is not None:
            motion.attach(self.model_dialog)
        self.model_dialog.show()
        self.model_dialog.raise_()
        self.model_dialog.activateWindow()

    def _update_parameter_summary(self):
        self.parameter_summary.setText(
            f"置信度 {self.config.confidence:.2f} · 主目标 ≥{self.config.min_pixel_height}px")

    def choose_video(self):
        if self.has_running_workers():
            return
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "添加视频（按列表顺序处理）", "", "视频 (*.mp4 *.mkv *.avi *.mov *.webm);;所有文件 (*)")
        if not paths:
            return
        errors = []
        existing = {str(Path(path).resolve()).casefold() for path in self.config.video_paths}
        for path in paths:
            if str(Path(path).resolve()).casefold() in existing:
                continue
            try:
                info = video_info(path)
                if info["frames"] <= 0 or info["fps"] <= 0:
                    raise ValueError(f"视频缺少有效帧数或 FPS: {path}")
                duration = info["duration"]
                item = QtWidgets.QTreeWidgetItem([f"{Path(path).name} · {duration:.0f}s", "—", "等待"])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, path)
                item.setToolTip(0, f"{path}\n{info['width']}×{info['height']} · {info['fps']:.1f} FPS · {info['frames']} 帧")
                self.video_queue.addTopLevelItem(item)
                existing.add(str(Path(path).resolve()).casefold())
            except Exception as error:
                errors.append(str(error))
        self._sync_queue()
        if errors:
            self._error("\n".join(errors))

    def _sync_queue(self):
        self.config.video_paths = [self.video_queue.topLevelItem(index).data(
            0, QtCore.Qt.ItemDataRole.UserRole) for index in range(self.video_queue.topLevelItemCount())]
        self.config.video_path = self.config.video_paths[0] if self.config.video_paths else ""
        self.video_label.setText(f"{len(self.config.video_paths)} 个视频" if self.config.video_paths else "未选择")
        self._invalidate_analysis()

    def move_video(self, offset):
        if self.has_running_workers():
            return
        index = self.video_queue.indexOfTopLevelItem(self.video_queue.currentItem())
        other = index + offset
        if index >= 0 and 0 <= other < self.video_queue.topLevelItemCount():
            item = self.video_queue.takeTopLevelItem(index)
            self.video_queue.insertTopLevelItem(other, item)
            self.video_queue.setCurrentItem(item)
            self._sync_queue()

    def remove_video(self):
        if self.has_running_workers():
            return
        index = self.video_queue.indexOfTopLevelItem(self.video_queue.currentItem())
        if index >= 0:
            self.video_queue.takeTopLevelItem(index)
            self._sync_queue()

    def _invalidate_analysis(self, *_):
        self.analysis_plan = self.analysis_signature = None
        self.confirmed_signature = None
        if hasattr(self, "stats") and self.stats.text().startswith(("预分析完成", "参数已确认")):
            self.stats.setText("待预分析")
        if hasattr(self, "analysis_summary"):
            self.analysis_summary.setText("待预分析")
            self.sampling_plan_table.clear()
            self.sampling_plan_table.hide()
            for index in range(self.video_queue.topLevelItemCount()):
                item = self.video_queue.topLevelItem(index)
                item.setText(1, "—")
                item.setToolTip(1, "")
                if item.text(2) in ("已分析", "预分析"):
                    item.setText(2, "等待")
                    item.setToolTip(2, "")
        if hasattr(self, "_project_controls"):
            self._refresh_task_controls()

    def _refresh_sampling_plan(self, *_):
        if self.analysis_plan is not None:
            self.confirmed_signature = None
            config = replace(self.config, target_count=self.target_count.value(),
                             min_sample_frames=self.sample_gap.value())
            self.analysis_plan = recommend_sampling(self.analysis_plan["videos"], config)
            self._show_analysis(self.analysis_plan)
            self._update_analysis_status()

    def _update_analysis_status(self):
        if self.confirmed_signature is None:
            summary = "预分析完成 · 待确认参数"
            detail = "打开“调整参数”并点击“使用参数”"
        else:
            summary = "参数已确认"
            detail = "可开始构建" if self.config.output_dir else "请选择输出目录"
            if self.analysis_signature != self.confirmed_signature:
                summary += " · 构建前将重新抽检"
        self.stats.setText(f"{summary}\n{detail}")
        self._refresh_task_controls()

    def choose_output(self):
        if self.has_running_workers():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.config.output_dir = path
            self.output_label.setText(path)
            if self.analysis_plan and self.stats.text().startswith(("预分析完成", "参数已确认")):
                self._update_analysis_status()
            else:
                self._refresh_task_controls()

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
        self._invalidate_analysis()
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
        self._update_model_status(model_id)

    def _update_model_status(self, model_id):
        if model_id == "local_yolo":
            status = "本地权重已选择 · 点击测试模型" if Path(self.config.model_path).is_file() else "未选择本地权重"
        else:
            status = {"missing": "未下载 · 点击下载", "downloaded": "本机已有 · 点击测试模型",
                      "ready": "本机已有 · 已通过加载测试"}[model_download_state(model_id)]
        self.model_status.setText(status)
        self.model_summary.setText(f"{get_model(model_id).title} · {status}")

    def edit_parameters(self):
        if self.has_running_workers():
            return
        shown_config = self._parameter_config()
        dialog = ParameterDialog(shown_config, self)
        motion = getattr(self.window, "button_motion", None)
        if motion is not None:
            motion.attach(dialog)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            dialog.deleteLater()
            return
        values = dialog.values()
        dialog.deleteLater()
        self.config = replace(self.config, **values)
        self._update_parameter_summary()
        if self.analysis_plan is None:
            return
        try:
            self.confirmed_signature = plan_signature(self._current_config())
        except (OSError, ValueError) as error:
            self._invalidate_analysis()
            self._error(str(error))
            return
        if (self.confirmed_signature != self.analysis_signature or
                any(values[key] != getattr(shown_config, key) for key in DYNAMIC_JUMP_KEYS)):
            self._use_recommended_jumps = False
        self._update_analysis_status()

    def save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存配置", "autolabelt.json", "JSON (*.json)")
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
            self._use_recommended_jumps = False
            if loaded.model_id == "local_yolo":
                self.local_model_path = loaded.model_path
            self.model_combo.setCurrentIndex(self.model_combo.findData(loaded.model_id))
            self.prompts.setText(loaded.prompt_labels)
            self.trust_code.setChecked(loaded.trust_remote_code)
            self.video_queue.clear()
            for video in video_paths(loaded):
                item = QtWidgets.QTreeWidgetItem([Path(video).name, "—", "等待"])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, video)
                item.setToolTip(0, video)
                self.video_queue.addTopLevelItem(item)
            self.target_count.setValue(loaded.target_count)
            frames = loaded.min_sample_frames
            if frames is None:
                try:
                    fps = video_info(video_paths(loaded)[0])["fps"]
                except (OSError, ValueError):
                    fps = 25
                fps = fps if math.isfinite(fps) and fps > 0 else 25
                frames = max(1, math.ceil(fps * loaded.min_sample_seconds))
            self.sample_gap.setValue(frames)
            self._sync_queue()
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
            "下载到本次临时目录？关闭软件前可以反复使用。")
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

    def _parameter_config(self):
        if self._use_recommended_jumps and self.analysis_plan is not None:
            return replace(self.config, **self.analysis_plan["dynamic_jumps"])
        return self.config

    def _current_config(self):
        labels = parse_prompts(self.prompts.text().strip())
        if self.config.model_id == "local_yolo" and labels != ["person"]:
            raise ValueError("当前本地 YOLO 基线只检测 person；如需 person、phone 等自由类别，请选择开放词汇模型")
        return replace(self._parameter_config(), prompt_labels=", ".join(labels),
                       trust_remote_code=self.trust_code.isChecked(),
                       target_count=self.target_count.value(),
                       min_sample_frames=self.sample_gap.value())

    def start(self):
        self._start_video_task(False)

    def analyze(self):
        self._start_video_task(True)

    def _start_video_task(self, analyze_only):
        if self.pipeline_worker is not None or self.has_running_workers():
            return
        try:
            if not analyze_only and self.confirmed_signature is None:
                raise ValueError("请先预分析，再在“调整参数”中点击“使用参数”")
            config = self._current_config()
            config.validate(require_output=not analyze_only)
            if not all(Path(path).is_file() for path in video_paths(config)):
                raise ValueError("视频队列中有文件无法访问，请检查素材路径")
            if not Path(config.model_path).exists():
                raise ValueError("请先在“模型加载”中选择或下载所选模型的权重")
            signature = plan_signature(config)
            if not analyze_only and signature != self.confirmed_signature:
                self._invalidate_analysis()
                raise ValueError("视频或模型文件已变化，请重新预分析并确认参数")
            if analyze_only:
                self.confirmed_signature = None
            plan = self.analysis_plan if signature == self.analysis_signature else None
            refresh = not analyze_only and plan is None
            self.pipeline_worker = VideoPipelineWorker(config, self, analyze_only, plan, refresh)
            self.pipeline_worker.progress.connect(self._progress)
            self.pipeline_worker.failed.connect(self._video_failed)
            self.pipeline_worker.completed.connect(self._completed)
            self.pipeline_worker.finished.connect(self._pipeline_finished)
            self.start_button.setEnabled(False)
            self._set_model_selection_enabled(False)
            self._set_batch_progress(0)
            self.batch_progress.setRange(0, 0)
            self.stats.setText("正在加载模型 · 准备预分析" if analyze_only else "正在加载模型 · 准备构建")
            for index in range(self.video_queue.topLevelItemCount()):
                self.video_queue.topLevelItem(index).setText(2, "等待")
            self.live_review_dir = None
            self._log("开始预分析视频队列" if analyze_only else "开始按序构建数据集")
            self.pipeline_worker.start()
            self._refresh_task_controls()
        except Exception as error:
            self._error(str(error))

    def _set_batch_progress(self, fraction):
        if self.batch_progress.maximum() == 0:
            self.batch_progress.setRange(0, 1000)
        self.batch_progress.setValue(int(1000 * fraction))

    def _progress(self, event):
        phase = event.get("phase")
        index = event.get("video_index", 0)
        count = event.get("video_count", 1)
        item = self.video_queue.topLevelItem(index)
        if phase == "plan":
            self.analysis_plan = event["plan"]
            self.analysis_signature = event["signature"]
            self._show_analysis(self.analysis_plan)
            target = self.analysis_plan["target"]
            self._set_batch_progress(event.get("batch_saved", 0) / target if target else 0)
            self._log("预分析完成 · 开始构建")
            return
        if phase == "existing":
            saved, target = event["batch_saved"], event["target"]
            self._log(f"输出目录已有 {saved:,} 张，继续构建")
            if target:
                self._set_batch_progress(saved / target)
            return
        if phase == "analysis":
            fraction = (index + event["fraction"]) / count
            self._set_batch_progress(fraction)
            self.stats.setText(f"预分析 · 视频 {index + 1}/{count} · {self.batch_progress.value() // 10}%\n"
                               f"{event['reason']}")
            if item:
                item.setText(2, "预分析")
            return
        if phase == "video_start":
            action = "补采" if event.get("supplemental") else "构建"
            if item:
                item.setText(2, "补采中" if event.get("supplemental") else "处理中")
            self.stats.setText(f"视频 {index + 1}/{count} · 开始{action}")
            self._log(f"视频 {index + 1}/{count} · 开始{action}")
            return
        if phase == "video_done":
            status = "已取消" if event["cancelled"] else "已完成"
            if item:
                item.setText(2, status)
                item.setToolTip(2, f"本轮保存 {event['saved']} 张，整批累计 {event['batch_saved']} 张")
            if not event["cancelled"]:
                target = event["target"]
                self._set_batch_progress(event["batch_saved"] / target if target else (index + 1) / count)
            self._log(f"视频 {index + 1}/{count} · {status}，本轮保存 {event['saved']:,} 张")
            return
        saved, target = event["batch_saved"], event["target"]
        fraction = saved / target if target else (index + event["fraction"]) / count
        self._set_batch_progress(fraction)
        tally = f"{saved:,}/{target:,}" if target else f"{saved:,}"
        state = "补采" if event.get("supplemental") else "构建"
        self.stats.setText(
            f"{state} · 视频 {index + 1}/{count} · {self.batch_progress.value() // 10}%\n"
            f"整批 {tally} 张 · 本轮推理 {event['scanned']:,} 帧")
        if event.get("image"):
            self._show_live_image(event["image"])

    def _show_analysis(self, plan):
        self.sampling_plan_table.clear()
        lines = []
        for index, report in enumerate(plan["videos"]):
            item = self.video_queue.topLevelItem(index)
            detail = analysis_detail(report)
            if item:
                item.setText(1, str(report["quota"]) if plan["target"] else "—")
                item.setText(2, "已分析")
                item.setToolTip(1, detail)
                item.setToolTip(2, detail)
            table_item = QtWidgets.QTreeWidgetItem([
                Path(report["path"]).name, f"{report['quota']:,} 张" if plan["target"] else "—",
                f"{report['recommended_jump']:,} 帧" if report["recommended_jump"] else "—"])
            for column in range(table_item.columnCount()):
                table_item.setToolTip(column, detail)
            self.sampling_plan_table.addTopLevelItem(table_item)
            lines.append(detail)
        self.sampling_plan_table.setVisible(bool(lines))
        probes = sum(report["probes"] for report in plan["videos"])
        estimate = f"预计 {plan['estimated_saved']:,} / {plan['target']:,} 张"
        if plan["target"] and plan["estimated_saved"] >= plan["target"]:
            estimate += f" · 约 {plan['estimated_seconds'] / 60:.1f} 分钟"
        if not plan["target"]:
            estimate = "设置数据集图片数量后生成分配计划"
        text = f"共 {plan['total_frames']:,} 帧 · 抽检 {probes:,} 帧\n{estimate}"
        self.analysis_summary.setText(text)
        totals = [f"{label} {sum(report['estimated_frames'][key] for report in plan['videos']):,} 帧"
                  for key, label in TARGET_COUNT_BUCKETS]
        self.analysis_summary.setToolTip("\n\n".join(["全片帧数估算 · " + " · ".join(totals), *lines]))

    def _video_failed(self, message):
        if self.batch_progress.maximum() == 0:
            self._set_batch_progress(0)
        self.stats.setText("处理失败 · 详情见运行记录")
        for index in range(self.video_queue.topLevelItemCount()):
            item = self.video_queue.topLevelItem(index)
            if item.text(2) in ("处理中", "补采中", "预分析"):
                item.setText(2, "失败")
        self._error(message)

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
        if result.get("analysis"):
            if result.get("cancelled") or result.get("plan") is None:
                self._log("预分析已取消")
                self._invalidate_analysis()
                if self.batch_progress.maximum() == 0:
                    self._set_batch_progress(0)
                self.stats.setText("预分析已取消")
                return
            self.analysis_plan = result["plan"]
            self.analysis_signature = result["signature"]
            self.confirmed_signature = None
            self._use_recommended_jumps = True
            self._show_analysis(result["plan"])
            self._set_batch_progress(1)
            self._update_analysis_status()
            self._log("预分析完成 · 动态步长已回填")
            return
        status = "已取消" if result.get("cancelled") else "处理完成"
        self._log(f"{status}：保存 {result['saved']}，淘汰 {result['discarded']}")
        if result.get("existing"):
            detail = f"{status} · 数据集 {result['dataset_total']:,} 张 · 本次新增 {result['saved']:,} 张"
        else:
            detail = f"{status} · 共保存 {result['saved']:,} 张"
        if result.get("shortfall"):
            detail += f" · 距目标少 {result['shortfall']} 张"
            self._log("未达到目标：在当前质量门槛与最小间隔下，可用采样位置不足。")
        self.stats.setText(detail)
        if not result.get("cancelled"):
            self._set_batch_progress(result["dataset_total"] / result["target"] if result.get("shortfall") else 1)
        elif self.batch_progress.maximum() == 0:
            self._set_batch_progress(0)
        if result["saved"] and self.live_review_dir != str(Path(self.config.output_dir) / "images"):
            self.open_review()

    def _pipeline_finished(self):
        worker = self.pipeline_worker
        self.pipeline_worker = None
        worker.deleteLater()
        self._refresh_task_controls()

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
            self._start_review_worker(
                "prepare_worker", ReviewPrepareWorker(self.config.output_dir, self),
                lambda _result: self._load_review_images(),
            )
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
        try:
            image, root = review_sync_location(view.image_path, self.config.output_dir)
        except ValueError as error:
            self._error(str(error))
            return
        default_classes = root / "classes.txt"
        classes_file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择类别文件", str(default_classes if default_classes.is_file() else root),
            "类别文件 (*.txt);;所有文件 (*)",
        )
        if not classes_file:
            return
        if not view.save_annotations_for_sync():
            self._log("当前复核标注保存失败，未写入任何 YOLO 标签。")
            return
        active_annotation = None
        if image is not None:
            annotation = Path(view.get_label_file())
            if annotation.is_file() or view.label_file is not None:
                active_annotation = (image, annotation)
        self._start_review_worker(
            "sync_worker", SyncWorker(root, classes_file, self, active_annotation=active_annotation), self._sync_done
        )
        self._log(f"正在同步更新标签：{root / 'labels'}")

    def _start_review_worker(self, slot, worker, completed):
        setattr(self, slot, worker)
        worker.completed.connect(completed)
        worker.failed.connect(self._error)
        worker.finished.connect(lambda: self._retire_review_worker(slot, worker))
        worker.start()
        self._refresh_task_controls()

    def _retire_review_worker(self, slot, worker):
        if getattr(self, slot) is worker:
            setattr(self, slot, None)
        worker.deleteLater()
        self._refresh_task_controls()

    def _sync_done(self, result):
        self._log(f"YOLO 标签已更新：{result['labels']} 个文件 · {result['labels_dir']}")
        if result["skipped"]:
            self._log(f"另有 {result['skipped']} 张图片未保存复核 JSON，保留其原标签。")

    def _log(self, message):
        message = str(message)
        if self.log.document().lastBlock().text() == message:
            return
        scrollbar = self.log.verticalScrollBar()
        position = scrollbar.value()
        follow = position == scrollbar.maximum()
        self.log.appendPlainText(message)
        if not follow:
            scrollbar.setValue(position)

    def _error(self, message):
        if hasattr(self.window, "_close_timer") and self.window._close_timer.isActive():
            self._log(str(message))
            return
        self._log("错误：" + str(message))
        self.model_status.setText("错误：" + str(message))
        QtWidgets.QMessageBox.warning(self, APP_NAME, str(message))

    def stop_workers(self):
        self.cancel()
