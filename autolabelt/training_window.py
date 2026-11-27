"""Independent detection training window with structured live telemetry."""

import json
import math
import re
import sys
import time
import uuid
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from .branding import APP_NAME
from .training_chart import MetricChart
from .training_data import prepare_training_dataset
from .training_records import create_run_directory, read_training_record, training_history, training_weights
from .training_widgets import AnimatedComboBox, OnnxConversionDialog, training_stylesheet
from .workbench.services.auto_training.ultralytics.exporter import ExportEventRedirector, ExportManager
from .workbench.services.auto_training.ultralytics.trainer import TrainingManager, TrainingEventRedirector
from .workbench.views.labeling.annotation_io import atomic_write_text

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OPTIMIZERS = {name.casefold(): name for name in ("auto", "AdamW", "SGD", "Adam", "RMSProp")}
CACHE_MODES = {"false": False, "ram": "ram", "disk": "disk"}


class DatasetPreparation(QtCore.QThread):
    prepared = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, source, destination, ratio, parent=None, yaml_path=None, split_manifest=None,
                 reference_splits=False, seed=42, copy_existing_splits=False):
        super().__init__(parent)
        self.arguments = source, destination, ratio
        self.yaml_path = yaml_path
        self.split_manifest = split_manifest
        self.reference_splits = reference_splits
        self.seed = seed
        self.copy_existing_splits = copy_existing_splits

    def run(self):
        try:
            self.prepared.emit(prepare_training_dataset(
                *self.arguments, seed=self.seed, cancelled=self.isInterruptionRequested,
                yaml_path=self.yaml_path, split_manifest=self.split_manifest,
                reference_splits=self.reference_splits, copy_existing_splits=self.copy_existing_splits))
        except Exception as error:
            self.failed.emit(str(error))


class TrainingWindow(QtWidgets.QMainWindow):
    busy_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None, can_start=lambda: True):
        super().__init__(parent, QtCore.Qt.WindowType.Window)
        self.setWindowTitle(f"{APP_NAME} · 训练中心")
        self.setObjectName("trainingWindow")
        self.setFont(QtGui.QFont("Microsoft YaHei", 9))
        self.setStyleSheet(training_stylesheet())
        self.resize(1360, 900)
        self.can_start = can_start
        self.manager = TrainingManager()
        self.bridge = TrainingEventRedirector()
        self.bridge.training_event_signal.connect(self.on_event)
        self.manager.callbacks = [self.bridge.emit_training_event]
        self.export_manager = ExportManager()
        self.export_bridge = ExportEventRedirector()
        self.export_bridge.export_event_signal.connect(self.on_export_event)
        self.export_manager.callbacks = [self.export_bridge.emit_export_event]
        self.export_record = None
        self.preparation = None
        self.yaml_preparation = None
        self.last_automatic_split = None
        self.last_dock_source = None
        self.record = None
        self.started_at = None
        self.metrics = []
        self.log_progress = ""
        self.settings = QtCore.QSettings()
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_elapsed)
        self.timer.start()
        self.load_history()

    def _build_ui(self):
        body = QtWidgets.QWidget()
        body.setObjectName("trainingBody")
        self.setCentralWidget(body)
        layout = QtWidgets.QHBoxLayout(body)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(20)
        history_panel = QtWidgets.QFrame()
        history_panel.setObjectName("trainingSidebar")
        history_panel.setFixedWidth(210)
        history_layout = QtWidgets.QVBoxLayout(history_panel)
        history_layout.setContentsMargins(16, 22, 16, 16)
        history_layout.setSpacing(16)
        history_title = QtWidgets.QLabel("训练记录")
        history_title.setObjectName("sectionTitle")
        history_layout.addWidget(history_title)
        self.history = QtWidgets.QListWidget()
        self.history.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.history.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history.currentItemChanged.connect(self.select_history)
        history_layout.addWidget(self.history)
        layout.addWidget(history_panel)
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(14)
        layout.addLayout(right, 1)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("训练中心")
        title.setObjectName("trainingTitle")
        header.addWidget(title)
        brand = QtWidgets.QLabel(APP_NAME)
        brand.setObjectName("trainingBrand")
        header.addWidget(brand)
        header.addStretch()
        right.addLayout(header)
        self.configuration = QtWidgets.QFrame()
        self.configuration.setProperty("card", True)
        config_layout = QtWidgets.QVBoxLayout(self.configuration)
        config_layout.setContentsMargins(20, 18, 20, 18)
        config_layout.setSpacing(14)
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.source = QtWidgets.QLineEdit()
        self.source.setPlaceholderText("选择数据集")
        self.source.setToolTip("images / labels / classes.txt，或已有 data.yaml；开始训练时自动生成配置")
        form.addRow("数据集", self.path_row(self.source, "dataset"))
        self.output = QtWidgets.QLineEdit(self.settings.value(
            "training/output", str(Path.home() / "AutoLabelTraining")))
        form.addRow("保存位置", self.path_row(self.output, "directory"))
        self.yaml_path = QtWidgets.QLineEdit()
        self.yaml_path.setPlaceholderText("导入或生成 data.yaml")
        yaml_row = QtWidgets.QHBoxLayout()
        yaml_row.addWidget(self.yaml_path)
        self.yaml_browse_button = QtWidgets.QPushButton("浏览")
        self.yaml_browse_button.clicked.connect(lambda: self.browse(self.yaml_path, "yaml"))
        self.yaml_generate_button = QtWidgets.QPushButton("生成")
        self.yaml_generate_button.setToolTip("在保存位置生成 data.yaml 和目录式训练集、验证集")
        self.yaml_generate_button.clicked.connect(self.generate_yaml)
        yaml_row.addWidget(self.yaml_browse_button)
        yaml_row.addWidget(self.yaml_generate_button)
        yaml_row.addSpacing(12)
        self.ratio = QtWidgets.QSpinBox()
        self.ratio.setRange(5, 95)
        self.ratio.setValue(80)
        self.ratio.setSuffix("%")
        self.ratio.setMinimumWidth(84)
        self.ratio.setToolTip("用于新划分；已有 train/val 目录时沿用原划分")
        ratio_label = QtWidgets.QLabel("训练比例")
        ratio_label.setBuddy(self.ratio)
        yaml_row.addWidget(ratio_label)
        yaml_row.addSpacing(8)
        yaml_row.addWidget(self.ratio)
        form.addRow("YAML", yaml_row)
        self.python = QtWidgets.QLineEdit(sys.executable)
        self.python.setToolTip("默认使用当前环境；仅在需要其他虚拟环境时修改")
        self.family = AnimatedComboBox()
        self.family.addItems(["YOLO", "RT-DETR"])
        self.family.setFixedWidth(125)
        self.model = AnimatedComboBox()
        self.model.setEditable(True)
        self.family.currentIndexChanged.connect(self.model_family_changed)
        self.model_family_changed()
        models = QtWidgets.QHBoxLayout()
        models.addWidget(self.family)
        models.addWidget(self.model, 1)
        browse = QtWidgets.QPushButton("本地权重")
        browse.clicked.connect(self.choose_weight)
        models.addWidget(browse)
        form.addRow("模型", models)
        config_layout.addLayout(form)
        self.epochs = QtWidgets.QSpinBox()
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(150)
        self.batch = QtWidgets.QSpinBox()
        self.batch.setRange(1, 4096)
        self.batch.setValue(24)
        self.imgsz = QtWidgets.QSpinBox()
        self.imgsz.setRange(32, 8192)
        self.imgsz.setSingleStep(32)
        self.imgsz.setValue(1024)
        self.device = AnimatedComboBox()
        self.device.setEditable(True)
        self.device.addItems(["auto", "0", "1", "cpu"])
        self.device.setCurrentText("1")
        self.device.setToolTip("GPU 从 0 编号，1 是第二张显卡。auto 自动选择设备")
        parameters = QtWidgets.QGridLayout()
        parameters.setHorizontalSpacing(14)
        parameters.setVerticalSpacing(5)
        fields = (("轮次", self.epochs), ("图像尺寸", self.imgsz), ("批量", self.batch), ("设备", self.device))
        for column, (text, widget) in enumerate(fields):
            label = QtWidgets.QLabel(text)
            label.setObjectName("fieldLabel")
            label.setBuddy(widget)
            parameters.addWidget(label, 0, column)
            parameters.addWidget(widget, 1, column)
            parameters.setColumnStretch(column, 1)
        self.advanced_button = QtWidgets.QPushButton("高级设置")
        self.advanced_button.clicked.connect(self.open_advanced_settings)
        parameters.addWidget(self.advanced_button, 1, len(fields))
        config_layout.addLayout(parameters)
        self._build_advanced_settings()
        self.source.textChanged.connect(self.source_changed)
        right.addWidget(self.configuration)
        monitor = QtWidgets.QFrame()
        monitor.setProperty("card", True)
        monitor_layout = QtWidgets.QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(20, 16, 20, 16)
        monitor_layout.setSpacing(10)
        status = QtWidgets.QHBoxLayout()
        self.state = QtWidgets.QLabel("就绪")
        self.state.setObjectName("trainingState")
        self.state.setWordWrap(True)
        status.addWidget(self.state, 1)
        self.elapsed = QtWidgets.QLabel("耗时 —  /  剩余 —")
        self.elapsed.setObjectName("trainingElapsed")
        status.addWidget(self.elapsed)
        monitor_layout.addLayout(status)
        self.epoch_progress = QtWidgets.QProgressBar()
        self.epoch_progress.setObjectName("epochProgress")
        self.epoch_progress.setFormat("轮次 —")
        monitor_layout.addWidget(self.epoch_progress)
        right.addWidget(monitor)
        tabs = QtWidgets.QTabWidget()
        charts = QtWidgets.QWidget()
        charts.setObjectName("trainingCharts")
        chart_layout = QtWidgets.QHBoxLayout(charts)
        chart_layout.setContentsMargins(12, 12, 12, 12)
        chart_layout.setSpacing(12)
        self.loss_chart = MetricChart("损失曲线", loss=True)
        self.metric_chart = MetricChart("验证指标")
        chart_layout.addWidget(self.loss_chart, 1)
        chart_layout.addWidget(self.metric_chart, 1)
        tabs.addTab(charts, "实时曲线")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.log.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.log.setMaximumBlockCount(3000)
        tabs.addTab(self.log, "实时日志")
        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        tabs.addTab(self.results, "结果与权重")
        right.addWidget(tabs, 1)
        actions = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("开始训练")
        self.start_button.setObjectName("startTraining")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QtWidgets.QPushButton("停止训练")
        self.stop_button.setObjectName("stopTraining")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.open_button = QtWidgets.QPushButton("打开目录")
        self.open_button.clicked.connect(self.open_directory)
        self.export_button = QtWidgets.QPushButton("格式转换")
        self.export_dialog = OnnxConversionDialog(self)
        self.export_dialog.local.clicked.connect(self.export_local_onnx)
        self.export_dialog.current.clicked.connect(self.export_onnx)
        self.export_dialog.stop_requested.connect(self.stop_training)
        self.export_button.clicked.connect(self.open_export_dialog)
        for button in (self.start_button, self.stop_button):
            actions.addWidget(button)
        actions.addStretch()
        for button in (self.open_button, self.export_button):
            actions.addWidget(button)
        right.addLayout(actions)

    def source_changed(self, text):
        self.yaml_path.clear()
        self.last_automatic_split = None

    def _build_advanced_settings(self):
        dialog = self.advanced_dialog = QtWidgets.QDialog(self)
        dialog.setObjectName("advancedTrainingSettings")
        dialog.setWindowTitle("高级设置")
        form = QtWidgets.QVBoxLayout(dialog)
        form.setContentsMargins(22, 20, 22, 20)
        form.setSpacing(14)
        self.advanced_numbers = []

        def number(default, minimum, maximum, decimals=0):
            field = QtWidgets.QDoubleSpinBox() if decimals else QtWidgets.QSpinBox()
            if decimals:
                field.setDecimals(decimals)
            field.setRange(minimum, maximum)
            field.setValue(default)
            field.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            field.setKeyboardTracking(False)
            self.advanced_numbers.append(field)
            return field

        self.workers = number(8, 0, 32)
        self.cache = QtWidgets.QLineEdit("ram")
        self.optimizer = QtWidgets.QLineEdit("AdamW")
        self.initial_lr = number(0.001, 0.000001, 1.0, 6)
        self.final_lr_ratio = number(0.01, 0.0, 1.0, 4)
        self.warmup_epochs = number(3.0, 0.0, 100.0, 1)
        self.patience = number(30, 0, 100000)
        self.close_mosaic = number(20, 0, 100000)
        self.mosaic = number(1.0, 0.0, 1.0, 2)
        self.scale = number(0.3, 0.0, 1.0, 2)
        self.fliplr = number(0.5, 0.0, 1.0, 2)
        self.hsv_h = number(0.01, 0.0, 1.0, 3)
        self.hsv_s = number(0.3, 0.0, 1.0, 2)
        self.hsv_v = number(0.3, 0.0, 1.0, 2)
        self.erasing = number(0.1, 0.0, 0.9, 2)
        self.amp = QtWidgets.QCheckBox("启用")
        self.amp.setChecked(True)
        self.deterministic = QtWidgets.QCheckBox("启用")
        self.seed = number(42, 0, 2147483647)
        self.optimizer.textChanged.connect(
            lambda name: self.initial_lr.setEnabled(name.strip().casefold() != "auto"))
        groups = (
            (
                ("数据加载", (
                    ("加载进程", self.workers, "常用 0–8，填 0 用主进程"),
                    ("图像缓存", self.cache, "填 ram、disk 或 false（关闭）"),
                )),
                ("迁移训练", (
                    ("优化器", self.optimizer, "填 auto、AdamW、SGD、Adam 或 RMSProp"),
                    ("初始学习率", self.initial_lr, "微调常用 0.0001–0.001，auto 时自动设置"),
                    ("末期学习率比例", self.final_lr_ratio, "填 0–1，常用 0.01–0.1"),
                    ("预热轮数", self.warmup_epochs, "常用 0–5，填 0 关闭预热"),
                )),
                ("训练控制", (
                    ("早停耐心轮数", self.patience, "常用 10–50，填 0 关闭早停"),
                    ("关闭 Mosaic", self.close_mosaic, "最后几轮关闭拼图，常用 10–20，0 则全程按设定概率增强"),
                )),
            ),
            (
                ("数据增强", (
                    ("Mosaic 概率", self.mosaic, "填 0–1，填 0 关闭拼图"),
                    ("缩放幅度", self.scale, "填 0–1，0.3 表示约 ±30%"),
                    ("水平翻转概率", self.fliplr, "填 0–1，0.5 表示 50%"),
                    ("色调变化", self.hsv_h, "填 0–1，常用 0.01–0.02"),
                    ("饱和度变化", self.hsv_s, "填 0–1，常用 0.2–0.7"),
                    ("亮度变化", self.hsv_v, "填 0–1，常用 0.2–0.4"),
                    ("随机擦除概率", self.erasing, "填 0–0.9，仅图像分类训练生效"),
                )),
                ("性能设置", (
                    ("混合精度（AMP）", self.amp, "GPU 训练时可节省显存"),
                    ("确定性训练", self.deterministic, "启用后优先使用可重复的运算"),
                    ("随机种子", self.seed, "非负整数，例如 42，用于训练和新划分"),
                )),
            ),
        )
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(14)
        for sections in groups:
            column = QtWidgets.QVBoxLayout()
            column.setSpacing(12)
            for title, rows in sections:
                column.addWidget(self._advanced_group(title, rows))
            column.addStretch()
            columns.addLayout(column, 1)
        form.addLayout(columns)

        python_row = QtWidgets.QHBoxLayout()
        python_label = QtWidgets.QLabel("Python 环境")
        python_label.setBuddy(self.python)
        python_row.addWidget(python_label)
        python_row.addLayout(self.path_row(self.python, "python"), 1)
        form.addLayout(python_row)
        self.advanced_error = QtWidgets.QLabel()
        self.advanced_error.setWordWrap(True)
        self.advanced_error.hide()
        form.addWidget(self.advanced_error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("确定")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept_advanced_settings)
        buttons.rejected.connect(dialog.reject)
        form.addWidget(buttons)
        dialog.resize(960, form.sizeHint().height())

    def _advanced_group(self, title, rows):
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("trainingParameterGroup")
        grid = QtWidgets.QGridLayout(group)
        grid.setContentsMargins(14, 15, 14, 12)
        grid.setHorizontalSpacing(9)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(2, 1)
        for row, (name, field, hint) in enumerate(rows):
            label = QtWidgets.QLabel(name)
            label.setBuddy(field)
            field.setFixedWidth(103)
            field.setToolTip(hint)
            detail = QtWidgets.QLabel(hint)
            detail.setObjectName("fieldLabel")
            detail.setWordWrap(True)
            grid.addWidget(label, row, 0)
            grid.addWidget(field, row, 1)
            grid.addWidget(detail, row, 2)
        return group

    def open_advanced_settings(self):
        if self.is_busy():
            return
        text_fields = (self.python, self.optimizer, self.cache)
        toggles = (self.amp, self.deterministic)
        original_text = [field.text() for field in text_fields]
        original_toggles = [field.isChecked() for field in toggles]
        original_numbers = [field.value() for field in self.advanced_numbers]
        self.advanced_error.hide()
        if self.advanced_dialog.exec() == QtWidgets.QDialog.DialogCode.Rejected:
            for field, value in zip(text_fields, original_text):
                field.setText(value)
            for field, value in zip(toggles, original_toggles):
                field.setChecked(value)
            for field, value in zip(self.advanced_numbers, original_numbers):
                field.setValue(value)

    def accept_advanced_settings(self):
        optimizer = OPTIMIZERS.get(self.optimizer.text().strip().casefold())
        if optimizer is None:
            self.advanced_error.setText("请输入受支持的优化器名称。")
            self.advanced_error.show()
            self.optimizer.setFocus()
            return
        cache = self.cache.text().strip().casefold()
        if cache not in CACHE_MODES:
            self.advanced_error.setText("图像缓存请填 ram、disk 或 false。")
            self.advanced_error.show()
            self.cache.setFocus()
            return
        python = self.python.text().strip()
        if not python or not Path(python).expanduser().is_file():
            self.advanced_error.setText("请选择有效的 Python 可执行文件。")
            self.advanced_error.show()
            self.python.setFocus()
            return
        self.optimizer.setText(optimizer)
        self.cache.setText(cache)
        self.advanced_dialog.accept()

    def path_row(self, edit, kind):
        row = QtWidgets.QHBoxLayout()
        row.addWidget(edit)
        button = QtWidgets.QPushButton("浏览")
        button.clicked.connect(lambda: self.browse(edit, kind))
        row.addWidget(button)
        return row

    def browse(self, edit, kind):
        if kind in ("dataset", "directory"):
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        else:
            location = edit.text()
            if kind == "yaml" and not location.strip():
                location = self.output.text().strip()
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择文件", location, "YAML (*.yaml *.yml)" if kind == "yaml" else "Python (*)")
        if path:
            edit.setText(path)
            if edit is self.output:
                self.load_history()

    def choose_weight(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择检测权重", "", "权重 (*.pt)")
        if path:
            self.model.setCurrentText(path)

    def model_family_changed(self):
        self.model.clear()
        self.model.addItems(["yolo11n.pt", "yolo11s.pt", "yolov8n.pt", "yolov8s.pt"]
                            if self.family.currentIndex() == 0 else ["rtdetr-l.pt", "rtdetr-x.pt"])

    def is_busy(self):
        return (self.manager.is_training or self.export_manager.is_exporting or self.export_worker_running()
                or any(thread and thread.isRunning() for thread in (self.preparation, self.yaml_preparation)))

    def export_worker_running(self):
        thread = self.export_manager.export_thread
        return bool(thread and thread.is_alive())

    def generate_yaml(self):
        if self.is_busy():
            return
        if not self.can_start():
            self.state.setText("请先结束数据集处理任务")
            return
        source_text = self.source.text().strip()
        output_text = self.output.text().strip()
        if not output_text:
            self.state.setText("请先选择保存位置")
            return
        source = Path(source_text).expanduser().resolve()
        if not source_text or not source.is_dir():
            self.state.setText("请先选择数据集目录")
            return
        try:
            output = Path(output_text).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            self.settings.setValue("training/output", str(output))
            target = output / f"data_splits_{uuid.uuid4().hex[:8]}"
            worker = DatasetPreparation(
                source, target, self.ratio.value() / 100, self,
                yaml_path=output / "data.yaml", seed=self.seed.value(),
                copy_existing_splits=not source.is_relative_to(output),
            )
            self._start_preparation("yaml_preparation", worker, self.yaml_generated, self.state.setText,
                                    "正在生成 YAML…")
        except Exception as error:
            self.state.setText(str(error))
            self.set_busy(False)

    def yaml_generated(self, summary):
        if self.yaml_preparation.isInterruptionRequested():
            self.state.setText("已取消")
            return
        self.yaml_path.setText(summary["data"])
        self.state.setText(f"YAML 已生成 · {summary['train']} 训练 / {summary['val']} 验证")

    def _start_preparation(self, slot, worker, ready, failed, message):
        setattr(self, slot, worker)
        worker.prepared.connect(ready)
        worker.failed.connect(failed)
        worker.finished.connect(lambda: self._retire_preparation(slot, worker))
        self.set_busy(True)
        self.state.setText(message)
        worker.start()

    def _retire_preparation(self, slot, worker):
        if getattr(self, slot) is worker:
            setattr(self, slot, None)
        worker.deleteLater()
        self.set_busy(self.is_busy())

    def set_busy(self, busy):
        exporting = self.export_worker_running() or self.export_manager.is_exporting
        self.configuration.setEnabled(not busy)
        self.advanced_button.setEnabled(not busy)
        self.history.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.stop_button.setText("停止转换" if exporting else "停止训练")
        self.stop_button.setEnabled(busy and (not exporting or self.export_manager.is_exporting))
        self.export_button.setEnabled(not busy or exporting)
        self._update_export_controls(busy)
        self.busy_changed.emit(busy)

    def set_source(self, path):
        if not path or self.is_busy() or self.yaml_path.text().strip():
            return
        current = self.source.text().strip()
        if not current or current == self.last_dock_source:
            self.last_dock_source = str(path)
            self.source.setText(self.last_dock_source)

    def start_training(self):
        if self.is_busy():
            return
        if not self.can_start():
            self.state.setText("请先结束数据集处理、模型操作或自动标注任务")
            return
        output_text = self.output.text().strip()
        if not output_text:
            self.state.setText("请先选择保存位置")
            return
        source_text = self.yaml_path.text().strip() or self.source.text().strip()
        source = Path(source_text).expanduser()
        python = Path(self.python.text().strip()).expanduser()
        if not source_text or not source.exists() or not python.is_file():
            self.state.setText("请选择有效数据集；环境路径可在高级设置中修改")
            return
        if self.imgsz.value() % 32:
            self.state.setText("图片尺寸须为 32 的整数倍")
            return
        if self.family.currentIndex() == 1 and self.imgsz.value() < 256:
            self.state.setText("RT-DETR 图像尺寸至少为 256")
            return
        try:
            output = Path(output_text).expanduser().resolve()
            directory = create_run_directory(output)
            self.settings.setValue("training/output", str(output))
            device = self.device.currentText().strip()
            args = {"model_family": "yolo" if self.family.currentIndex() == 0 else "rtdetr",
                    "model": self.model.currentText().strip(), "epochs": self.epochs.value(),
                    "batch": self.batch.value(), "imgsz": self.imgsz.value(),
                    "device": int(device) if device.isdecimal() else device,
                    "workers": self.workers.value(), "cache": CACHE_MODES[self.cache.text()],
                    "optimizer": self.optimizer.text(),
                    "lrf": self.final_lr_ratio.value(), "warmup_epochs": self.warmup_epochs.value(),
                    "patience": self.patience.value(), "close_mosaic": self.close_mosaic.value(),
                    "mosaic": self.mosaic.value(), "scale": self.scale.value(), "fliplr": self.fliplr.value(),
                    "hsv_h": self.hsv_h.value(), "hsv_s": self.hsv_s.value(), "hsv_v": self.hsv_v.value(),
                    "erasing": self.erasing.value(), "amp": self.amp.isChecked(),
                    "deterministic": self.deterministic.isChecked(), "seed": self.seed.value(),
                    "project": str(directory), "name": "train", "save": True, "plots": True}
            if args["optimizer"] != "auto":
                args["lr0"] = self.initial_lr.value()
            self.record = {"directory": str(directory), "source": str(source.resolve()),
                           "python": str(python.resolve()), "args": args, "status": "准备数据",
                           "base_model": args["model"]}
            self.history.clearSelection()
            self.history.setCurrentRow(-1)
            self.metrics = []
            self.log.clear()
            self.log_progress = ""
            self.results.clear()
            self.elapsed.setText("耗时 —  /  剩余 —")
            self.epoch_progress.setValue(0)
            self.render_metrics(reset=True)
            self.persist()
            split_manifest = None
            if self.last_automatic_split and self.last_automatic_split[0] == source.resolve():
                split_manifest = self.last_automatic_split[1]
            worker = DatasetPreparation(
                str(source), directory / "dataset", self.ratio.value() / 100, self,
                split_manifest=split_manifest, reference_splits=True, seed=self.seed.value(),
            )
            self._start_preparation(
                "preparation", worker, self.dataset_ready,
                lambda error: self.on_event("training_error", {"error": error}),
                "准备数据 · 自动生成 YAML…",
            )
        except Exception as error:
            self.on_event("training_error", {"error": str(error)})

    def dataset_ready(self, summary):
        if self.preparation.isInterruptionRequested():
            self.on_event("training_stopped", {})
            return
        self.record["args"]["data"] = summary["data"]
        self.record["names"] = summary["names"]
        self.record["sample_manifest"] = str(summary["manifest"])
        if summary["automatic_split_manifest"]:
            self.last_automatic_split = (Path(self.record["source"]), summary["automatic_split_manifest"])
        self.append_log(f"训练集 {summary['train']} 张，验证集 {summary['val']} 张；类别：{summary['names']}")
        if summary["warning"]:
            self.append_log(summary["warning"])
        self.launch()

    def launch(self):
        self.started_at = time.monotonic()
        self.record["status"] = "启动训练"
        self.persist()
        self.set_busy(True)
        self.state.setText("加载模型…")
        self.manager.configure_environment(self.record["python"], auto_install_packages=False)
        ok, message = self.manager.start_training(dict(self.record["args"]))
        if not ok:
            self.on_event("training_error", {"error": message})

    def stop_training(self):
        if self.export_manager.is_exporting or self.export_worker_running():
            self.export_manager.stop_export()
            self.stop_button.setEnabled(False)
            return
        if self.yaml_preparation and self.yaml_preparation.isRunning():
            self.yaml_preparation.requestInterruption()
        if self.preparation and self.preparation.isRunning():
            self.preparation.requestInterruption()
        self.manager.stop_training()
        self.stop_button.setEnabled(False)
        self.state.setText("正在停止…")

    def on_event(self, event, data):
        if event.endswith("_log"):
            progress = data.get("progress")
            if progress == "update":
                self.update_log_progress(data["message"])
            elif progress == "finish":
                self.finish_log_progress()
            else:
                self.append_log(data.get("message", ""))
        elif event == "training_started":
            self.state.setText("准备训练…")
        elif event == "training_phase":
            self.state.setText(f"验证中 · 第 {self.epoch_progress.value() + 1} 轮")
        elif event == "training_progress":
            epoch, total = data["epoch"], data["total_epochs"]
            self.epoch_progress.setRange(0, total)
            self.epoch_progress.setValue(epoch - 1)
            self.epoch_progress.setFormat(f"已完成 {epoch - 1} / {total} 轮")
            self.state.setText(f"训练中 · {epoch} / {total} 轮")
            memory = data.get("gpu_reserved_gb")
            self.state.setText(self.state.text() + f" · {data.get('device', '')}"
                               + (f" · 显存 {memory:.2f} GB" if memory is not None else ""))
        elif event == "training_metrics":
            epoch = data["epoch"]
            total = data.get("total_epochs", self.epoch_progress.maximum())
            self.epoch_progress.setRange(0, total)
            self.epoch_progress.setValue(epoch)
            self.epoch_progress.setFormat(f"已完成 {epoch} / {total} 轮")
            self.state.setText(self.epoch_progress.format())
            self.metrics = [(e, m) for e, m in self.metrics if e != epoch]
            values = {k: float(v) for k, v in data["metrics"].items() if math.isfinite(float(v))}
            self.metrics.append((epoch, values))
            columns = (
                ("metrics/precision(B)", "Box(P)"), ("metrics/recall(B)", "R"),
                ("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
            )
            summary = "  ".join(f"{name} {values[key]:.4f}" for key, name in columns if key in values)
            if summary:
                self.append_log(f"Epoch {epoch}/{total}  验证 all  {summary}")
            self.render_metrics(animate=True)
            self.persist()
        elif event in ("training_completed", "training_error", "training_stopped", "environment_error"):
            status = {"training_completed": "训练完成", "training_stopped": "已停止"}.get(event, "训练失败")
            message = data.get("error", "")
            self.state.setText(f"{status} {message}")
            self.append_log(f"{status} {message}")
            if self.record:
                self.record["status"] = status
                self.record["error"] = message
                self.persist()
                self.show_results()
            self.started_at = None
            self.set_busy(self.is_busy())
            self.load_history()

    def update_log_progress(self, message):
        line = ANSI_ESCAPE.sub("", message).rstrip()
        if not line.strip() or line == self.log_progress:
            return
        if self.log_progress:
            cursor = QtGui.QTextCursor(self.log.document().lastBlock())
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfBlock,
                                QtGui.QTextCursor.MoveMode.MoveAnchor)
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock,
                                QtGui.QTextCursor.MoveMode.KeepAnchor)
            cursor.insertText(line)
        else:
            self.log.appendPlainText(line)
        self.log_progress = line

    def finish_log_progress(self):
        if self.log_progress:
            self.write_log(self.log_progress)
            self.log_progress = ""

    def append_log(self, message, *, persist=True):
        self.finish_log_progress()
        lines = []
        previous = self.log.document().lastBlock().text()
        for line in ANSI_ESCAPE.sub("", message).splitlines():
            line = line.rstrip()
            if line.strip() and line != previous:
                lines.append(line)
                previous = line
        if not lines:
            return
        message = "\n".join(lines)
        self.log.appendPlainText(message)
        if persist:
            self.write_log(message)

    def write_log(self, message):
        if self.record:
            try:
                with (Path(self.record["directory"]) / "training.log").open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
            except OSError:
                pass

    def persist(self):
        if self.record:
            self.record["metrics"] = self.metrics
            atomic_write_text(Path(self.record["directory"]) / "task.json",
                              json.dumps(self.record, ensure_ascii=False, indent=2))

    def render_metrics(self, *, animate=False, reset=False):
        for chart in (self.loss_chart, self.metric_chart):
            chart.set_metrics(self.metrics, animate=animate, reset=reset)

    def update_elapsed(self):
        if self.started_at:
            elapsed = time.monotonic() - self.started_at
            progress = self.epoch_progress.value() / max(1, self.epoch_progress.maximum())
            remaining = f"{elapsed * (1 - progress) / progress / 60:.1f} 分钟" if progress > 0 else "估算中"
            self.elapsed.setText(f"耗时 {elapsed / 60:.1f} 分钟  /  剩余 ≈ {remaining}")
        if not self.is_busy() and not self.start_button.isEnabled():
            self.set_busy(False)

    def load_history(self):
        blocker = QtCore.QSignalBlocker(self.history)
        self.history.clear()
        active_directory = Path(self.record.get("directory", "")) if self.record else None
        for created, path, title, status, source_path in training_history(Path(self.output.text()).expanduser()):
            item = QtWidgets.QListWidgetItem(f"{title}\n{created:%m-%d %H:%M} · {status}")
            item.setToolTip(f"{title}\n{created:%Y-%m-%d %H:%M} · {status}\n数据集：{source_path}\n任务：{path.parent}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
            self.history.addItem(item)
            if path.parent == active_directory:
                self.history.setCurrentItem(item)
        del blocker

    def select_history(self, item):
        if item is None or self.is_busy():
            return
        try:
            path = Path(item.data(QtCore.Qt.ItemDataRole.UserRole))
            self.record = read_training_record(path)
            self.metrics = self.record.get("metrics", [])
            self.render_metrics(animate=True, reset=True)
            self.state.setText(self.record["status"] + " · 历史任务")
            self.show_results()
            self.set_busy(False)
        except (OSError, ValueError) as error:
            self.state.setText(str(error))

    def show_results(self):
        directory = Path(self.record["directory"])
        weights = training_weights(directory)
        lines = [f"状态：{self.record['status']}", f"目录：{directory}"]
        if self.record.get("args", {}).get("data"):
            lines.append(f"YAML：{self.record['args']['data']}")
        if self.record.get("sample_manifest"):
            lines.append(f"样本清单：{self.record['sample_manifest']}")
        for name in ("best.pt", "last.pt"):
            path = weights / name
            lines.append(f"{name}：{path if path.is_file() else '尚未生成'}")
        if self.record.get("onnx"):
            lines.append(f"ONNX：{self.record['onnx']}")
        if self.metrics:
            lines += [f"{key}: {value:.5g}" for key, value in self.metrics[-1][1].items()]
        self.results.setPlainText("\n".join(lines))

    def open_directory(self):
        if self.record:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.record["directory"]))

    def best_checkpoint(self):
        if self.record and self.record.get("directory"):
            path = training_weights(self.record["directory"]) / "best.pt"
            if path.is_file():
                return path
        return None

    def _update_export_controls(self, busy):
        self.export_dialog.set_controls(busy=busy, has_checkpoint=self.best_checkpoint() is not None)

    def open_export_dialog(self):
        self._update_export_controls(self.is_busy())
        self.export_dialog.open()

    def export_onnx(self):
        if not self._can_export():
            return
        checkpoint = self.best_checkpoint()
        if checkpoint is None:
            self._show_export_error("该任务尚未生成 best.pt")
            return
        self._export_checkpoint(checkpoint, self.record)

    def export_local_onnx(self):
        if not self._can_export():
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.export_dialog, "选择本地 PT 权重", self.output.text().strip(), "PyTorch 权重 (*.pt)"
        )
        if not filename:
            return
        checkpoint = Path(filename)
        if checkpoint.suffix.lower() != ".pt" or not checkpoint.is_file():
            self._show_export_error("请选择有效的 .pt 权重文件")
            return
        self._export_checkpoint(checkpoint)

    def _can_export(self):
        if self.is_busy():
            return False
        if not self.can_start():
            self._show_export_error("请先结束正在运行的数据集或模型任务")
            return False
        return True

    def _show_export_error(self, message):
        self.export_dialog.finish_conversion("转换失败")
        self.state.setText(message)
        if self.export_dialog.isVisible():
            QtWidgets.QMessageBox.warning(self.export_dialog, "格式转换", message)

    def _export_checkpoint(self, checkpoint, record=None):
        python = (record or {}).get("python") or self.python.text().strip()
        if not python or not Path(python).expanduser().is_file():
            self._show_export_error("找不到用于转换的 Python 环境")
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.export_dialog, "导出 ONNX 模型", str(checkpoint.with_suffix(".onnx")), "ONNX 模型 (*.onnx)"
        )
        if not filename:
            return
        target = Path(filename)
        if target.suffix.lower() != ".onnx":
            target = target.with_suffix(".onnx")
            if target.exists() and QtWidgets.QMessageBox.question(
                self.export_dialog, "替换 ONNX", f"是否替换 {target}？"
            ) != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self.export_manager.configure_environment(str(Path(python).expanduser()), auto_install_packages=False)
        args = (record or {}).get("args", {})
        self.export_record = record
        ok, message = self.export_manager.start_checkpoint_export(
            str(checkpoint.resolve()), destination=str(target.absolute()),
            settings={"model_family": args.get("model_family"), "imgsz": args.get("imgsz")},
        )
        if not ok:
            self.export_record = None
            self._show_export_error(message)
            return
        self.export_dialog.start_conversion()
        self.state.setText("正在转换为 ONNX…")
        self.set_busy(True)

    def on_export_event(self, event, data):
        record = self.export_record
        persist = record is not None and record is self.record
        if event == "export_log":
            if data.get("message"):
                self.append_log(data["message"], persist=persist)
            return
        if event == "export_completed":
            path = data["exported_path"]
            if persist:
                record["onnx"] = path
                self.persist()
                self.show_results()
            self.export_dialog.finish_conversion("转换完成", completed=True)
            message = f"ONNX 已导出：{path}"
        elif event == "export_error":
            message = f"转换失败：{data['error']}"
        elif event == "export_stopped":
            self.export_dialog.finish_conversion("转换已停止")
            message = "格式转换已停止"
        else:
            return
        self.export_record = None
        self.state.setText(message)
        self.append_log(message, persist=persist)
        self.set_busy(self.is_busy())
        self.stop_button.setEnabled(False)
        if event == "export_error":
            self._show_export_error(message)

    def closeEvent(self, event):
        # Closing this window only hides it; the application owns task lifetime.
        event.ignore()
        self.hide()
