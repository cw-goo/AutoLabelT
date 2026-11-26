import base64
import json
import os.path as osp
from PIL import Image

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
    QProgressDialog,
    QDialog,
    QLabel,
    QLineEdit,
    QSpinBox,
    QPushButton,
    QDialogButtonBox,
    QApplication,
)

from autolabel.workbench.app_info import __version__
from autolabel.workbench.views.labeling.annotation_io import (
    AnnotationTransaction, atomic_write_json, describe_file_error,
)
from autolabel.workbench.views.labeling.label_schema import palette_names, set_label_palette
from autolabel.workbench.views.labeling.utils.theme import get_theme
from autolabel.workbench.services.auto_labeling import (
    _BATCH_PROCESSING_AUTO_GRID_MODELS,
    _BATCH_PROCESSING_INVALID_MODELS,
    _BATCH_PROCESSING_TEXT_PROMPT_MODELS,
    _BATCH_PROCESSING_VIDEO_MODELS,
    _SKIP_DET_MODELS,
)
from autolabel.workbench.views.labeling.logger import logger
from autolabel.workbench.views.labeling.schema import IMAGE_TAGS_FIELD
from autolabel.workbench.views.labeling.shape import Shape
from autolabel.workbench.views.labeling.utils._io import io_open
from autolabel.workbench.views.labeling.utils.image_tags import normalize_image_tags
from autolabel.workbench.views.labeling.utils.qt import new_icon_path
from autolabel.workbench.views.labeling.utils.style import get_dialog_style
from autolabel.workbench.views.labeling.utils.supplement import (
    read_yolo_baseline, supplementary_shapes, yolo_label_path,
)
from autolabel.workbench.views.labeling.widgets.popup import Popup

__all__ = ["run_all_images"]


class BatchRangeDialog(QDialog):
    def __init__(self, image_count, start_index=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Auto Run"))
        self.setMinimumWidth(440)
        theme = get_theme()
        self.setStyleSheet(get_dialog_style() + f"""
            QLabel#rangeTitle {{ font-size: 18px; font-weight: 600; }}
            QLabel#rangeHint, QLabel#rangeSummary {{
                color: {theme["text_secondary"]};
                font-size: 12px;
            }}
            QSpinBox {{
                padding: 6px 12px;
                font-size: 14px;
            }}
            QPushButton {{ min-width: 60px; }}
            QPushButton#runButton {{
                background-color: {theme["primary"]};
                color: white;
                border-color: {theme["primary"]};
            }}
            QPushButton#runButton:hover {{
                background-color: {theme["primary_hover"]};
            }}
            QPushButton#runButton:pressed {{
                background-color: {theme["primary_pressed"]};
            }}
            """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)
        title = QLabel(self.tr("Image range"))
        title.setObjectName("rangeTitle")
        layout.addWidget(title)
        layout.addSpacing(6)
        hint = QLabel(self.tr("Select the first and last images to process."))
        hint.setObjectName("rangeHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addSpacing(20)

        range_layout = QGridLayout()
        range_layout.setHorizontalSpacing(16)
        range_layout.setVerticalSpacing(8)
        range_layout.setColumnStretch(0, 1)
        range_layout.setColumnStretch(1, 1)
        self.from_input = QSpinBox()
        self.from_input.setRange(1, image_count)
        self.from_input.setValue(start_index)
        self.to_input = QSpinBox()
        self.to_input.setRange(start_index, image_count)
        self.to_input.setValue(image_count)
        self.from_input.valueChanged.connect(self.to_input.setMinimum)
        for column, (text, field) in enumerate(
            (
                (self.tr("From"), self.from_input),
                (self.tr("To"), self.to_input),
            )
        ):
            label = QLabel(text)
            label.setBuddy(field)
            field.setMinimumHeight(40)
            field.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            range_layout.addWidget(label, 0, column)
            range_layout.addWidget(field, 1, column)
        range_layout.setRowMinimumHeight(2, 8)
        summary = QLabel()
        summary.setObjectName("rangeSummary")

        def update_summary():
            selected = self.to_input.value() - self.from_input.value() + 1
            summary.setText(
                self.tr("%s of %s images selected") % (selected, image_count)
            )

        self.from_input.valueChanged.connect(update_summary)
        self.to_input.valueChanged.connect(update_summary)
        update_summary()
        range_layout.addWidget(
            summary,
            3,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        cancel_button = QPushButton(self.tr("Cancel"))
        cancel_button.clicked.connect(self.reject)
        buttons.addWidget(cancel_button, 1)
        run_button = QPushButton(self.tr("Run"))
        run_button.setObjectName("runButton")
        run_button.setDefault(True)
        run_button.clicked.connect(self.accept)
        buttons.addWidget(run_button, 1)
        range_layout.addLayout(buttons, 5, 1)
        options = QGridLayout()
        options.setVerticalSpacing(8)
        self.annotation_mode = QtWidgets.QComboBox()
        self.annotation_mode.addItem("按模型设置", "model")
        self.annotation_mode.addItem("清除旧标注后重新推理", "clear")
        self.annotation_mode.addItem("漏标补充", "supplement")
        mode_label = QLabel("标注模式")
        mode_label.setBuddy(self.annotation_mode)
        options.addWidget(mode_label, 0, 0)
        options.addWidget(self.annotation_mode, 0, 1)
        self.supplement_iou = QtWidgets.QDoubleSpinBox()
        self.supplement_iou.setRange(0.05, 1.0)
        self.supplement_iou.setSingleStep(0.05)
        self.supplement_iou.setValue(0.60)
        self.supplement_iou.setToolTip(
            "新旧框的交并比达到此值时，保留原框。阈值越低，跳过的重叠框越多。"
        )
        iou_label = QLabel("重复框判定（IoU）")
        iou_label.setBuddy(self.supplement_iou)
        options.addWidget(iou_label, 1, 0)
        options.addWidget(self.supplement_iou, 1, 1)

        def show_threshold():
            visible = self.annotation_mode.currentData() == "supplement"
            iou_label.setVisible(visible)
            self.supplement_iou.setVisible(visible)

        self.annotation_mode.currentIndexChanged.connect(show_threshold)
        show_threshold()
        range_layout.addLayout(options, 4, 0, 1, 2)
        layout.addLayout(range_layout)


class TextInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.tr("Enter Text Prompt"))
        self.setFixedSize(400, 180)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.MSWindowsFixedSizeDialogHint
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        prompt_label = QLabel(self.tr("Please enter your text prompt:"))
        prompt_label.setStyleSheet(
            f"font-size: 13px; color: {get_theme()['text']}; font-weight: 500;"
        )
        layout.addWidget(prompt_label)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText(self.tr("Enter prompt here..."))
        layout.addWidget(self.text_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        t = get_theme()
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {t["background"]};
                border-radius: 10px;
            }}

            QLineEdit {{
                border: 1px solid {t["border"]};
                border-radius: 8px;
                background-color: {t["background_secondary"]};
                font-size: 13px;
                height: 36px;
                padding: 0 12px;
                color: {t["text"]};
            }}

            QLineEdit:hover {{
                background-color: {t["background_hover"]};
            }}

            QLineEdit:focus {{
                border: 2px solid {t["highlight"]};
                background-color: {t["background_secondary"]};
            }}

            QPushButton {{
                min-width: 100px;
                height: 36px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
            }}

            QPushButton[text="OK"] {{
                background-color: {t["primary"]};
                color: white;
                border: none;
            }}

            QPushButton[text="OK"]:hover {{
                background-color: {t["primary_hover"]};
            }}

            QPushButton[text="OK"]:pressed {{
                background-color: {t["primary"]};
            }}

            QPushButton[text="Cancel"] {{
                background-color: {t["surface"]};
                color: {t["text"]};
                border: 1px solid {t["border"]};
            }}

            QPushButton[text="Cancel"]:hover {{
                background-color: {t["background_hover"]};
            }}

            QPushButton[text="Cancel"]:pressed {{
                background-color: {t["surface"]};
            }}
        """)

    def get_input_text(self):
        if self.exec() == QDialog.DialogCode.Accepted:
            return self.text_input.text().strip()
        return ""


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def load_existing_shapes(image_file):
    """
    Loads existing shapes from the JSON file for skip detection.

    Args:
        image_file (str): The path to the image file.

    Returns:
        list: A list of Shape objects loaded from the JSON file, or None if
              the file does not exist or contains no shapes.
    """
    label_file = osp.splitext(image_file)[0] + ".json"
    if not osp.exists(label_file):
        return None

    try:
        with io_open(label_file, "r") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            return None

        existing_shapes = []
        for shape_data in shapes:
            shape = Shape()
            shape.load_from_dict(shape_data, close=False)
            if shape.shape_type in ["rectangle", "rotation", "polygon"]:
                shape.selected = True
                existing_shapes.append(shape)

        return existing_shapes if existing_shapes else None

    except Exception as e:
        logger.warning(f"Failed to load existing shapes: {e}")
        return None


def finish_processing(self, progress_dialog):
    if not getattr(self, "_batch_processing_active", False):
        progress_dialog.close()
        return

    cancelled = getattr(self, "cancel_processing", False)
    supplement = getattr(self, "_batch_supplement_iou", None) is not None
    added, skipped = getattr(self, "_batch_supplement_counts", (0, 0))
    try:
        transaction = getattr(self, "_batch_transaction", None)
        if transaction is not None:
            if cancelled:
                transaction.rollback()
                _restore_batch_palette(self)
            else:
                transaction.commit()
                config = self.auto_labeling_widget.model_manager.loaded_model_config or {}
                if supplement:
                    set_label_palette(self, self._batch_palette + list(self._batch_supplement_labels))
                elif isinstance(config.get("classes"), (list, tuple)):
                    set_label_palette(self, config["classes"])
        target_file = self.image_list[self.current_index]
        self.import_image_folder(osp.dirname(target_file), load=False)
        target_index = self.fn_to_index[str(target_file)]
        signals_blocked = self.file_list_widget.blockSignals(True)
        try:
            self.file_list_widget.setCurrentRow(target_index)
        finally:
            self.file_list_widget.blockSignals(signals_blocked)
        self.load_file(target_file)
        QApplication.processEvents()
    except Exception as error:
        _abort_batch(self, progress_dialog, error)
        return
    finally:
        _reset_batch_processing_state(self)
        progress_dialog.close()

    if cancelled:
        message = "已取消，旧标注已恢复。"
    elif supplement:
        message = f"漏标补充完成 · 新增 {added:,} 个，跳过 {skipped:,} 个重复标注"
    else:
        message = self.tr("Processing completed successfully!")
    popup = Popup(
        message,
        self,
        icon=new_icon_path("copy-green", "svg"),
    )
    popup.show_popup(self, position="center")


def cancel_operation(self):
    self.cancel_processing = True


def _start_batch_processing(self):
    try:
        selected = self.image_list[self._batch_start_index:self._batch_end_index]
        outputs = [_batch_output_path(self, image) for image in selected]
        if len({osp.normcase(path) for path in outputs}) != len(outputs):
            raise ValueError("所选图片存在同名文件，标注输出路径冲突，请分开处理。")
        paths = set(outputs)
        if getattr(self, "_batch_clear_annotations", False):
            paths.update(osp.splitext(image)[0] + ".json" for image in selected)
        self._batch_transaction = AnnotationTransaction(sorted(paths))
        self._batch_transaction.preflight()
        # Read inputs before changing any old annotation or starting inference.
        for image in selected:
            with open(image, "rb") as stream:
                stream.read(1)
        self._batch_palette = palette_names(self)
        if getattr(self, "_batch_supplement_iou", None) is not None:
            self._batch_supplement_labels = {}
            self._batch_supplement_counts = [0, 0]
        if getattr(self, "_batch_clear_annotations", False):
            _clear_batch_annotations(self)
    except Exception as error:
        _abort_batch(self, None, error)
        return
    self._batch_processing_active = True
    try:
        show_progress_dialog_and_process(self)
    except Exception as error:
        _abort_batch(self, None, error)


def _batch_output_path(self, image):
    sidecar = osp.splitext(image)[0] + ".json"
    return osp.abspath(osp.join(self.output_dir, osp.basename(sidecar))
                       if self.output_dir else sidecar)


def _restore_batch_palette(self):
    if hasattr(self, "_batch_palette"):
        set_label_palette(self, self._batch_palette)


def _abort_batch(self, progress_dialog, error):
    transaction = getattr(self, "_batch_transaction", None)
    try:
        if transaction is not None and not transaction.finished:
            transaction.rollback()
        _restore_batch_palette(self)
        if getattr(self, "filename", None) and getattr(self, "_batch_processing_active", False):
            self.load_file(self.filename)
    except Exception as recovery_error:
        error = OSError(f"{error}\n{recovery_error}")
    finally:
        _reset_batch_processing_state(self)
        if progress_dialog is not None:
            progress_dialog.close()
    message = describe_file_error(error)
    logger.error(f"Batch annotation failed: {message}")
    popup = Popup(message, self, icon=new_icon_path("error", "svg"))
    popup.show_popup(self, position="center")


def _clear_batch_annotations(self):
    """Hide selected annotations with rollback protection before fresh inference."""
    selected = self.image_list[self._batch_start_index:self._batch_end_index]
    self._batch_transaction.clear()
    config = self.auto_labeling_widget.model_manager.loaded_model_config
    classes = config.get("classes", [])
    if isinstance(classes, (list, tuple)):
        set_label_palette(self, classes)
    else:
        set_label_palette(self, [])
    if self.filename in selected:
        self.set_clean()
        self.load_file(self.filename)


def _reset_batch_processing_state(self):
    self._batch_processing_active = False
    for attribute in (
        "text_prompt",
        "run_tracker",
        "image_index",
        "current_index",
        "_batch_start_index",
        "_batch_end_index",
        "_batch_clear_annotations",
        "_batch_supplement_iou",
        "_batch_supplement_labels",
        "_batch_supplement_counts",
        "_batch_transaction",
        "_batch_palette",
    ):
        if hasattr(self, attribute):
            delattr(self, attribute)


def _reset_auto_labeling_tracker(self):
    model_manager = self.auto_labeling_widget.model_manager
    if model_manager.loaded_model_config is None:
        return
    model_manager.set_auto_labeling_reset_tracker()


def _new_annotation(self, image_file, shapes, description=""):
    image_data = None
    if self._config["store_data"]:
        with open(image_file, "rb") as stream:
            image_data = base64.b64encode(stream.read()).decode("utf-8")
    image_width, image_height = get_image_size(image_file)
    return {
        "version": __version__,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": osp.basename(image_file),
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width,
        "description": description,
    }


def _save_supplement(self, image_file, output, predicted, threshold):
    transaction = self._batch_transaction
    transaction.watch(output)
    output_exists = osp.isfile(output)
    sidecar = osp.splitext(image_file)[0] + ".json"
    baseline = output if output_exists else sidecar
    transaction.watch(baseline)
    data = None
    if osp.isfile(baseline):
        with io_open(baseline, "r") as stream:
            data = json.load(stream)
    elif not output_exists:
        label, root = yolo_label_path(image_file)
        if label is not None:
            transaction.watch(label)
            if label.is_file():
                data = read_yolo_baseline(image_file, label, root, transaction.watch)

    original = data.get("shapes") if data is not None else []
    if not isinstance(original, list):
        raise ValueError(f"原标注的形状列表无效：{baseline}")
    added = supplementary_shapes(original, predicted, threshold)
    self._batch_supplement_counts[0] += len(added)
    self._batch_supplement_counts[1] += len(predicted) - len(added)
    self._batch_supplement_labels.update(dict.fromkeys(shape["label"] for shape in original + added))
    if not added and (output_exists or data is None):
        return
    if data is None:
        data = _new_annotation(self, image_file, added)
    elif added:
        data["shapes"] = original + added
        data["checked"] = False
    atomic_write_json(transaction.stage_path(output), data, ensure_ascii=False, indent=2)


def save_auto_labeling_result(self, image_file, auto_labeling_result):
    try:
        label_file = _batch_output_path(self, image_file)

        if auto_labeling_result is None:
            raise RuntimeError(f"模型未返回有效结果，已停止保存：{image_file}")
        else:
            new_shapes = [
                shape.to_dict() for shape in auto_labeling_result.shapes
            ]
            new_description = auto_labeling_result.description
            new_tags = getattr(auto_labeling_result, "tags", None)
            new_flags = getattr(auto_labeling_result, "flags", None)
            replace = auto_labeling_result.replace
        config = self.auto_labeling_widget.model_manager.loaded_model_config
        if config.get("type") == "autolabel_local_yolo":
            unknown = {shape["label"] for shape in new_shapes} - set(config["classes"])
            if unknown:
                raise ValueError(f"模型结果包含未定义类别 {sorted(unknown)}：{image_file}")

        threshold = getattr(self, "_batch_supplement_iou", None)
        if threshold is not None:
            _save_supplement(self, image_file, label_file, new_shapes, threshold)
            return

        if osp.exists(label_file):
            with io_open(label_file, "r") as f:
                data = json.load(f)

            if replace:
                if (
                    data["shapes"] != new_shapes
                    or data.get("description", "") != new_description
                ):
                    data["checked"] = False
                data["shapes"] = new_shapes
                data["description"] = new_description
            else:
                if new_shapes or new_description:
                    data["checked"] = False
                data["shapes"].extend(new_shapes)
                if "description" in data:
                    data["description"] += new_description
                else:
                    data["description"] = new_description
            if new_tags is not None:
                tags = normalize_image_tags(
                    new_tags, f"auto labeling result for {image_file}"
                )
                if data.get(IMAGE_TAGS_FIELD) != tags:
                    data["checked"] = False
                data[IMAGE_TAGS_FIELD] = tags
        else:
            data = _new_annotation(self, image_file, new_shapes, new_description)
            if new_tags is not None:
                data[IMAGE_TAGS_FIELD] = normalize_image_tags(
                    new_tags, f"auto labeling result for {image_file}"
                )

        if new_flags is not None:
            flags = data.setdefault("flags", {})
            if any(
                key not in flags or flags[key] != value
                for key, value in new_flags.items()
            ):
                data["checked"] = False
            flags.update(new_flags)

        transaction = getattr(self, "_batch_transaction", None)
        target = transaction.stage_path(label_file) if transaction is not None else label_file
        atomic_write_json(target, data, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to save auto labeling result for image file '{image_file}': {str(e)}"
        )
        raise


class BatchProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        app,
        image_list,
        image_index,
        model_type,
        text_prompt,
        run_tracker,
        skip_detection,
    ):
        super().__init__()
        self.app = app
        self.image_list = image_list
        self.image_index = image_index
        self.model_type = model_type
        self.text_prompt = text_prompt
        self.run_tracker = run_tracker
        self.skip_detection = skip_detection
        self.model_snapshot = app.auto_labeling_widget.model_manager.loaded_model_config

    def run(self):
        total_images = len(self.image_list)
        start_index = self.image_index
        image_count = total_images - start_index
        try:
            while (
                self.image_index < total_images
                and not self.app.cancel_processing
            ):
                image_file = self.image_list[self.image_index]
                if self.app.auto_labeling_widget.model_manager.loaded_model_config is not self.model_snapshot:
                    raise RuntimeError("推理期间模型已改变，本次任务已停止。")

                if self.text_prompt:
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=True,
                    )
                elif self.run_tracker:
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=True,
                    )
                else:
                    existing_shapes = None
                    if (
                        self.model_type in _SKIP_DET_MODELS
                        and self.skip_detection
                    ):
                        existing_shapes = load_existing_shapes(image_file)
                    result = self.app.auto_labeling_widget.model_manager.predict_shapes(
                        self.app.image,
                        image_file,
                        batch=True,
                        existing_shapes=existing_shapes,
                    )

                save_auto_labeling_result(self.app, image_file, result)
                self.image_index += 1
                completed = self.image_index - start_index
                self.progress_updated.emit(
                    completed,
                    f"Progress: {completed}/{image_count}",
                )

            self.app.image_index = self.image_index
            self.processing_finished.emit()
        except Exception as e:
            self.app.image_index = self.image_index
            self.error_occurred.emit(describe_file_error(e))
        finally:
            _reset_auto_labeling_tracker(self.app)


def process_next_image(self, progress_dialog, batch=True):
    """Process images in batch mode.

    Args:
        progress_dialog: Progress dialog widget for displaying progress.
        batch: If True, results are saved directly without updating canvas.
               If False, results trigger UI updates and canvas refresh.
               Defaults to True for batch processing mode.
    """
    model_type = self.auto_labeling_widget.model_manager.loaded_model_config[
        "type"
    ]
    model = self.auto_labeling_widget.model_manager.loaded_model_config[
        "model"
    ]
    total_images = self._batch_end_index
    image_count = total_images - self._batch_start_index
    self._progress_dialog = progress_dialog

    batch_processing_mode = "default"
    if model_type == "remote_server":
        batch_processing_mode = model.get_batch_processing_mode()

    if (
        model_type not in _BATCH_PROCESSING_VIDEO_MODELS
        and batch_processing_mode != "video"
    ):
        skip_detection = (
            getattr(self, "_batch_supplement_iou", None) is None
            and self.auto_labeling_widget.button_skip_detection.isChecked()
        )
        self._batch_thread = BatchProcessingThread(
            self,
            self.image_list[:total_images],
            self.image_index,
            model_type,
            self.text_prompt,
            self.run_tracker,
            skip_detection,
        )

        def _on_progress(value, label):
            progress_dialog.setValue(value)
            progress_dialog.setLabelText(label)

        def _on_error(msg):
            _abort_batch(self, progress_dialog, RuntimeError(msg))

        self._batch_thread.progress_updated.connect(_on_progress)
        self._batch_thread.processing_finished.connect(
            lambda: finish_processing(self, progress_dialog)
        )
        self._batch_thread.error_occurred.connect(_on_error)
        self._batch_thread.start()
        return

    try:
        while (self.image_index < total_images) and (
            not self.cancel_processing
        ):
            image_file = self.image_list[self.image_index]

            batch_processing_mode = "default"
            if model_type == "remote_server":
                batch_processing_mode = model.get_batch_processing_mode()
                if batch_processing_mode == "video":
                    model._widget = self
                    self.filename = image_file
                    self.load_file(self.filename)
                    batch = False
            elif model_type in _BATCH_PROCESSING_VIDEO_MODELS:
                self.filename = image_file
                self.load_file(self.filename)
                batch = False

            if self.text_prompt:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        text_prompt=self.text_prompt,
                        batch=batch,
                    )
                )
            elif self.run_tracker:
                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        run_tracker=self.run_tracker,
                        batch=batch,
                    )
                )
                if batch_processing_mode == "video":
                    logger.info("Video propagation completed, breaking loop")
                    self.image_index = total_images
                    break
            else:
                existing_shapes = None
                if (
                    model_type in _SKIP_DET_MODELS
                    and self.auto_labeling_widget.button_skip_detection.isChecked()
                ):
                    existing_shapes = load_existing_shapes(image_file)

                auto_labeling_result = (
                    self.auto_labeling_widget.model_manager.predict_shapes(
                        self.image,
                        image_file,
                        batch=batch,
                        existing_shapes=existing_shapes,
                    )
                )

            if batch:
                save_auto_labeling_result(
                    self, image_file, auto_labeling_result
                )
            elif not self.save_labels(_batch_output_path(self, image_file)):
                raise RuntimeError(f"无法保存模型结果：{image_file}")

            self.image_index += 1
            completed = self.image_index - self._batch_start_index
            progress_dialog.setValue(completed)
            progress_dialog.setLabelText(
                f"Progress: {completed}/{image_count}"
            )
            QApplication.processEvents()

        finish_processing(self, progress_dialog)

    except Exception as e:
        _abort_batch(self, progress_dialog, e)
    finally:
        _reset_auto_labeling_tracker(self)


def show_progress_dialog_and_process(self):
    self.cancel_processing = False
    image_count = self._batch_end_index - self._batch_start_index

    progress_dialog = QProgressDialog(
        self.tr("Processing..."),
        self.tr("Cancel"),
        0,
        image_count,
        self,
    )
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setWindowTitle(self.tr("Batch Processing"))
    progress_dialog.setMinimumWidth(400)
    progress_dialog.setMinimumHeight(150)
    progress_dialog.setAutoClose(False)
    progress_dialog.setAutoReset(False)

    progress_dialog.setValue(0)
    progress_dialog.setLabelText(f"Progress: 0/{image_count}")
    progress_bar = progress_dialog.findChild(QtWidgets.QProgressBar)

    if progress_bar:
        model_type = (
            self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "type", ""
            )
        )
        batch_processing_mode = "default"
        if model_type == "remote_server":
            model = self.auto_labeling_widget.model_manager.loaded_model_config.get(
                "model"
            )
            batch_processing_mode = model.get_batch_processing_mode()

        def update_progress(value):
            if batch_processing_mode != "video":
                progress_dialog.setLabelText(f"{value}/{image_count}")

        progress_bar.valueChanged.connect(update_progress)

    t = get_theme()
    progress_dialog.setStyleSheet(f"""
        QProgressDialog {{
            background-color: {t["background"]};
            border-radius: 12px;
            min-width: 280px;
            min-height: 120px;
            padding: 20px;
        }}
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: {t["surface"]};
            text-align: center;
            color: {t["text"]};
            font-size: 13px;
            min-height: 20px;
            max-height: 20px;
            margin: 16px 0;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {t["primary"]},
                stop:0.5 {t["highlight"]},
                stop:1 {t["primary"]});
            border-radius: 3px;
        }}
        QLabel {{
            color: {t["text"]};
            font-size: 13px;
            font-weight: 500;
            margin-bottom: 8px;
        }}
        QPushButton {{
            background-color: {t["surface"]};
            border: 1px solid {t["border"]};
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: {t["primary"]};
            min-width: 82px;
            height: 36px;
            padding: 0 16px;
            margin-top: 16px;
        }}
        QPushButton:hover {{
            background-color: {t["background_hover"]};
        }}
        QPushButton:pressed {{
            background-color: {t["surface"]};
        }}
    """)
    progress_dialog.canceled.connect(lambda: cancel_operation(self))
    progress_dialog.show()

    QTimer.singleShot(200, lambda: process_next_image(self, progress_dialog))


def run_all_images(self):
    if getattr(self, "_batch_processing_active", False):
        logger.warning("Batch processing is already running.")
        return

    if len(self.image_list) < 1:
        return

    if not self.may_continue():
        return

    config = self.auto_labeling_widget.model_manager.loaded_model_config
    if config is None:
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr("Model is not loaded. Choose a mode to continue.")
        )
        return

    model_type = config["type"]
    if model_type in _BATCH_PROCESSING_INVALID_MODELS:
        logger.warning(
            f"The model `{model_type}`"
            f" is not supported for this action."
            f" Please choose a valid model to execute."
        )
        self.auto_labeling_widget.model_manager.new_model_status.emit(
            self.tr(
                "Invalid model type, please choose a valid model_type to run."
            )
        )
        return

    current_index = self.fn_to_index[str(self.filename)]
    remote_mode = "default"
    if model_type == "remote_server":
        get_mode = getattr(config["model"], "get_batch_processing_mode", None)
        if get_mode is not None:
            remote_mode = get_mode()
        if remote_mode is None:
            self.auto_labeling_widget.model_manager.new_model_status.emit(
                self.tr("Batch processing is not supported for the current task.")
            )
            return
    response = BatchRangeDialog(
        len(self.image_list), current_index + 1, parent=self
    )
    supports_supplement = model_type not in _BATCH_PROCESSING_VIDEO_MODELS and remote_mode != "video"
    annotation_mode = response.annotation_mode
    default_mode = "clear" if model_type == "autolabel_local_yolo" else "model"
    if supports_supplement:
        sidecar = osp.splitext(self.filename)[0] + ".json"
        label, _ = yolo_label_path(self.filename)
        if (osp.isfile(_batch_output_path(self, self.filename)) or osp.isfile(sidecar)
                or (label is not None and label.is_file())):
            default_mode = "supplement"
    else:
        annotation_mode.removeItem(annotation_mode.findData("supplement"))
    annotation_mode.setCurrentIndex(annotation_mode.findData(default_mode))
    if response.exec() != QDialog.DialogCode.Accepted:
        return

    logger.info("Start running all images...")

    self.current_index = current_index
    self._batch_start_index = response.from_input.value() - 1
    self._batch_end_index = response.to_input.value()
    mode = annotation_mode.currentData()
    self._batch_clear_annotations = mode == "clear"
    self._batch_supplement_iou = response.supplement_iou.value() if mode == "supplement" else None
    self.image_index = self._batch_start_index
    self.text_prompt = ""
    self.run_tracker = False

    if model_type == "remote_server":
        if remote_mode == "video":
            self.run_tracker = True
            _start_batch_processing(self)
        elif remote_mode == "text_prompt":
            text_input_dialog = TextInputDialog(parent=self)
            self.text_prompt = text_input_dialog.get_input_text()
            if self.text_prompt:
                _start_batch_processing(self)
        else:
            _start_batch_processing(self)
    elif model_type in _BATCH_PROCESSING_AUTO_GRID_MODELS:
        self.auto_labeling_widget.model_manager.set_auto_labeling_marks(
            [{"type": "auto_grid"}]
        )
        _start_batch_processing(self)
    elif model_type in _BATCH_PROCESSING_TEXT_PROMPT_MODELS:
        text_input_dialog = TextInputDialog(parent=self)
        self.text_prompt = text_input_dialog.get_input_text()
        if self.text_prompt or model_type == "yoloe":
            _start_batch_processing(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        == "florence2"
    ):
        self.text_prompt = self.auto_labeling_widget.edit_text.text()
        _start_batch_processing(self)
    elif (
        self.auto_labeling_widget.model_manager.loaded_model_config["type"]
        in _BATCH_PROCESSING_VIDEO_MODELS
    ):
        self.run_tracker = True
        _start_batch_processing(self)
    else:
        _start_batch_processing(self)
