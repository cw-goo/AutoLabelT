"""Use a local Ultralytics detection weight in AutoLabel's image workflow."""

import cv2
from PyQt6 import QtCore

from autolabel.workbench.services.auto_labeling.model import Model
from autolabel.workbench.services.auto_labeling.types import AutoLabelingResult
from autolabel.workbench.views.labeling.shape import Shape
from autolabel.workbench.views.labeling.utils.opencv import qt_img_to_rgb_cv_img


def parse_class_names(text):
    names = [name.strip() for name in text.replace("，", ",").split(",")]
    if not names or any(not name for name in names):
        raise ValueError("请按顺序输入类别名称，多个类别用逗号分隔")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("类别名称不能重复")
    return names


class LocalYOLOImageModel(Model):
    class Meta(Model.Meta):
        required_config_names = ["model_path", "classes"]
        widgets = ["button_run", "input_conf", "edit_conf"]

    def __init__(self, model_config, on_message):
        super().__init__(model_config, on_message)
        from ultralytics import YOLO

        self.model = YOLO(self.config["model_path"])
        if self.model.task != "detect":
            raise ValueError("本地权重目前须为 YOLO 目标检测模型")
        self.class_names = list(self.config["classes"])
        if not self.class_names or len(self.class_names) > len(self.model.names):
            raise ValueError("输入的类别数量超过模型支持的类别数量")
        self.confidence = float(self.config.get("conf_threshold", 0.25))
        self.image_size = int(self.config.get("imgsz", 1280))

    def predict_shapes(self, image, filename=None):
        if image is None:
            return AutoLabelingResult([], replace=True)
        rgb = qt_img_to_rgb_cv_img(image, filename)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = self.model.predict(
            source=bgr, imgsz=self.image_size, conf=self.confidence,
            classes=list(range(len(self.class_names))),
            verbose=False)[0]
        shapes = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for box, class_id, score in zip(boxes, classes, scores):
                if class_id < 0 or class_id >= len(self.class_names):
                    continue
                x1, y1, x2, y2 = map(float, box)
                shape = Shape(flags={})
                for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                    shape.add_point(QtCore.QPointF(x, y))
                shape.shape_type = "rectangle"
                shape.closed = True
                shape.label = self.class_names[class_id]
                shape.score = float(score)
                shapes.append(shape)
        return AutoLabelingResult(shapes, replace=True)

    def unload(self):
        del self.model

    def set_auto_labeling_conf(self, value):
        self.confidence = float(value)
