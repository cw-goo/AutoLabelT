"""Qt charts for training metrics and recorded runs."""

import math
from dataclasses import dataclass

from PyQt6 import QtCore, QtGui, QtWidgets


LOSS_STYLES = {
    "box_loss": ("Box", "#13998b"),
    "cls_loss": ("分类", "#3578d9"),
    "dfl_loss": ("DFL", "#9865cb"),
    "giou_loss": ("GIoU", "#13998b"),
    "l1_loss": ("L1", "#ba7544"),
}
METRIC_STYLES = {
    "metrics/mAP50-95(B)": ("mAP50–95", "#7961cd"),
    "metrics/mAP50(B)": ("mAP50", "#13998b"),
    "metrics/precision(B)": ("Precision", "#367bd5"),
    "metrics/recall(B)": ("Recall", "#d18242"),
}


@dataclass
class Curve:
    label: str
    color: str
    values: dict[int, float]
    dashed: bool = False


class MetricChart(QtWidgets.QFrame):
    def __init__(self, title, *, loss=False, parent=None):
        super().__init__(parent)
        self.setObjectName("trainingChart")
        self.rows = []
        self.loss = loss
        self.curves = {}
        self.legend_buttons = {}
        self._legend_columns = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 13, 16, 12)
        layout.setSpacing(9)
        header = QtWidgets.QHBoxLayout()
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("chartHeading")
        header.addWidget(heading)
        header.addStretch()
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("chartSummary")
        self.summary.setToolTip("该曲线最新的实测值")
        self.summary.hide()
        header.addWidget(self.summary)
        layout.addLayout(header)

        self.plot = MetricPlot(percent=not loss, parent=self)
        layout.addWidget(self.plot, 1)
        self.legend = QtWidgets.QWidget()
        self.legend_layout = QtWidgets.QGridLayout(self.legend)
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_layout.setHorizontalSpacing(6)
        self.legend_layout.setVerticalSpacing(5)
        self.legend_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.legend.hide()
        layout.addWidget(self.legend)

    def set_metrics(self, rows, *, animate=False, reset=False):
        self.rows = list(rows)
        values = {}
        for epoch, metrics in sorted(rows, key=lambda row: row[0]):
            for key, raw in metrics.items():
                belongs = "_loss" in key if self.loss else key.startswith("metrics/")
                if not belongs:
                    continue
                if isinstance(raw, (int, float)) and math.isfinite(raw):
                    values.setdefault(key, {})[epoch] = float(raw)
        if self.loss:
            ordered = (f"{split}/{kind}" for split in ("train", "val") for kind in LOSS_STYLES)
        else:
            ordered = iter(METRIC_STYLES)
        keys = [key for key in ordered if key in values]
        keys += sorted(values.keys() - set(keys))
        curves = {}
        for key in keys:
            if self.loss:
                split, _, kind = key.partition("/")
                name, color = LOSS_STYLES.get(kind, (kind.replace("_loss", ""), "#667a94"))
                label = f"{'验证' if split == 'val' else '训练'} {name}"
                curves[key] = Curve(label, color, values[key], dashed=split == "val")
            else:
                label, color = METRIC_STYLES.get(key, (key.removeprefix("metrics/"), "#667a94"))
                curves[key] = Curve(label, color, values[key])
        if reset or keys != list(self.curves):
            self._build_legend(curves, keep_selection=not reset)
        self.curves = curves
        self._update_plot(animate=animate, reset=reset)

    def _build_legend(self, curves, *, keep_selection):
        previous = {key: button.isChecked() for key, button in self.legend_buttons.items()}
        for button in self.legend_buttons.values():
            self.legend_layout.removeWidget(button)
            button.hide()
            button.deleteLater()
        self.legend_buttons.clear()
        for key, curve in curves.items():
            button = QtWidgets.QToolButton()
            button.setObjectName("chartSeries")
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            button.setCheckable(True)
            button.setChecked(previous.get(key, True) if keep_selection else True)
            button.setText(curve.label)
            button.setToolTip(f"{curve.label} · 点击切换显示")
            swatch = QtGui.QPixmap(16, 12)
            swatch.fill(QtCore.Qt.GlobalColor.transparent)
            with QtGui.QPainter(swatch) as painter:
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.setPen(_series_pen(curve, 2))
                painter.drawLine(1, 6, 15, 6)
            button.setIcon(QtGui.QIcon(swatch))
            button.setIconSize(swatch.size())
            button.toggled.connect(self._update_plot)
            self.legend_buttons[key] = button
        self.legend.setVisible(bool(curves))
        self._layout_legend(force=True)

    def _layout_legend(self, *, force=False):
        if not self.legend_buttons:
            return
        size = max(button.sizeHint().width() for button in self.legend_buttons.values())
        available = max(1, (self.width() - 32 + 6) // (size + 6))
        columns = min(3 if self.loss else 4, available)
        if columns == self._legend_columns and not force:
            return
        self._legend_columns = columns
        for index, button in enumerate(self.legend_buttons.values()):
            row, column = divmod(index, columns)
            self.legend_layout.removeWidget(button)
            self.legend_layout.addWidget(button, row, column, QtCore.Qt.AlignmentFlag.AlignLeft)
        chip_height = max(button.sizeHint().height() for button in self.legend_buttons.values())
        self.legend.setMinimumHeight(2 * chip_height + self.legend_layout.verticalSpacing())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._layout_legend()

    def _update_plot(self, _checked=None, *, animate=False, reset=False):
        visible = [curve for key, curve in self.curves.items() if self.legend_buttons[key].isChecked()]
        if visible:
            lead = visible[0]
            value = lead.values[max(lead.values)]
            self.summary.setText(f"{lead.label}  {self.plot.format_value(value)}")
        self.summary.setVisible(bool(visible))
        self.plot.set_curves(visible, animate=animate, reset=reset, has_data=bool(self.curves))


def _series_pen(curve, width):
    pen = QtGui.QPen(QtGui.QColor(curve.color), width)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    if curve.dashed:
        pen.setDashPattern([3, 3])
    return pen


class MetricPlot(QtWidgets.QWidget):
    def __init__(self, *, percent, parent=None):
        super().__init__(parent)
        self.percent = percent
        self.curves = []
        self.epochs = []
        self.ticks = self._y_ticks()
        self.has_data = False
        self.mouse_position = None
        self.hovered_epoch = None
        self.reveal = 1.0
        self.reveal_start = 1
        self.setMinimumSize(236, 132)
        self.setMouseTracking(True)
        self.animation = QtCore.QVariantAnimation(self)
        self.animation.setDuration(420)
        self.animation.setStartValue(0.0)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        self.animation.valueChanged.connect(self._advance)

    def set_curves(self, curves, *, animate, reset, has_data):
        previous_end = self.epochs[-1] if self.epochs and not reset else 0
        self.curves = curves
        self.epochs = sorted({epoch for curve in curves for epoch in curve.values})
        self.ticks = self._y_ticks()
        self.has_data = has_data
        if reset:
            self.mouse_position = self.hovered_epoch = None
        self.animation.stop()
        self.reveal = 1.0
        if animate and self.isVisible() and self.epochs and self.epochs[-1] > previous_end:
            self.reveal_start = max(1, previous_end)
            self.reveal = 0.0
            self.animation.start()
        self.update()

    def _advance(self, value):
        self.reveal = value
        self.update()

    def format_value(self, value):
        return f"{value:.2%}" if self.percent else f"{value:.4f}"

    def plot_rect(self):
        return QtCore.QRectF(self.rect()).adjusted(45, 9, -19, -26)

    def mouseMoveEvent(self, event):
        plot = self.plot_rect()
        self.mouse_position = event.position() if plot.contains(event.position()) else None
        self.hovered_epoch = None
        if self.mouse_position is not None and self.epochs:
            end = max(2, self.epochs[-1])
            epoch = 1 + (self.mouse_position.x() - plot.left()) / plot.width() * (end - 1)
            self.hovered_epoch = min(self.epochs, key=lambda value: abs(value - epoch))
        self.update()

    def leaveEvent(self, event):
        self.mouse_position = self.hovered_epoch = None
        self.update()
        super().leaveEvent(event)

    def _y_ticks(self):
        values = [value for curve in self.curves for value in curve.values.values()]
        if self.percent and all(0 <= value <= 1 for value in values):
            return [0, .25, .5, .75, 1]
        low, high = min([0, *values]), max([.001, *values])
        interval = (high - low) / 4
        power = 10 ** math.floor(math.log10(interval))
        step = next(factor * power for factor in (1, 2, 2.5, 5, 10) if factor * power >= interval)
        start, end = math.floor(low / step), math.ceil(high / step)
        return [mark * step for mark in range(start, end + 1)]

    def paintEvent(self, event):
        plot = self.plot_rect()
        ticks = self.ticks
        end = max(2, self.epochs[-1]) if self.epochs else 2

        def position(epoch, value):
            x = plot.left() + (epoch - 1) / (end - 1) * plot.width()
            y = plot.bottom() - (value - ticks[0]) / (ticks[-1] - ticks[0]) * plot.height()
            return QtCore.QPointF(x, y)

        with QtGui.QPainter(self) as painter:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            self._draw_axes(painter, plot, ticks, end, position)
            if not self.epochs:
                painter.setPen(QtGui.QColor("#677e93"))
                painter.drawText(plot, QtCore.Qt.AlignmentFlag.AlignCenter,
                                 "点击图例显示曲线" if self.has_data else "等待训练数据")
                return
            self._draw_curves(painter, plot, position)
            if self.hovered_epoch in self.epochs:
                self._draw_tooltip(painter, plot, position)

    def _draw_axes(self, painter, plot, ticks, end, position):
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#f6f9fc"))
        painter.drawRoundedRect(plot, 8, 8)
        for tick in ticks:
            y = position(1, tick).y()
            painter.setPen(QtGui.QColor("#e3eaf2"))
            painter.drawLine(QtCore.QPointF(plot.left(), y), QtCore.QPointF(plot.right(), y))
            if self.epochs:
                painter.setPen(QtGui.QColor("#566d83"))
                label = f"{tick:.0%}" if self.percent else f"{tick:.3g}"
                painter.drawText(QtCore.QRectF(0, y - 8, plot.left() - 7, 16),
                                 QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, label)
        if not self.epochs:
            return
        painter.setPen(QtGui.QColor("#566d83"))
        painter.drawText(QtCore.QRectF(0, plot.bottom() + 3, plot.left() - 21, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "轮次")
        count = min(6, end)
        for index in range(count):
            epoch = round(1 + (end - 1) * index / (count - 1))
            x = position(epoch, ticks[0]).x()
            painter.setPen(QtGui.QColor("#e3eaf2"))
            painter.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
            painter.setPen(QtGui.QColor("#566d83"))
            painter.drawText(QtCore.QRectF(x - 21, plot.bottom() + 3, 42, 18),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(epoch))

    def _draw_curves(self, painter, plot, position):
        painter.save()
        reveal_end = self.reveal_start + (self.epochs[-1] - self.reveal_start) * self.reveal
        right = position(reveal_end, 0).x() if self.reveal < 1 else plot.right()
        painter.setClipRect(plot.adjusted(-4, -4, right - plot.right() + 4, 4))
        if len(self.epochs) == 1:
            painter.setOpacity(self.reveal)
        for index, curve in enumerate(self.curves):
            points = [position(epoch, value) for epoch, value in curve.values.items()]
            line = QtGui.QPainterPath(points[0])
            for point in points[1:]:
                line.lineTo(point)
            if index == 0 and len(points) > 1:
                area = QtGui.QPainterPath(line)
                area.lineTo(points[-1].x(), plot.bottom())
                area.lineTo(points[0].x(), plot.bottom())
                area.closeSubpath()
                gradient = QtGui.QLinearGradient(0, plot.top(), 0, plot.bottom())
                tint = QtGui.QColor(curve.color)
                tint.setAlpha(45)
                gradient.setColorAt(0, tint)
                tint.setAlpha(2)
                gradient.setColorAt(1, tint)
                painter.fillPath(area, QtGui.QBrush(gradient))
            painter.setPen(_series_pen(curve, 2.25))
            painter.strokePath(line, painter.pen())
            painter.setPen(QtGui.QPen(QtGui.QColor(curve.color), 1.5))
            painter.setBrush(QtGui.QColor("#ffffff"))
            painter.drawEllipse(points[-1], 3, 3)
        painter.restore()

    def _draw_tooltip(self, painter, plot, position):
        epoch = self.hovered_epoch
        entries = [(curve, curve.values[epoch]) for curve in self.curves if epoch in curve.values]
        x = position(epoch, 0).x()
        painter.setPen(QtGui.QPen(QtGui.QColor("#9aadc0"), 1, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(QtCore.QPointF(x, plot.top()), QtCore.QPointF(x, plot.bottom()))
        labels = [(curve, self.format_value(value)) for curve, value in entries]
        for curve, value in entries:
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
            painter.setBrush(QtGui.QColor(curve.color))
            painter.drawEllipse(position(epoch, value), 4, 4)
        metrics = painter.fontMetrics()
        label_width = max(metrics.horizontalAdvance(curve.label) for curve, _ in labels)
        value_width = max(metrics.horizontalAdvance(value) for _, value in labels)
        row_height = min(19, max(13, (self.height() - 34) // len(entries)))
        width = min(self.width() - 4, max(126, label_width + value_width + 54))
        height = 30 + row_height * len(entries)
        left = x + 13 if x + 13 + width < self.width() else x - width - 13
        left = max(2, min(left, self.width() - width - 2))
        top = max(2, min(self.mouse_position.y() + 12, self.height() - height - 2))
        card = QtCore.QRectF(left, top, width, height)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#21364b"))
        painter.drawRoundedRect(card, 8, 8)
        painter.setPen(QtGui.QColor("#f0f6fb"))
        painter.drawText(QtCore.QPointF(left + 12, top + 18), f"第 {epoch} 轮")
        for row, (curve, value) in enumerate(labels):
            y = top + 28 + (row + 1) * row_height - 5
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor(curve.color).lighter(135))
            painter.drawEllipse(QtCore.QPointF(left + 14, y - 4), 3, 3)
            painter.setPen(QtGui.QColor("#e2ecf5"))
            label = metrics.elidedText(curve.label, QtCore.Qt.TextElideMode.ElideRight, width - value_width - 45)
            painter.drawText(QtCore.QPointF(left + 23, y), label)
            painter.drawText(QtCore.QRectF(left + width - value_width - 12, y - 13, value_width, 16),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, value)
