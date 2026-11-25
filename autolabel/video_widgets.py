"""Reusable workflow cards for the video sidebar."""

from PyQt6 import QtCore, QtWidgets


class WorkflowCard(QtWidgets.QFrame):
    def __init__(self, number, title, caption, parent=None):
        super().__init__(parent)
        self.setProperty("workflowCard", True)
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(11, 9, 11, 10)
        self.body.setSpacing(8)
        header_widget = QtWidgets.QWidget()
        header_widget.setFixedHeight(24)
        header = QtWidgets.QHBoxLayout(header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        badge = QtWidgets.QLabel(number)
        badge.setObjectName("workflowBadge")
        badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(22, 18)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("workflowCardTitle")
        hint = QtWidgets.QLabel(caption)
        hint.setObjectName("workflowCardCaption")
        hint.setVisible(bool(caption))
        header.addWidget(badge)
        header.addWidget(heading)
        header.addStretch()
        header.addWidget(hint)
        self.body.addWidget(header_widget)
        self.header = header
        self.caption = hint

    def set_caption_widget(self, widget):
        self.header.replaceWidget(self.caption, widget)
        self.caption.hide()
        self.caption.deleteLater()
        self.caption = widget
        self.header.setStretch(self.header.count() - 1, 1)


class SummaryLabel(QtWidgets.QLabel):
    """Bound summary height and retain the complete text in a plain tooltip."""

    def __init__(self, text="", lines=1, parent=None):
        super().__init__(parent)
        self.lines = lines
        self.full_text = ""
        self.setObjectName("workflowSummary")
        self.setProperty("tone", "muted")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setText(text)

    def setText(self, text):
        self.full_text = str(text)
        self.setToolTip(self.full_text)
        self.setAccessibleName(self.full_text)
        self._refresh_text()

    def text(self):
        return self.full_text

    def _refresh_text(self):
        metrics = self.fontMetrics()
        lines = self.full_text.splitlines()[:self.lines]
        height = metrics.lineSpacing() * max(1, len(lines)) + 2
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        super().setText("\n".join(metrics.elidedText(
            line, QtCore.Qt.TextElideMode.ElideRight, max(10, self.width())) for line in lines))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_text()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.FontChange, QtCore.QEvent.Type.StyleChange):
            self._refresh_text()
