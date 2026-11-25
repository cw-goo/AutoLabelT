"""Product identity and retained third-party attribution."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QLabel, QPushButton, QVBoxLayout

from autolabel.branding import APP_NAME, APP_SUMMARY, app_version


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"关于 {APP_NAME}")
        self.resize(470, 240)
        layout = QVBoxLayout(self)
        title = QLabel(f"{APP_NAME} {app_version()}")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        notice = QLabel(
            f"{APP_SUMMARY}\n\n"
            "标注工作台基于 X-AnyLabeling 4.2.0，遵循 GPL-3.0。\n"
            "许可及来源说明见 LICENSE 和 THIRD_PARTY_NOTICES.md。"
        )
        notice.setWordWrap(True)
        notice.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(notice)
        close = QPushButton("关闭")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
