"""Keep the class palette and editor history in the same workspace context."""

from PyQt6.QtCore import Qt


def palette_names(view):
    return [view.unique_label_list.item(index).data(Qt.ItemDataRole.UserRole)
            for index in range(view.unique_label_list.count())]


def set_label_palette(view, labels):
    labels = list(dict.fromkeys(labels))
    view.unique_label_list.clear()
    view.label_dialog.label_list.clear()
    view.load_labels(labels)
    view.label_dialog.label_list.addItems(labels)
    if getattr(view.label_dialog, "_sort_labels", False):
        view.label_dialog.sort_labels()

