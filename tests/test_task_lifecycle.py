"""Cancellation, task exclusion and nonblocking desktop shutdown."""

import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest


@pytest.fixture
def application(monkeypatch):
    from PyQt6 import QtWidgets
    from autolabel.workbench.resources import resources

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("worker_name", ["pipeline_worker", "model_worker",
                                         "sync_worker", "prepare_worker"])
def test_running_task_blocks_conflicting_actions(application, monkeypatch, worker_name):
    from PyQt6 import QtWidgets
    from autolabel.video_dock import VideoDatasetDock

    window = QtWidgets.QMainWindow()
    dock = VideoDatasetDock(window)
    busy = [True]
    setattr(dock, worker_name, SimpleNamespace(isRunning=lambda: busy[0]))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        lambda *a: pytest.fail("project changed while task runs"))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        lambda *a: pytest.fail("output changed while task runs"))
    dock._refresh_task_controls()
    assert not dock.start_button.isEnabled() and not dock.sync_button.isEnabled()
    assert not dock.output_button.isEnabled()
    dock.start()
    dock.sync_review()
    dock.open_review()
    dock.choose_video()
    dock.choose_output()
    dock.load_config()
    dock._model_task("test")
    busy[0] = False
    dock._refresh_task_controls()
    assert dock.start_button.isEnabled() and dock.output_button.isEnabled()
    dock.close()


def test_cancelled_model_test_cannot_be_marked_ready(monkeypatch):
    from autolabel.models import worker as module

    worker = module.ModelWorker("test", "local_yolo", path="fake.pt")
    monkeypatch.setattr(module, "create_detector", lambda *a: SimpleNamespace(
        predict=lambda frame: worker.cancel()))
    monkeypatch.setattr(module, "mark_model_ready",
                        lambda *a: pytest.fail("cancelled test marked ready"))
    failures, completed = [], []
    worker.failed.connect(failures.append)
    worker.completed.connect(completed.append)
    worker.run()
    assert failures and not completed


def test_silent_installer_can_be_cancelled_without_stdout(monkeypatch, tmp_path):
    from autolabel.models import worker as module

    worker = module.ModelWorker("install_runtime", "yolo26_n")
    original_popen = subprocess.Popen
    children = []
    interpreters = []
    marker = tmp_path / "started"

    def spawn(command, **kwargs):
        child = original_popen([sys.executable, "-c",
                                "import os,pathlib,sys,time; "
                                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)",
                                str(marker)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", spawn)

    def cancel_when_started():
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if marker.exists() and marker.read_text():
                interpreters.append(psutil.Process(int(marker.read_text())))
                break
            time.sleep(0.01)
        worker.cancel()

    timer = threading.Timer(0.05, cancel_when_started)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(InterruptedError):
            worker._install_runtime()
        assert children and all(child.poll() is not None for child in children)
        assert interpreters and all(not child.is_running() for child in interpreters)
        assert time.monotonic() - start < 8
    finally:
        timer.cancel()
        timer.join()
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
        for child in interpreters:
            if child.is_running():
                child.kill()
                child.wait(timeout=3)


def test_close_keeps_event_loop_alive_until_worker_finishes(application, monkeypatch):
    from PyQt6 import QtGui
    from autolabel.workbench import config as settings
    from autolabel.workbench.views.mainwindow import MainWindow
    from autolabel.desktop import AutoLabelMainWindow

    window = AutoLabelMainWindow(application, config=settings.get_default_config())
    running, cancelled = [True], []
    window.video_dock.pipeline_worker = SimpleNamespace(
        isRunning=lambda: running[0], cancel=lambda: cancelled.append(True))
    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted() and cancelled
    assert window._close_timer.isActive()
    application.processEvents()
    running[0] = False
    monkeypatch.setattr(MainWindow, "closeEvent", lambda self, event: event.accept())
    window._close_when_idle()
    assert not window._close_timer.isActive()
    window.close()
