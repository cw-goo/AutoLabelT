"""Qt-independent, atomic annotation writes and reversible batch replacement."""

import errno
import json
import logging
import os
from pathlib import Path
import tempfile


def describe_file_error(error):
    """Keep the failing path and distinguish ACL, locking and disk failures."""
    detail = str(error)
    cause = error
    while cause.__cause__ is not None:
        cause = cause.__cause__
    if isinstance(cause, PermissionError) or getattr(cause, "errno", None) == errno.EACCES:
        if getattr(cause, "winerror", None) in (32, 33):
            return f"文件被其他程序占用，请关闭占用程序后重试。\n{detail}"
        return (
            "无法访问文件。请检查文件/目录的安全权限和只读属性，"
            "或将数据集复制到当前账户可读写的目录后重试。\n" + detail
        )
    if getattr(cause, "errno", None) == errno.ENOSPC:
        return "磁盘空间不足，请释放空间后重试。\n" + detail
    return detail


def _temporary_path(destination, suffix):
    destination = Path(destination)
    descriptor, name = tempfile.mkstemp(
        prefix=f".autolabel_{destination.name}_", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_text(path, text, encoding="utf-8"):
    """Serialize first, then replace from a closed file on the same filesystem."""
    path = Path(path)
    temporary = _temporary_path(path, ".tmp")
    try:
        with temporary.open("w", encoding=encoding) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning("Cannot remove temporary annotation: %s", temporary)


def atomic_write_json(path, data, **kwargs):
    encoding = kwargs.pop("encoding", "utf-8")
    atomic_write_text(path, json.dumps(data, **kwargs), encoding)


class AnnotationTransaction:
    """Preflight a fixed set of paths, stage outputs, and roll back on failure.

    Fresh inference can move old files out of the active folder before running.
    Backups stay beside their original files until commit/rollback completes.
    No ownership or ACL changes are made by the application.
    """

    def __init__(self, paths):
        self.paths = tuple(dict.fromkeys(Path(path).absolute() for path in paths))
        self.staged = {}
        self.backups = {}
        self.installed = set()
        self.source_versions = {}
        self.finished = False

    def __enter__(self):
        self.preflight()
        return self

    def __exit__(self, *_):
        if not self.finished:
            self.rollback()

    def preflight(self):
        for path in self.paths:
            if path.is_symlink():
                raise ValueError(f"标注输出不能是符号链接：{path}")
            if path.exists():
                # Opening without writing checks both permissions without truncation.
                with path.open("r+b"):
                    pass
        for directory in dict.fromkeys(path.parent for path in self.paths):
            probe = _temporary_path(directory / "probe", ".tmp")
            probe.unlink()

    def _backup(self, path):
        if path not in self.backups and path.exists():
            backup = _temporary_path(path, ".backup")
            try:
                os.replace(path, backup)
            except BaseException:
                backup.unlink(missing_ok=True)
                raise
            self.backups[path] = backup

    def clear(self):
        """Hide old annotations, retaining a backup for cancellation or failure."""
        try:
            for path in self.paths:
                self._backup(path)
        except BaseException:
            self.rollback()
            raise

    def stage_path(self, path):
        path = Path(path).absolute()
        if self.finished or path not in self.paths:
            raise ValueError(f"标注不在本次任务的输出范围内：{path}")
        if path not in self.staged:
            self.staged[path] = _temporary_path(path, ".tmp")
        return self.staged[path]

    @staticmethod
    def _file_version(path):
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return stat.st_ino, stat.st_mtime_ns, stat.st_size

    def watch(self, path):
        path = Path(path).absolute()
        if path not in self.source_versions:
            self.source_versions[path] = self._file_version(path)

    def commit(self):
        try:
            for path, version in self.source_versions.items():
                if self._file_version(path) != version:
                    raise OSError(f"漏标补充期间原标注文件发生变化，请重新运行：{path}")
            for path, staged in self.staged.items():
                self._backup(path)
                os.replace(staged, path)
                self.installed.add(path)
        except BaseException:
            self.rollback()
            raise
        self.finished = True
        self._cleanup()

    def rollback(self):
        failures = []
        for path in self.installed:
            if path not in self.backups:
                try:
                    path.unlink(missing_ok=True)
                except OSError as error:
                    failures.append(str(error))
        for path, backup in list(self.backups.items()):
            try:
                os.replace(backup, path)
                del self.backups[path]
            except OSError as error:
                failures.append(f"{path}（原始备份保留在 {backup}）：{error}")
        self.finished = True
        self._cleanup(remove_backups=False)
        if failures:
            raise OSError("恢复旧标注失败，请保留以下备份：\n" + "\n".join(failures))

    def _cleanup(self, remove_backups=True):
        paths = list(self.staged.values())
        if remove_backups:
            paths.extend(self.backups.values())
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Cannot remove temporary annotation: %s", path)
