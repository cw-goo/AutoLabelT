import io
import contextlib


@contextlib.contextmanager
def io_open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    with io.open(name, mode, encoding=encoding) as stream:
        yield stream
