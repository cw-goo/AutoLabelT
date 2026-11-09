"""Application identity shared by the bundled workbench."""
from autolabel.branding import APP_NAME, app_version

__appname__ = APP_NAME
__appdescription__ = "AutoLabel video and image annotation workbench"
__version__ = app_version()
__url__ = "README.md"

CLI_HELP_MSG = """AutoLabel

Launch: autolabel
Commands: autolabel help | checks | version | config | convert
Conversion help: autolabel convert --help
Documentation: README.md and docs/
Third-party attribution: THIRD_PARTY_NOTICES.md
"""


def __getattr__(name):
    if name == "__preferred_device__":
        from autolabel.workbench.views.common.device_manager import get_preferred_device
        return get_preferred_device()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
