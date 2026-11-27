"""Application identity shared by the bundled workbench."""
from autolabelt.branding import APP_NAME, APP_SUMMARY, LAUNCH_COMMAND, app_version

__appname__ = APP_NAME
__appdescription__ = APP_SUMMARY
__version__ = app_version()
__url__ = "README.md"

CLI_HELP_MSG = f"""{APP_NAME}

Launch: {LAUNCH_COMMAND}
Commands: {LAUNCH_COMMAND} help | checks | version | config | convert
Conversion help: {LAUNCH_COMMAND} convert --help
Documentation: README.md and docs/
Third-party attribution: THIRD_PARTY_NOTICES.md
"""


def __getattr__(name):
    if name == "__preferred_device__":
        from autolabelt.workbench.views.common.device_manager import get_preferred_device
        return get_preferred_device()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
