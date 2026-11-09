"""AutoLabel launcher identity and settings migration tests."""

from pathlib import Path


def test_svg_export_keeps_adjacent_colour_shapes_opaque(tmp_path):
    from PyQt6 import QtGui
    from tools.export_brand_icons import render_svg

    source = tmp_path / "strips.svg"
    source.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        + ''.join(f'<path d="M0 {y}h32v1H0Z" fill="rgb(0,{128+y},220)"/>'
                  for y in range(32)) + '</svg>', encoding="utf-8")
    output = tmp_path / "small.png"
    render_svg(source, output, 13)
    image = QtGui.QImage(str(output))
    assert image.size().width() == image.size().height() == 13
    assert all(image.pixelColor(x, y).alpha() == 255
               for x in range(13) for y in range(13))


def test_branded_banner_has_no_upstream_links():
    from autolabel.branding import startup_banner

    banner = startup_banner()
    assert "AutoLabel" in banner
    assert "VIDEO  >  FRAMES  >  LABELS  >  DATASET" in banner
    assert "autolabel.com" not in banner
    assert "github.com/CVHub520" not in banner
    assert "\x1b[" not in banner
    assert len({len(line) for line in banner.splitlines()}) == 1


def test_banner_colors_only_in_interactive_mode():
    from io import StringIO
    from autolabel.branding import startup_banner, write_startup_banner

    assert "\x1b[" in startup_banner(use_color=True)
    output = StringIO()
    write_startup_banner(output)
    assert "AutoLabel" in output.getvalue()
    assert "\x1b[" not in output.getvalue()


def test_autolabel_config_migrates_without_changing_upstream(monkeypatch, tmp_path):
    from autolabel.workbench import config

    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    previous = tmp_path / ".xanylabelingrc"
    previous.write_text("language: zh_CN\n", encoding="utf-8")
    monkeypatch.setenv("AUTOLABEL_INTEGRATED_WORKBENCH", "1")
    config.get_default_config()
    copied = tmp_path / ".autolabelrc"
    assert copied.read_bytes() == previous.read_bytes()
    assert config.get_user_config_name() == ".autolabelrc"
    monkeypatch.delenv("AUTOLABEL_INTEGRATED_WORKBENCH")
    assert config.get_user_config_name() == ".autolabelrc"
    assert previous.read_text(encoding="utf-8") == "language: zh_CN\n"


def test_autolabel_cli_reports_own_version_and_config(monkeypatch, capsys, tmp_path):
    import sys
    from autolabel.workbench import config
    from autolabel.desktop import main

    monkeypatch.setenv("AUTOLABEL_INTEGRATED_WORKBENCH", "1")
    monkeypatch.setattr(config, "_work_directory", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["autolabel", "version"])
    main()
    assert "0.1.0" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["autolabel", "--work-dir", str(tmp_path), "config"])
    main()
    assert str(Path(tmp_path) / ".autolabelrc") in capsys.readouterr().out
