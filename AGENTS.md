# Repository Guidelines

## Project Structure & Module Organization

`autolabel/` contains the AutoLabel video workflow, model management, review logic, and desktop integration. `autolabel/workbench/` contains the bundled X-AnyLabeling annotation workbench and image labeling features. Keep changes to these packages scoped to the feature they affect. `databuild.py` is the verified video algorithm baseline; compare behavior against it when changing frame sampling or filtering. Tests live in `tests/`; architecture notes and UI images live in `docs/`; third-party license text and attribution live in `LICENSE` and `THIRD_PARTY_NOTICES.md`. Treat `artifacts/`, `build/`, and `*.egg-info/` as generated output, not source.

## Build, Test, and Development Commands

Use Python 3.11 (the package supports `>=3.11,<3.13`). Install the application and test dependency with `python -m pip install -e ".[test]"`. Run the integrated desktop workbench with `python -m autolabel`; run the original annotation app with `python -m autolabel.workbench.app --no-auto-update-check`. Run the full suite with `python -m pytest -q tests`. On Windows hosts where pytest cannot use its default temporary directory, set `TEMP` and `TMP` to writable directories and pass `--basetemp=artifacts/test_runs/current`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and descriptive test names. Keep Qt-independent data and validation logic in `autolabel/core.py` where practical; desktop UI code belongs in the desktop and UI modules. Preserve the established JSON and YOLO label formats. No formatter or linter is configured in `pyproject.toml`, so match nearby code and keep diffs focused.

## Testing Guidelines

Tests use pytest and follow `tests/test_*.py` with `test_*` functions. Add regression cases for changed sampling, model lifecycle, label conversion, or review behavior. Use pytest's `tmp_path` and `monkeypatch` for filesystem and environment changes; Qt tests can set `QT_QPA_PLATFORM=offscreen`. No coverage threshold is configured. Run the relevant test file during development and the full suite before submitting.

## Commit & Pull Request Guidelines

This checkout has no Git history, so no existing commit-message convention can be verified. Use a short imperative subject describing the change, such as `Fix video frame filtering`. In pull requests, explain the user-visible behavior, link the relevant issue if one exists, list tests run, and include screenshots for UI changes. Do not commit model weights, local test artifacts, or generated build output. Preserve the X-AnyLabeling GPL license obligations when changing or redistributing bundled code.
