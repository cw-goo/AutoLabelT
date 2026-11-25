#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3.11}"
install_dir="${AUTOLABEL_INSTALL_DIR:-$HOME/.local/share/autolabel}"

"$python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
shopt -s nullglob
wheels=("$bundle_dir"/autolabel-*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "Expected one AutoLabelT wheel next to install.sh" >&2
    exit 1
fi

if [[ ! -x "$install_dir/bin/python" ]]; then
    "$python" -m venv "$install_dir"
fi
install_dir="$(cd -- "$install_dir" && pwd)"
venv_python="$install_dir/bin/python"
"$venv_python" -m pip install --upgrade "${wheels[0]}"
if [[ ! -x "$install_dir/bin/autolabelt" || -x "$install_dir/bin/autolabel" ]]; then
    "$venv_python" -m pip install --force-reinstall --no-deps "${wheels[0]}"
fi
"$venv_python" -c "import autolabel.workbench, PyQt6, cv2, torch, ultralytics, onnx, onnxruntime; from importlib.resources import files; assert files('autolabel').joinpath('assets/logo.png').is_file(); assert files('autolabel.workbench').joinpath('configs/autolabel_config.yaml').is_file(); from autolabel.workbench.resources import resources"

command_dir="$HOME/.local/bin"
mkdir -p -- "$command_dir"
ln -sfnT -- "$install_dir/bin/autolabelt" "$command_dir/autolabelt"
legacy_launcher="$install_dir/AutoLabel"
if [[ -f "$legacy_launcher" && ! -L "$legacy_launcher" ]] &&
    cmp -s -- "$legacy_launcher" <(printf '#!/usr/bin/env bash\nexec %q -m autolabel "$@"\n' "$venv_python"); then
    rm -- "$legacy_launcher"
fi
echo "Installed AutoLabelT to $install_dir"
echo "Start it with: autolabelt (or $command_dir/autolabelt)"
