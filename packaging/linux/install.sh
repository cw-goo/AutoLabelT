#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON:-python3.11}"
install_dir="${AUTOLABEL_INSTALL_DIR:-$HOME/.local/share/autolabel}"

"$python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
shopt -s nullglob
wheels=("$bundle_dir"/autolabel-*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "Expected one AutoLabel wheel next to install.sh" >&2
    exit 1
fi

if [[ ! -x "$install_dir/bin/python" ]]; then
    "$python" -m venv "$install_dir"
fi
venv_python="$install_dir/bin/python"
"$venv_python" -m pip install --upgrade "${wheels[0]}"
"$venv_python" -c "import autolabel.workbench, PyQt6, cv2, torch, ultralytics; from importlib.resources import files; assert files('autolabel').joinpath('assets/logo.svg').is_file(); assert files('autolabel.workbench').joinpath('configs/autolabel_config.yaml').is_file(); from autolabel.workbench.resources import resources"

launcher="$install_dir/AutoLabel"
printf '#!/usr/bin/env bash\nexec %q -m autolabel "$@"\n' "$venv_python" > "$launcher"
chmod 755 "$launcher"
echo "Installed AutoLabel to $install_dir"
echo "Start it with: $launcher"
