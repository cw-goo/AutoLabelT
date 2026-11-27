"""One ordered class mapping for importing, validating and exporting labels."""

from pathlib import Path
import re


def read_class_names(file_path):
    names = [name.strip() for name in Path(file_path).read_text(encoding="utf-8-sig").splitlines()]
    if not names or any(not name for name in names):
        raise ValueError(
            f"Classes file '{file_path}' is empty or contains blank "
            "class names. Use one class name per line."
        )
    numbered = [re.fullmatch(r"([0-9]+)\s+(.+)", name) for name in names]
    if all(numbered):
        entries = sorted((int(match[1]), match[2].strip()) for match in numbered)
        if [class_id for class_id, _ in entries] != list(range(len(entries))):
            raise ValueError(
                f"Invalid class IDs in '{file_path}': numbered classes "
                "must use each ID from 0 to N-1 exactly once."
            )
        names = [name for _, name in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class names in '{file_path}'.")
    return names
