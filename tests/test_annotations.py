"""The label I/O checks remain independent of the removed prototype interface."""
import pytest

from autolabel.annotations import read_labels, write_labels


def test_annotation_roundtrip_and_unknown_class(tmp_path):
    label = tmp_path / "one.txt"
    write_labels(label, [(0, 0.1, 0.2, 0.5, 0.9)])
    assert read_labels(label, 1)[0] == pytest.approx((0, 0.1, 0.2, 0.5, 0.9))
    with pytest.raises(ValueError, match="unknown class"):
        read_labels(label, 0)
