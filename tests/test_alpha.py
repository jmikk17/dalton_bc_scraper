import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent

OPERATOR_DICT = {"XDIPLEN": 0, "YDIPLEN": 1, "ZDIPLEN": 2}


def extract_alpha_from_json(json_file: Path) -> np.ndarray:
    """Extract alpha values from a JSON file."""
    with Path(json_file).open("r") as f:
        data = json.load(f)

    return data.get("polarizability", None).get("alpha_tot", None)


def extract_alpha_from_out(out_file: Path, *, cc_wf: bool) -> np.ndarray:
    """Extract alpha values from an OUT file."""
    alpha_ref = np.zeros((3, 3))

    with Path(out_file).open("r") as f:
        data = f.read()

    if cc_wf:
        pattern = (
            r"(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+([-+]?\d+\.\d+(?:E[-+]?\d+)?)"
        )
    else:
        pattern = r"@\s*-<<\s*(\w+)\s*;\s*(\w+)\s*>>\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)"

    for match in re.finditer(pattern, data):
        operator1 = match.group(1)
        operator2 = match.group(2)
        value = match.group(3)

        alpha_ref[OPERATOR_DICT[operator1], OPERATOR_DICT[operator2]] = float(value)

        if operator1 != operator2 and not cc_wf:
            alpha_ref[OPERATOR_DICT[operator2], OPERATOR_DICT[operator1]] = float(value)

    return alpha_ref


@pytest.mark.parametrize(
    ("basename", "cc"),
    [
        ("pbe_h2o", False),
        ("casH2_h2", False),
        ("cc2_h2o", True),
    ],
)
def test_alpha(basename: str, cc: bool, tmp_path: Path) -> None:
    """Test the alpha values extracted from the OUT file and the JSON file."""
    dalton_file = TEST_DIR / f"atmmom1_{basename}.out"
    ref_alpha_file = TEST_DIR / f"alpha_{basename}.out"

    json_file = tmp_path / f"atmmom1_{basename}.json"

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            str(dalton_file),
            "--o",
            str(json_file),
        ],
        check=True,
    )

    alpha_ref = extract_alpha_from_out(ref_alpha_file, cc_wf=cc)
    alpha = extract_alpha_from_json(json_file)

    assert np.allclose(alpha_ref, alpha, atol=1e-6), (
        f"Alpha values do not match for {basename}: REF file: {alpha_ref}, JSON file: {alpha}"
    )
