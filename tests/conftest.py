import os
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / ".tau2-bench" / "data"
if "TAU2_DATA_DIR" not in os.environ and DATA.exists():
    os.environ["TAU2_DATA_DIR"] = str(DATA)

# tau2 reads TAU2_DATA_DIR at import time, so set it before anything imports tau2.
if "TAU2_DATA_DIR" not in os.environ:
    raise RuntimeError(
        "Set TAU2_DATA_DIR or clone tau2-bench at the pinned commit into .tau2-bench/ "
        "(see .github/workflows/ci.yml)"
    )

from costlayer.ledger import Ledger  # noqa: E402


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", "d1", phase=1, phase_cap=2.5)
