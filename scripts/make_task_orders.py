"""Write results/task_orders.json: the sweep order of the tasks of each test.

Outside the optimizer package on purpose: it imports tau2 to rebuild the same orders the
runners used (retail: official train split shuffled with seed 42, first N_train; banking:
`banking_order(seed=42, exclude=ACTION)`, first 12). The failure-anatomy analysis reads the file
and never imports tau2.
"""

from __future__ import annotations

import json
from pathlib import Path

from costlayer import banking_phase2 as b2
from costlayer.runner import split_tasks

RETAIL_N_TRAIN = 21  # reports/phase-2.md


def main() -> None:
    out = {
        "retail_train_sweep": [t.id for t in split_tasks("train")[:RETAIL_N_TRAIN]],
        "banking_sweep": b2.train_ids(),
    }
    Path("results/task_orders.json").write_text(json.dumps(out, indent=1) + "\n")
    print({k: len(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
