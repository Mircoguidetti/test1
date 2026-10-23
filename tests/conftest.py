import json
from pathlib import Path

import pytest

from costlayer.batch import run_batch
from costlayer.domains import make_split
from costlayer.learn import learn_profile
from costlayer.pricing import HAIKU
from costlayer.sim import ScriptedLLM


@pytest.fixture(scope="session")
def built(tmp_path_factory):
    """Scripted baseline on a train subset, a learned profile, and layer runs on the same tasks."""
    d = tmp_path_factory.mktemp("built")
    llm = ScriptedLLM()
    train = make_split()["train"]
    refs = [r for r in train if r.domain == "retail"][:14] + [
        r for r in train if r.domain == "airline"
    ][:8]
    run_batch(
        refs, ["baseline"], llm, HAIKU, d / "runs.jsonl", d / "t", workers=1, label_baseline=True
    )
    traces = [json.loads(p.read_text()) for p in (d / "t").glob("*.json")]
    profile = learn_profile(traces, min_support=2)
    for ts in profile.types.values():
        for tp in ts.values():
            tp.compact_after_tokens = 2500
    run_batch(refs, ["layer"], llm, HAIKU, d / "runs.jsonl", d / "t", profile=profile, workers=1)
    return {"dir": d, "llm": llm, "refs": refs, "profile": profile, "traces": traces}


@pytest.fixture
def tmp_ledger(tmp_path: Path):
    from costlayer.ledger import Ledger

    return Ledger(tmp_path / "ledger.jsonl", phase=1)
