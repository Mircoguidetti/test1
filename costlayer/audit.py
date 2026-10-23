"""List every use of tau2's OpenAI (or otherwise non-ours) model defaults."""

from __future__ import annotations

import re
from pathlib import Path

import tau2

DEFAULTS = {
    "DEFAULT_LLM_AGENT": "agent model (overridden: we use our own agent, model per config)",
    "DEFAULT_LLM_USER": "user simulator (overridden: anthropic/claude-haiku-5-5)",
    "DEFAULT_LLM_NL_ASSERTIONS": "LLM judge for NL assertions (not reached with ENV evaluation)",
    "DEFAULT_LLM_ENV_INTERFACE": "LLM environment interface (not used by retail)",
    "DEFAULT_SEMANTIC_COMMUNICATE_MODEL": "semantic communicate judge (not reached with ENV)",
    "DEFAULT_ELICITATION_LLM_USER": "intake domain user (not our domain)",
    "DEFAULT_LLM_EVAL_USER_SIMULATOR": "user-simulator eval (not used)",
}


def audit() -> list[dict[str, object]]:
    root = Path(tau2.__file__).parent
    cfg = (root / "config.py").read_text()
    rows = []
    for name, note in DEFAULTS.items():
        m = re.search(rf"^{name}\s*=\s*(.+)$", cfg, re.M)
        uses = []
        for f in root.rglob("*.py"):
            if f.name == "config.py":
                continue
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if re.search(rf"\b{name}\b", line) and not line.lstrip().startswith("from "):
                    uses.append(f"{f.relative_to(root.parent)}:{i}")
        rows.append(
            {"name": name, "value": m.group(1).strip() if m else "?", "note": note, "uses": uses}
        )
    return rows
