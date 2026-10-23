"""Learned per-task-type profile (written from training traces only)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TypeProfile:
    n_traces: int
    tools: list[str]  # lever 8: tools to expose
    sections: list[str]  # lever 8: policy headings to keep (lower case)
    path: list[str]  # lever 6: suggested tool sequence
    effort: str | None  # lever 7
    max_tokens: int  # lever 11
    tool_fields: dict[str, list[str]] = field(default_factory=dict)  # lever 9: keys to DROP
    compact_after_tokens: int | None = None  # lever 10
    max_steps: int = 30  # loop stop / fallback bound
    transfer_rate: float = 0.0  # share of training traces that ended in a transfer


@dataclass
class Profile:
    meta: dict
    types: dict[str, dict[str, TypeProfile]]  # domain -> task type -> profile

    def get(self, domain: str, task_type: str) -> TypeProfile | None:
        return self.types.get(domain, {}).get(task_type)

    def to_json(self) -> str:
        return json.dumps(
            {
                "meta": self.meta,
                "types": {d: {t: asdict(p) for t, p in ts.items()} for d, ts in self.types.items()},
            },
            indent=1,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Profile:
        raw = json.loads(text)
        types = {d: {t: TypeProfile(**p) for t, p in ts.items()} for d, ts in raw["types"].items()}
        return cls(meta=raw["meta"], types=types)

    @classmethod
    def load(cls, path: Path) -> Profile:
        return cls.from_json(Path(path).read_text())

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json() + "\n")
