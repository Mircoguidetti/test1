"""Prices per million tokens and cost computation."""

from __future__ import annotations

from dataclasses import dataclass

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_write: float  # 5 minute TTL
    cache_read: float


# Sonnet cache read: CLAUDE.md lists $0.10, but 0.1x of the $2 input price is $0.20 and the
# Anthropic model table says $0.20. We use the higher number: it can only make the cap
# stricter and the savings estimates more conservative. Verify before the first Sonnet run.
PRICES: dict[str, Price] = {
    SONNET: Price(input=2.0, output=10.0, cache_write=2.5, cache_read=0.20),
    HAIKU: Price(input=0.10, output=0.50, cache_write=0.125, cache_read=0.01),
}
# Haiku 5.5 prompts above 100K tokens are priced higher.
HAIKU_LONG = Price(input=0.50, output=2.50, cache_write=0.625, cache_read=0.05)
LONG_PROMPT_TOKENS = 100_000


@dataclass
class Usage:
    input_tokens: int = 0  # fresh (uncached) input
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0  # thinking included

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.output_tokens + other.output_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input": self.input_tokens,
            "cache_write": self.cache_write_tokens,
            "cache_read": self.cache_read_tokens,
            "output": self.output_tokens,
        }


def price_for(model: str, prompt_tokens: int = 0) -> Price:
    if model not in PRICES:
        raise KeyError(f"No price for model {model!r}")
    if model == HAIKU and prompt_tokens > LONG_PROMPT_TOKENS:
        return HAIKU_LONG
    return PRICES[model]


def cost_usd(model: str, usage: Usage) -> float:
    p = price_for(model, usage.prompt_tokens)
    return (
        usage.input_tokens * p.input
        + usage.cache_write_tokens * p.cache_write
        + usage.cache_read_tokens * p.cache_read
        + usage.output_tokens * p.output
    ) / 1e6


def est_tokens(text: str) -> int:
    """Rough token estimate (about 3.5 characters per token)."""
    return -(-len(text) * 2 // 7)


def worst_case_cost(model: str, prompt_chars: int, max_tokens: int) -> float:
    """Conservative pre-call estimate: no cache discount, 2.5 chars/token, full max_tokens out."""
    prompt_tokens = prompt_chars * 2 // 5 + 1
    p = price_for(model, prompt_tokens)
    return (prompt_tokens * max(p.input, p.cache_write) + max_tokens * p.output) / 1e6
