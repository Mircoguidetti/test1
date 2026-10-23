"""Run one tau-bench task with the baseline agent or with the layer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from tau_bench.types import Action

from costlayer.classifier import classify
from costlayer.domains import TaskRef, anthropic_tools, make_env
from costlayer.layer import Controller
from costlayer.llm import LLM, Response
from costlayer.pricing import Usage, est_tokens
from costlayer.profile import Profile
from costlayer.user import AnthropicUser

MAX_STEPS = 30


@dataclass
class StepLog:
    step: int
    kind: str  # "tool" | "respond" | "redo"
    tools: list[str]
    usage: dict[str, int]
    cost: float
    stop_reason: str
    system_tokens_est: int
    tools_tokens_est: int
    messages_tokens_est: int
    applied: bool
    removed_tokens: dict[str, int]
    fallback: bool
    t: float


@dataclass
class RunResult:
    run_id: str
    domain: str
    idx: int
    mode: str
    agent_model: str
    reward: float
    done: bool
    task_type: str
    layer_applied: bool
    fallback: str | None
    compacted: bool
    trimmed: int
    steps: list[StepLog]
    usage: Usage
    cost_agent: float
    cost_classifier: float
    cost_user: float
    cost_label: float = 0.0  # classifier call used only to label a baseline training trace
    error: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def cost_product(self) -> float:
        """What a customer pays for the agent: agent calls plus the layer's classifier."""
        return self.cost_agent + self.cost_classifier

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "idx": self.idx,
            "mode": self.mode,
            "model": self.agent_model,
            "reward": self.reward,
            "done": self.done,
            "task_type": self.task_type,
            "layer_applied": self.layer_applied,
            "fallback": self.fallback,
            "compacted": self.compacted,
            "trimmed": self.trimmed,
            "n_steps": len(self.steps),
            "tokens": self.usage.as_dict(),
            "cost_agent": round(self.cost_agent, 6),
            "cost_classifier": round(self.cost_classifier, 6),
            "cost_user": round(self.cost_user, 6),
            "cost_label": round(self.cost_label, 6),
            "error": self.error,
        }


def _tool_calls(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [b for b in content if b.get("type") == "tool_use"]


def _text(content: list[dict[str, Any]]) -> str:
    return "\n".join(b["text"] for b in content if b.get("type") == "text").strip()


def run_task(
    ref: TaskRef,
    mode: str,
    llm: LLM,
    agent_model: str,
    profile: Profile | None = None,
    user_model: str | None = None,
    compact_override: int | None = None,
    max_steps: int = MAX_STEPS,
    label_baseline: bool = False,
) -> RunResult:
    assert mode in ("baseline", "layer")
    run_id = f"{ref.key}-{mode}"
    user = AnthropicUser(llm, run_id, **({"model": user_model} if user_model else {}))
    env = make_env(ref, user)
    first = env.reset(task_index=ref.idx).observation
    tools = anthropic_tools(env)
    ctl = Controller(
        domain=ref.domain,
        wiki=env.wiki,
        tools=tools,
        agent_model=agent_model,
        run_id=run_id,
        compact_override=compact_override,
    )
    cost_classifier = 0.0
    if mode == "layer":
        assert profile is not None
        ctl.task_type, cost_classifier = classify(llm, ref.domain, run_id, first)
        ctl.profile = profile.get(ref.domain, ctl.task_type)

    cost_label = 0.0
    if mode == "baseline" and label_baseline:
        ctl.task_type, cost_label = classify(llm, ref.domain, run_id, first)

    messages: list[dict[str, Any]] = [{"role": "user", "content": first}]
    steps: list[StepLog] = []
    total = Usage()
    cost_agent = 0.0
    reward, done, error = 0.0, False, None
    t0 = time.time()

    def call(kind: str, step_i: int) -> Response:
        nonlocal total, cost_agent, messages
        messages = ctl.maybe_compact(messages)
        req, meta = ctl.build_request(messages)
        resp = llm.create(req)
        total = total + resp.usage
        cost_agent += resp.cost
        steps.append(
            StepLog(
                step=step_i,
                kind=kind,
                tools=[c["name"] for c in _tool_calls(resp.content)],
                usage=resp.usage.as_dict(),
                cost=resp.cost,
                stop_reason=resp.stop_reason,
                system_tokens_est=est_tokens(req.system),
                tools_tokens_est=est_tokens(json.dumps(req.tools)),
                messages_tokens_est=est_tokens(json.dumps(req.messages)),
                applied=meta.applied,
                removed_tokens=meta.removed_tokens,
                fallback=meta.fallback,
                t=round(time.time() - t0, 2),
            )
        )
        return resp

    try:
        for step_i in range(max_steps):
            resp = call("tool", step_i)
            reason = ctl.inspect(resp.content, resp.stop_reason)
            if reason:  # leave the learned path: discard this answer, redo with the baseline
                ctl.fall_back(reason)
                steps[-1].kind = "redo"
                resp = call("tool", step_i)
            if resp.stop_reason == "refusal" or not resp.content:
                error = f"stop_reason={resp.stop_reason}"
                break
            messages.append({"role": "assistant", "content": resp.content})
            calls = _tool_calls(resp.content)
            if calls:
                execute, skipped = ctl.select_calls(calls)
                results: list[dict[str, Any]] = []
                for c in calls:
                    if c["id"] in skipped:
                        results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": c["id"],
                                "content": skipped[c["id"]],
                                "is_error": True,
                            }
                        )
                        continue
                    r = env.step(Action(name=c["name"], kwargs=c["input"]))
                    reward, done = r.reward, r.done
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": c["id"],
                            "content": ctl.process_result(c, r.observation),
                        }
                    )
                    if done:
                        break
                if done:
                    break
                messages.append({"role": "user", "content": results})
            else:
                r = env.step(
                    Action(
                        name="respond", kwargs={"content": _text(resp.content) or "(no response)"}
                    )
                )
                reward, done = r.reward, r.done
                if done:
                    break
                messages.append({"role": "user", "content": r.observation})
            ctl.note_step()
    except Exception as e:  # keep the batch alive; BudgetExceeded is re-raised by the caller
        from costlayer.ledger import BudgetExceeded

        if isinstance(e, BudgetExceeded):
            raise
        error = f"{type(e).__name__}: {e}"[:300]

    return RunResult(
        run_id=run_id,
        domain=ref.domain,
        idx=ref.idx,
        mode=mode,
        agent_model=agent_model,
        reward=reward if done else 0.0,
        done=done,
        task_type=ctl.task_type,
        layer_applied=mode == "layer" and ctl.profile is not None,
        fallback=ctl.fallback_reason,
        compacted=ctl.compacted,
        trimmed=len(ctl.trimmed_calls),
        steps=steps,
        usage=total,
        cost_agent=cost_agent,
        cost_classifier=cost_classifier,
        cost_user=user.cost,
        cost_label=cost_label,
        error=error,
        messages=messages,
    )


__all__ = ["RunResult", "StepLog", "run_task"]
