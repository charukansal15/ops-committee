"""Colab-ready TRL scaffold for training against The Ops Committee.

This file is intentionally dependency-light in the repo. In Colab, install the
latest OpenEnv, TRL, Transformers, Accelerate, and optionally Unsloth, then use
the hooks below to connect a policy model to the environment's tool actions.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.committee import run_rule_based_committee


SYSTEM_PROMPT = """You are one member of The Ops Committee.
Read the observation, respect your role, and emit exactly one JSON action.
Fixer proposes operational tools. Banker audits cost. Shield audits policy.
"""


def format_episode_for_sft(level: int, seed: int) -> list[dict[str, str]]:
    """Create a small behavior-cloning trace from the rule-based committee."""

    run = run_rule_based_committee(level=level, seed=seed, max_steps=20)
    messages: list[dict[str, str]] = []
    for item in run["trajectory"]:
        if "action" not in item:
            continue
        observation = item["observation"]
        action = item["action"]
        messages.append(
            {
                "system": SYSTEM_PROMPT,
                "user": json.dumps(observation, sort_keys=True, default=str),
                "assistant": json.dumps(action, sort_keys=True, default=str),
            }
        )
    return messages


def build_sft_dataset(n_episodes: int = 200, seed_start: int = 1000) -> list[dict[str, str]]:
    """Build a multi-level SFT corpus from rule-based committee trajectories."""

    dataset: list[dict[str, str]] = []
    for index in range(n_episodes):
        level = (index % 3) + 1
        seed = seed_start + index
        dataset.extend(format_episode_for_sft(level=level, seed=seed))
    return dataset


def to_chatml_text(example: dict[str, str]) -> str:
    return (
        "<|system|>\n"
        f"{example['system']}\n"
        "<|user|>\n"
        f"{example['user']}\n"
        "<|assistant|>\n"
        f"{example['assistant']}"
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, str]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def reward_from_observation(observation: dict[str, Any]) -> float:
    """Scalar reward hook for TRL/GRPO style training loops."""

    reward = observation.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward)
    breakdown = observation.get("reward_breakdown", {})
    return float(breakdown.get("total", 0.0))


def main() -> None:
    dataset = build_sft_dataset(n_episodes=200)
    write_jsonl(ROOT / "artifacts" / "sft_dataset.jsonl", dataset)
    try:
        import trl  # type: ignore  # noqa: F401
    except Exception:
        print(
            "Built artifacts/sft_dataset.jsonl. Install TRL/OpenEnv in Colab, "
            "then run training/ops_committee_colab.ipynb for the concrete "
            "SFT training loop."
        )
        print(f"examples={len(dataset)}")
        print(json.dumps(dataset[:2], indent=2))
        return

    print(
        "Built artifacts/sft_dataset.jsonl. TRL is installed; run "
        "training/ops_committee_colab.ipynb or import build_sft_dataset(), "
        "to_chatml_text(), and reward_from_observation() from this module."
    )
    print(f"examples={len(dataset)}")


if __name__ == "__main__":
    main()
