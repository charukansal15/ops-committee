"""Run a compact hero trajectory for the hackathon pitch."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.committee import run_rule_based_committee


def main() -> None:
    result = run_rule_based_committee(level=3, seed=101, max_steps=20)
    print(
        json.dumps(
            {
                "policy": result["policy"],
                "level": result["level"],
                "seed": result["seed"],
                "terminal_reason": result["terminal_reason"],
                "steps": result["steps"],
                "total_reward": result["total_reward"],
            },
            indent=2,
        )
    )
    print("\nTimeline:")
    for index, item in enumerate(result["trajectory"]):
        action = item.get("action")
        observation = item["observation"]
        if action:
            actor = action.get("actor")
            tool = action.get("tool")
            decision = action.get("decision")
            print(f"{index:02d}. {actor} -> {tool} {decision or ''}".rstrip())
        print(f"    {observation['message']}")
        feedback = observation.get("pending_feedback") or []
        for note in feedback[:2]:
            print(f"    {note}")


if __name__ == "__main__":
    main()
