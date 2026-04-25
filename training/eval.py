"""Evaluate baseline committees and emit judge-facing artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.committee import run_random_committee, run_rule_based_committee


def _plot_reward_comparison(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    policies = {
        "random_approve_all": ("Random Baseline", "crimson"),
        "rule_based_committee": ("Rule-Based Committee", "seagreen"),
    }
    levels = sorted({int(row["level"]) for row in rows})

    fig, ax = plt.subplots(figsize=(8, 5))
    for policy, (label, color) in policies.items():
        by_level: dict[int, list[float]] = {}
        for row in rows:
            if row["policy"] == policy:
                by_level.setdefault(int(row["level"]), []).append(
                    float(row["total_reward"])
                )
        xs = sorted(by_level)
        ys = [sum(by_level[x]) / len(by_level[x]) for x in xs]
        ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=2)

    ax.set_xticks(levels)
    ax.set_xlabel("Chaos Level")
    ax.set_ylabel("Average Episode Reward")
    ax.set_title("Rule-Based Committee vs Random Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eval() -> dict[str, object]:
    ARTIFACTS.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    hero_trajectory = None
    for level in (1, 2, 3):
        for seed in (101, 102, 103):
            for runner in (run_random_committee, run_rule_based_committee):
                result = runner(level=level, seed=seed, max_steps=20)
                if (
                    result["policy"] == "rule_based_committee"
                    and level == 3
                    and seed == 101
                ):
                    hero_trajectory = result
                rows.append(
                    {
                        "policy": result["policy"],
                        "level": level,
                        "seed": seed,
                        "terminal_reason": result["terminal_reason"],
                        "steps": result["steps"],
                        "total_reward": result["total_reward"],
                    }
                )

    summary: dict[str, dict[str, float]] = {}
    for policy in sorted({str(row["policy"]) for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "episodes": float(len(policy_rows)),
            "average_reward": round(
                sum(float(row["total_reward"]) for row in policy_rows) / len(policy_rows),
                3,
            ),
            "clean_recovery_rate": round(
                sum(row["terminal_reason"] == "recovered_cleanly" for row in policy_rows)
                / len(policy_rows),
                3,
            ),
            "failure_rate": round(
                sum(
                    row["terminal_reason"]
                    in {"budget_exhausted", "catastrophic_outage", "security_breach"}
                    for row in policy_rows
                )
                / len(policy_rows),
                3,
            ),
        }

    with (ARTIFACTS / "eval_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "runs": rows, "hero_trajectory": hero_trajectory},
            f,
            indent=2,
            default=str,
        )
    with (ARTIFACTS / "eval_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_reward_comparison(rows, ARTIFACTS / "reward_comparison.png")
    return {"summary": summary, "artifact_dir": str(ARTIFACTS)}


if __name__ == "__main__":
    print(json.dumps(run_eval(), indent=2))
