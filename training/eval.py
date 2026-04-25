"""Evaluate baseline committees and emit judge-facing artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.committee import run_random_committee, run_rule_based_committee


def _write_png(path: Path, width: int, height: int, pixels: list[list[tuple[int, int, int]]]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in pixels)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _draw_line(
    pixels: list[list[tuple[int, int, int]]],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    x, y = x1, y1
    while True:
        if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
            pixels[y][x] = color
        if x == x2 and y == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "G": ("01111", "10000", "10000", "10011", "10001", "10001", "01111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


def _draw_text(
    pixels: list[list[tuple[int, int, int]]],
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: int = 2,
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT.get(char, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "0":
                    continue
                for sy in range(scale):
                    for sx in range(scale):
                        px = cursor + gx * scale + sx
                        py = y + gy * scale + sy
                        if 0 <= py < len(pixels) and 0 <= px < len(pixels[0]):
                            pixels[py][px] = color
        cursor += (len(glyph[0]) + 1) * scale


def _plot_reward_comparison(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 900, 520
    margin = 70
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    for x in range(margin, width - margin):
        pixels[height - margin][x] = (30, 30, 30)
    for y in range(margin, height - margin):
        pixels[y][margin] = (30, 30, 30)
    _draw_text(pixels, width // 2 - 45, height - 40, "LEVEL", (20, 20, 20), scale=2)
    _draw_text(pixels, 12, 34, "REWARD", (20, 20, 20), scale=2)
    _draw_text(pixels, width - 280, 35, "GREEN=COMMITTEE", (45, 120, 80), scale=2)
    _draw_text(pixels, width - 230, 60, "RED=RANDOM", (190, 60, 55), scale=2)

    levels = sorted({int(row["level"]) for row in rows})
    policies = ["random_approve_all", "rule_based_committee"]
    rewards = [float(row["total_reward"]) for row in rows]
    lo, hi = min(rewards), max(rewards)
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    colors = {
        "random_approve_all": (190, 60, 55),
        "rule_based_committee": (45, 120, 80),
    }
    for policy in policies:
        points: list[tuple[int, int]] = []
        for level in levels:
            policy_rewards = [
                float(row["total_reward"])
                for row in rows
                if row["policy"] == policy and int(row["level"]) == level
            ]
            avg = sum(policy_rewards) / len(policy_rewards)
            x = margin + int((level - min(levels)) * (width - 2 * margin) / max(1, max(levels) - min(levels)))
            y = height - margin - int((avg - lo) * (height - 2 * margin) / (hi - lo))
            points.append((x, y))
            for yy in range(y - 4, y + 5):
                for xx in range(x - 4, x + 5):
                    if 0 <= yy < height and 0 <= xx < width:
                        pixels[yy][xx] = colors[policy]
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            _draw_line(pixels, x1, y1, x2, y2, colors[policy])

    _write_png(path, width, height, pixels)


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
