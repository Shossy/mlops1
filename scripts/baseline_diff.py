"""Append Markdown baseline vs current metrics comparison for CML reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: baseline_diff.py <current_metrics.json> <baseline_metrics.json>",
            file=sys.stderr,
        )
        return 2
    cur_path = Path(sys.argv[1])
    base_path = Path(sys.argv[2])
    if not cur_path.is_file() or not base_path.is_file():
        return 0
    cur = json.loads(cur_path.read_text(encoding="utf-8"))
    base = json.loads(base_path.read_text(encoding="utf-8"))
    keys = sorted(
        set(cur)
        & set(base)
        & {"test_r2", "test_rmse", "test_mae", "test_rmse_inr", "test_mae_inr"}
    )
    if not keys:
        return 0
    print("\n### Baseline comparison (main)\n")
    print("| Metric | Baseline | Current | Delta |")
    print("| --- | ---: | ---: | ---: |")
    for k in keys:
        try:
            b = float(base[k])
            c = float(cur[k])
            d = c - b
            print(f"| {k} | {b:.5g} | {c:.5g} | {d:+.5g} |")
        except (TypeError, ValueError):
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
