#!/usr/bin/env python3
"""
run_all_strategies.py — 更新版
==============================
批量執行所有已註冊策略嘅 optimize + wfo + backtest。
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from strategies import AVAILABLE_STRATEGIES

DEFAULT_STRATEGIES = AVAILABLE_STRATEGIES.copy()
ALLOWED_STRATEGIES = set(AVAILABLE_STRATEGIES)


def run_cmd(cmd: list):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(cmd))
    print("=" * 100)
    p = subprocess.run(cmd, text=True)
    return p.returncode


def safe_load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def extract_summary_for_strategy(strategy: str):
    base = Path("results") / strategy
    best_params = safe_load_json(base / "best_params.json")
    wfo = safe_load_json(base / "walk_forward_result.json")
    bt_metrics = safe_load_json(base / "backtest_metrics.json")

    row = {
        "strategy": strategy,
        "best_metric_insample": best_params.get("best_metric_insample"),
        "consistency_adjusted_score": best_params.get(
            "consistency_adjusted_score"
        ),
        "oos_sharpe_single": (
            best_params.get("oos_metrics", {}).get("sharpe")
            if isinstance(best_params.get("oos_metrics"), dict)
            else None
        ),
        "bt_sharpe": bt_metrics.get("sharpe"),
        "bt_total_return": bt_metrics.get("total_return"),
        "bt_max_drawdown": bt_metrics.get("max_drawdown"),
        "bt_win_rate": bt_metrics.get("win_rate"),
        "wfo_n_folds": wfo.get("n_folds"),
        "wfo_total_return": (
            wfo.get("overall_oos_metrics", {}).get("total_return")
            if isinstance(wfo.get("overall_oos_metrics"), dict)
            else None
        ),
        "wfo_sharpe": (
            wfo.get("overall_oos_metrics", {}).get("sharpe")
            if isinstance(wfo.get("overall_oos_metrics"), dict)
            else None
        ),
        "wfo_max_drawdown": (
            wfo.get("overall_oos_metrics", {}).get("max_drawdown")
            if isinstance(wfo.get("overall_oos_metrics"), dict)
            else None
        ),
        "final_capital": wfo.get("final_capital"),
        "best_params": best_params.get("best_params"),
    }
    return row


def save_summary(rows, tag: str = ""):
    summary_dir = Path("results/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""

    json_path = summary_dir / f"run_summary{suffix}_{ts}.json"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "strategies": AVAILABLE_STRATEGIES,
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    latest_json = summary_dir / "run_summary_latest.json"
    latest_json.write_text(
        json_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print(f"\n✅ Summary: {json_path}")
    print(f"✅ Latest:  {latest_json}")


def parse_strategies(raw: str):
    raw = (raw or "").strip().lower()
    if raw == "all":
        return DEFAULT_STRATEGIES.copy()

    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not items:
        return DEFAULT_STRATEGIES.copy()

    invalid = [x for x in items if x not in ALLOWED_STRATEGIES]
    if invalid:
        raise ValueError(
            f"不支援策略: {invalid} | 可用: {sorted(ALLOWED_STRATEGIES)}"
        )
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", type=str, default="all")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--backtest-start", type=str, default="2020-01-01")
    parser.add_argument("--trials-opt", type=int, default=30)
    parser.add_argument("--trials-wfo", type=int, default=20)
    parser.add_argument("--metric", type=str, default="robust_sharpe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-optimize", action="store_true")
    parser.add_argument("--skip-wfo", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()

    strategies = parse_strategies(args.strategies)

    if not strategies:
        print("⚠️ 無可用策略。")
        return

    print(f"🎯 本次策略: {strategies}")
    fail_map = {}

    for s in strategies:
        fail_map[s] = []

        if not args.skip_optimize:
            rc = run_cmd([
                "python", "scripts/run_experiment.py",
                "--mode", "optimize",
                "--strategy", s,
                "--metric", args.metric,
                "--trials", str(args.trials_opt),
                "--start", args.start,
                "--end", args.end,
                "--seed", str(args.seed),
            ])
            if rc != 0:
                fail_map[s].append("optimize")

        if not args.skip_wfo:
            rc = run_cmd([
                "python", "scripts/run_experiment.py",
                "--mode", "wfo",
                "--strategy", s,
                "--metric", args.metric,
                "--trials", str(args.trials_wfo),
                "--start", args.start,
                "--end", args.end,
                "--seed", str(args.seed),
            ])
            if rc != 0:
                fail_map[s].append("wfo")

        if not args.skip_backtest:
            rc = run_cmd([
                "python", "scripts/run_experiment.py",
                "--mode", "backtest",
                "--strategy", s,
                "--start", args.backtest_start,
                "--end", args.end,
                "--seed", str(args.seed),
            ])
            if rc != 0:
                fail_map[s].append("backtest")

    rows = [extract_summary_for_strategy(s) for s in strategies]
    save_summary(rows)

    print("\n📌 結果摘要:")
    any_fail = False
    for s in strategies:
        if fail_map[s]:
            any_fail = True
            print(f"  ❌ {s}: failed = {fail_map[s]}")
        else:
            print(f"  ✅ {s}: 全部通過")

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()