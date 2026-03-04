#!/usr/bin/env python3
"""
run_experiment.py — 更新版
==========================
整合新策略 (Trend Alpha, Mean Reversion) 到實驗流程。
"""
import os
import sys
import json
import argparse
import random
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from layers.data_hub import DataHub
from backtest.universal_backtester import (
    UniversalBacktester,
    TransactionCostModel,
    PerformanceAnalyzer,
)
from backtest.optimizer import StrategyOptimizer

# ─── 新策略註冊 ───
from strategies import (
    get_strategy_class,
    get_strategy_param_space,
    AVAILABLE_STRATEGIES,
)


OPTIMIZE_UNIVERSE = [
    "SPY", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "JPM", "UNH", "V", "MA", "JNJ", "PG", "WMT",
    "PANW", "CRWD", "DDOG", "ZS", "FTNT", "NET",
    "ON", "MCHP", "SWKS",
    "DHI", "LEN", "PHM",
    "DECK", "LULU", "BURL",
    "BAC", "GS", "MS", "SCHW", "C",
    "XOM", "CVX", "COP", "SLB",
    "LLY", "ABBV", "MRK", "TMO", "ISRG",
    "CAT", "DE", "GE", "HON",
]


def _get_git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return ""


def _save_run_metadata(output_dir: Path, args, cfg: dict, universe: list):
    meta = {
        "mode": args.mode,
        "strategy": args.strategy,
        "start": args.start,
        "end": args.end,
        "trials": args.trials,
        "metric": args.metric,
        "seed": args.seed,
        "git_hash": _get_git_hash(),
        "universe_size": len(universe),
        "universe": universe,
        "config_snapshot": cfg,
        "available_strategies": AVAILABLE_STRATEGIES,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def run():
    parser = argparse.ArgumentParser(
        description="量化策略實驗平台",
        epilog=f"可用策略: {', '.join(AVAILABLE_STRATEGIES)}",
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "optimize", "wfo"],
        required=True,
    )
    parser.add_argument("--strategy", type=str, required=True)
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--metric", type=str, default="robust_sharpe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--execution-delay",
        type=int,
        default=1,
        help="訂單延遲天數 (default=1, 更真實)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = (
        load_config("config.yaml")
        if os.path.exists("config.yaml")
        else load_config("config.json")
    )

    print(f"📦 正在準備數據... (策略: {args.strategy})")
    hub = DataHub(cfg)
    prices_dict = hub.load_price_dict()

    if args.mode in ["optimize", "wfo"]:
        prices_dict = {
            k: v
            for k, v in prices_dict.items()
            if k in OPTIMIZE_UNIVERSE and not v.empty
        }
        print(
            f"⚡ 優化模式啟動，股票池: {len(prices_dict)} 隻"
        )

    if not prices_dict:
        print("❌ 載入數據失敗")
        return

    StrategyClass = get_strategy_class(args.strategy)
    output_dir = Path(f"results/{args.strategy}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_run_metadata(
        output_dir=output_dir,
        args=args,
        cfg=cfg,
        universe=sorted(list(prices_dict.keys())),
    )

    cost_model = TransactionCostModel(
        commission_rate=0.001, slippage_bps=10.0
    )

    if args.mode == "backtest":
        params_file = output_dir / "best_params.json"
        params = {}
        if params_file.exists():
            try:
                with open(params_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                params = saved.get("best_params", {})
                print(f"🔍 讀取到優化參數: {params}")
            except Exception:
                print("⚠️ 無法讀取 best_params.json，使用預設參數")

        try:
            strategy = StrategyClass(**params)
        except TypeError as e:
            print(f"⚠️ 參數唔相容: {e}\n   改用預設參數")
            strategy = StrategyClass()

        backtester = UniversalBacktester(
            initial_capital=100_000.0,
            cost_model=cost_model,
            execution_delay=args.execution_delay,
        )

        equity_df = backtester.run(
            strategy=strategy,
            data_dict=prices_dict,
            start_date=args.start,
            end_date=args.end,
        )

        if equity_df is not None and not equity_df.empty:
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.analyze(
                equity_df, trade_log=backtester.trade_log
            )
            print("\n🏆 回測結果")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k.ljust(18)}: {v:.4f}")
                else:
                    print(f"  {k.ljust(18)}: {v}")

            equity_df.to_csv(output_dir / "backtest_equity.csv")
            backtester.trade_log.to_csv(
                output_dir / "backtest_trades.csv", index=False
            )

            # 同時儲存 metrics
            with (output_dir / "backtest_metrics.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(metrics, f, indent=2, default=str)

            print(f"\n💾 回測結果已儲存至 {output_dir}")

    elif args.mode in ["optimize", "wfo"]:
        param_space = get_strategy_param_space(args.strategy)

        if not param_space:
            print(
                f"⚠️ 策略 {args.strategy} 無定義參數空間，用預設參數 backtest"
            )
            return

        optimizer = StrategyOptimizer(
            strategy_class=StrategyClass,
            param_space=param_space,
            prices_dict=prices_dict,
            start_date=args.start,
            end_date=args.end,
            metric=args.metric,
            n_trials=args.trials,
            cost_model=cost_model,
            optuna_timeout_sec=1800,
        )

        if args.mode == "optimize":
            start_dt = pd.to_datetime(args.start)
            end_dt = pd.to_datetime(args.end)
            split_dt = start_dt + pd.Timedelta(
                days=int((end_dt - start_dt).days * 0.75)
            )
            train_end = split_dt.strftime("%Y-%m-%d")
            test_start = (
                split_dt + pd.Timedelta(days=1)
            ).strftime("%Y-%m-%d")

            result = optimizer.optimize_single_period(
                train_start=args.start,
                train_end=train_end,
                test_start=test_start,
                test_end=args.end,
                save_best_to=output_dir / "best_params.json",
            )
            print(f"\n✅ 最佳參數: {result.get('best_params')}")
            print(
                f"📈 IS Metric ({args.metric}): "
                f"{result.get('best_metric_insample', 0):.4f}"
            )
            print(
                f"⚖️ Consistency Score: "
                f"{result.get('consistency_adjusted_score', 0):.4f}"
            )
            print(f"📉 OOS Metrics: {result.get('oos_metrics')}")

        elif args.mode == "wfo":
            print("🔄 開始 Walk-Forward Optimization (WFO)...")
            result = optimizer.optimize_walk_forward(
                window_train_years=2,
                window_test_years=1,
                step_years=1,
                save_dir=output_dir,
            )
            print(f"\n✅ WFO 完成！結果已儲存至 {output_dir}")

            if "overall_oos_metrics" in result:
                print("\n📊 OOS 綜合成績:")
                for k, v in result["overall_oos_metrics"].items():
                    if isinstance(v, float):
                        print(f"  {k.ljust(18)}: {v:.4f}")


if __name__ == "__main__":
    run()