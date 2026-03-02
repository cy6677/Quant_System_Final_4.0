#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from layers.data_hub import DataHub
from backtest.universal_backtester import UniversalBacktester, PerformanceAnalyzer

from strategies.strategy_a import StrategyA_VCPBreakout
from strategies.strategy_stat_arb import StrategyD_StatArb
from strategies.strategy_long_term import LongTermStrategy


def run_meta_backtest(
    start_date: str = "2018-01-01", end_date: str = "2023-12-31"
):
    print(
        "🚀 啟動 機構級動態倉位分配器 (Regime-Switching Meta-Allocator)..."
    )

    cfg = (
        load_config("config.yaml") if os.path.exists("config.yaml") else {}
    )
    hub = DataHub(cfg)
    prices_dict = hub.load_price_dict()
    if not prices_dict:
        print("❌ 無法載入價格數據")
        return

    benchmark = cfg.get("benchmark", "SPY")
    has_benchmark = benchmark in prices_dict
    if not has_benchmark:
        print(f"⚠️ 找不到基準 {benchmark}，將退回靜態平均分配。")
    else:
        spy_df = prices_dict[benchmark]

    # ✅ FIX: 用 data_dict= 對齊 backtester 簽名
    print("\n🏃‍♂️ 正在獨立回測 Strategy A (VCP Breakout)...")
    strat_a = StrategyA_VCPBreakout()
    bt_a = UniversalBacktester(initial_capital=100000.0)
    equity_a = bt_a.run(strat_a, data_dict=prices_dict, start_date=start_date, end_date=end_date)

    print("🏃‍♂️ 正在獨立回測 Strategy B (MeanReversion)...")
    strat_b = StrategyD_StatArb()
    bt_b = UniversalBacktester(initial_capital=100000.0)
    equity_b = bt_b.run(strat_b, data_dict=prices_dict, start_date=start_date, end_date=end_date)

    print("🏃‍♂️ 正在獨立回測 Strategy C (LongTerm)...")
    strat_c = LongTermStrategy()
    bt_c = UniversalBacktester(initial_capital=100000.0)
    equity_c = bt_c.run(strat_c, data_dict=prices_dict, start_date=start_date, end_date=end_date)

    if (
        equity_a is None
        or equity_a.empty
        or equity_b is None
        or equity_b.empty
        or equity_c is None
        or equity_c.empty
    ):
        print("❌ 回測失敗，部分資金曲線為空")
        return

    df_returns = pd.DataFrame(index=equity_a.index)
    df_returns["ret_A"] = equity_a["equity"].pct_change().fillna(0)
    df_returns["ret_B"] = equity_b["equity"].pct_change().fillna(0)
    df_returns["ret_C"] = equity_c["equity"].pct_change().fillna(0)

    # 預設權重
    df_returns["weight_A"] = 0.33
    df_returns["weight_B"] = 0.33
    df_returns["weight_C"] = 0.34

    if has_benchmark:
        # ✅ FIX: 用 shift(1) 嘅 Close 計算 200MA，避免前瞻偏差
        spy_close = spy_df["Close"].reindex(df_returns.index).ffill()
        spy_close_prev = spy_close.shift(1)
        spy_200ma = spy_close_prev.rolling(200).mean()

        # Hysteresis Regime Detection
        raw_signal = (spy_close_prev > spy_200ma).fillna(False).astype(int)
        smoothed_signal = raw_signal.rolling(5).mean()
        is_bull = smoothed_signal >= 0.8
        is_bear = smoothed_signal <= 0.2

        # 牛市
        df_returns.loc[is_bull, "weight_A"] = 0.40
        df_returns.loc[is_bull, "weight_B"] = 0.20
        df_returns.loc[is_bull, "weight_C"] = 0.40

        # 熊市
        df_returns.loc[is_bear, "weight_A"] = 0.10
        df_returns.loc[is_bear, "weight_B"] = 0.60
        df_returns.loc[is_bear, "weight_C"] = 0.30

        # 中間地帶 → ffill
        neutral_mask = ~is_bull & ~is_bear
        for col in ["weight_A", "weight_B", "weight_C"]:
            df_returns.loc[neutral_mask, col] = np.nan
            df_returns[col] = df_returns[col].ffill().fillna(0.33)

        print("✅ 已啟用「帶 Hysteresis 嘅 Regime-Switching」機制！")

    # 計算混合資金曲線
    df_returns["meta_return"] = (
        (df_returns["ret_A"] * df_returns["weight_A"])
        + (df_returns["ret_B"] * df_returns["weight_B"])
        + (df_returns["ret_C"] * df_returns["weight_C"])
    )

    initial_capital = 100000.0
    df_returns["equity"] = (
        initial_capital * (1 + df_returns["meta_return"]).cumprod()
    )

    print("\n📊 Meta-Allocator 綜合績效報告:")
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze(df_returns[["equity"]])

    for k, v in metrics.items():
        if "return" in k or "drawdown" in k or "rate" in k:
            print(f"  {k.ljust(20)}: {v * 100:>8.2f}%")
        else:
            print(f"  {k.ljust(20)}: {v:>8.4f}")

    print("-" * 45)
    print(
        f"💰 最終資金 (由 10 萬起步): ${df_returns['equity'].iloc[-1]:,.2f}"
    )

    # 繪圖
    try:
        import matplotlib.pyplot as plt

        print("\n📈 正在繪製終極資金曲線圖 (Equity Curve)...")
        plt.figure(figsize=(14, 8))

        plt.plot(
            df_returns.index,
            (1 + df_returns["ret_A"]).cumprod() * 100000,
            label="Strategy A (VCP)",
            alpha=0.4,
            linestyle="--",
        )
        plt.plot(
            df_returns.index,
            (1 + df_returns["ret_B"]).cumprod() * 100000,
            label="Strategy B (StatArb)",
            alpha=0.4,
            linestyle="--",
        )
        plt.plot(
            df_returns.index,
            (1 + df_returns["ret_C"]).cumprod() * 100000,
            label="Strategy C (LongTerm)",
            alpha=0.4,
            linestyle="--",
        )

        plt.plot(
            df_returns.index,
            df_returns["equity"],
            label="🔥 Meta-Portfolio (All-in-One)",
            color="red",
            linewidth=2.5,
        )

        if has_benchmark:
            bear_mask = is_bear
            y_min, y_max = plt.ylim()
            plt.fill_between(
                df_returns.index,
                y_min,
                y_max,
                where=bear_mask,
                color="grey",
                alpha=0.15,
                label="Bear Market (SPY < 200MA)",
            )

        plt.title(
            "Institutional Quant System - Meta Allocator Equity Curve",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Portfolio Value (USD)", fontsize=12)
        plt.legend(fontsize=12, loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        pic_name = "meta_portfolio_equity.png"
        plt.savefig(pic_name, dpi=300)
        print(f"✅ 圖表已成功儲存為 {pic_name}！")
        plt.show()
    except ImportError:
        print("⚠️ 未安裝 matplotlib，跳過繪圖步驟。")


if __name__ == "__main__":
    run_meta_backtest()