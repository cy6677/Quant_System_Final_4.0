#!/usr/bin/env python3
"""
daily_screener.py
=================
每日掃描 — 搵今日有信號嘅股票。
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

from config import load_config
from layers.data_hub import DataHub
from strategies.base import calc_rsi, calc_zscore, calc_momentum_score, calc_sma


def run():
    cfg = load_config()
    hub = DataHub(cfg)

    print("📡 每日篩選器")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    prices = hub.load_price_dict()
    if not prices:
        print("❌ 無數據")
        return

    # ── Trend Signals ──
    print("\n🔵 趨勢動量信號 (Trend Alpha)")
    print("-" * 60)

    trend_candidates = []
    for ticker, df in prices.items():
        if ticker == "SPY" or df.empty or len(df) < 252:
            continue

        close = df["Close"].astype(float)
        price = float(close.iloc[-1])

        # SMA-200 filter
        sma200 = float(close.iloc[-200:].mean())
        if price < sma200:
            continue

        # Multi-horizon momentum
        scores = []
        for lb in [63, 126, 252]:
            if len(close) >= lb + 10:
                s = calc_momentum_score(close, lb)
                if np.isfinite(s):
                    scores.append(s)

        if not scores:
            continue

        avg_score = np.mean(scores)
        if avg_score > 0.3:
            trend_candidates.append({
                "ticker": ticker,
                "price": price,
                "momentum": round(avg_score, 3),
                "vs_sma200": f"{(price / sma200 - 1) * 100:+.1f}%",
            })

    trend_candidates.sort(key=lambda x: x["momentum"], reverse=True)
    for c in trend_candidates[:15]:
        print(
            f"  📈 {c['ticker'].ljust(6)} | "
            f"${c['price']:>8.2f} | "
            f"Mom: {c['momentum']:>6.3f} | "
            f"vs SMA200: {c['vs_sma200']}"
        )

    # ── Mean Reversion Signals ──
    print(f"\n🔴 均值回歸信號 (Mean Reversion)")
    print("-" * 60)

    mr_candidates = []
    for ticker, df in prices.items():
        if ticker == "SPY" or df.empty or len(df) < 60:
            continue

        close = df["Close"].astype(float)
        price = float(close.iloc[-1])

        # RSI(2)
        rsi = calc_rsi(close, 2)
        rsi_val = float(rsi.iloc[-1]) if len(rsi) > 0 and np.isfinite(rsi.iloc[-1]) else 50

        if rsi_val > 15:
            continue  # 只要極端超賣

        # Z-score
        ret = close.pct_change()
        zs = calc_zscore(ret, 20)
        zs_val = float(zs.iloc[-1]) if len(zs) > 0 and np.isfinite(zs.iloc[-1]) else 0

        # SMA-200 check (唔好接 falling knife)
        if len(close) >= 200:
            sma200 = float(close.iloc[-200:].mean())
            if price < sma200 * 0.85:
                continue

        mr_candidates.append({
            "ticker": ticker,
            "price": price,
            "rsi2": round(rsi_val, 1),
            "zscore": round(zs_val, 2),
        })

    mr_candidates.sort(key=lambda x: x["rsi2"])
    for c in mr_candidates[:10]:
        print(
            f"  📉 {c['ticker'].ljust(6)} | "
            f"${c['price']:>8.2f} | "
            f"RSI(2): {c['rsi2']:>5.1f} | "
            f"Z: {c['zscore']:>6.2f}"
        )

    # ── Market Overview ──
    spy = prices.get("SPY")
    if spy is not None and not spy.empty and len(spy) > 252:
        close = spy["Close"].astype(float)
        spy_price = float(close.iloc[-1])
        sma50 = float(close.iloc[-50:].mean())
        sma200 = float(close.iloc[-200:].mean())
        ret_21d = float(close.iloc[-1] / close.iloc[-21] - 1)
        vol = float(close.pct_change().iloc[-21:].std() * np.sqrt(252))

        # Breadth
        above_200 = sum(
            1 for t, d in prices.items()
            if t != "SPY" and not d.empty and len(d) >= 200
            and float(d["Close"].iloc[-1]) > float(d["Close"].iloc[-200:].mean())
        )
        total = sum(
            1 for t, d in prices.items()
            if t != "SPY" and not d.empty and len(d) >= 200
        )
        breadth = above_200 / total if total > 0 else 0

        print(f"\n📊 市場概覽 (SPY)")
        print("-" * 60)
        print(f"  價格:       ${spy_price:.2f}")
        print(f"  vs SMA-50:  {(spy_price / sma50 - 1) * 100:+.1f}%")
        print(f"  vs SMA-200: {(spy_price / sma200 - 1) * 100:+.1f}%")
        print(f"  21d Return: {ret_21d * 100:+.1f}%")
        print(f"  21d Vol:    {vol * 100:.1f}%")
        print(f"  Breadth:    {breadth * 100:.0f}% above SMA-200")

        if spy_price > sma200 and breadth > 0.5:
            print(f"  Regime:     🟢 Likely RISK_ON / STRONG_TREND")
        elif spy_price < sma200:
            print(f"  Regime:     🔴 Likely CRISIS / Downtrend")
        else:
            print(f"  Regime:     🟡 Likely RANGE")

    print(f"\n✅ 掃描完成")


if __name__ == "__main__":
    run()