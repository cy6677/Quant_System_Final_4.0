#!/usr/bin/env python3
"""
Meta Allocator - 機構級動態倉位分配器

⚠️ 所有舊策略已清除。請實現新策略後重新整合。
"""
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


def run_meta_backtest(
    start_date: str = "2018-01-01", end_date: str = "2023-12-31"
):
    print(
        "🚀 啟動 機構級動態倉位分配器 (Regime-Switching Meta-Allocator)..."
    )
    print("⚠️ 所有舊策略已清除。請先實現新的 5-Engine 策略後再執行此腳本。")
    print("📋 需要註冊的引擎:")
    print("   - Engine 1: Trend Alpha")
    print("   - Engine 2: Mean Reversion Alpha")
    print("   - Engine 3: Factor Alpha")
    print("   - Engine 4: Carry Alpha")
    print("   - Engine 5: Convex / Tail Alpha")
    print("\n請在 strategies/ 目錄下實現新策略，然後更新此文件。")
    return


if __name__ == "__main__":
    run_meta_backtest()