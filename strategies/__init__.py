"""
strategies/__init__.py
======================
策略註冊中心 — 所有策略嘅 import 同 param space 定義喺呢度。
"""
from typing import Dict, Any, Union, List, Tuple


def get_strategy_class(name: str):
    """
    根據名稱返回策略 class。

    Usage
    -----
    >>> StratClass = get_strategy_class("trend_alpha")
    >>> strategy = StratClass(lookback_fast=63, top_n=15)
    """
    name = name.lower().strip()

    if name == "trend_alpha":
        from strategies.trend_alpha import TrendAlphaStrategy
        return TrendAlphaStrategy

    if name in ("mean_reversion", "mean_rev", "meanrev"):
        from strategies.mean_reversion import MeanReversionStrategy
        return MeanReversionStrategy

    # ─── 未來引擎在此加入 ───
    # if name == "factor_alpha":
    #     from strategies.factor_alpha import FactorAlphaStrategy
    #     return FactorAlphaStrategy
    # if name == "carry_alpha":
    #     from strategies.carry_alpha import CarryAlphaStrategy
    #     return CarryAlphaStrategy
    # if name == "convex_tail":
    #     from strategies.convex_tail import ConvexTailStrategy
    #     return ConvexTailStrategy

    raise ValueError(
        f"找唔到策略: '{name}' | "
        f"可用策略: trend_alpha, mean_reversion"
    )


def get_strategy_param_space(
    name: str,
) -> Dict[str, Union[List[Any], Tuple[float, float], Any]]:
    """
    返回策略嘅參數搜尋空間（俾 Optimizer 用）。

    規則
    ----
    - tuple(a, b)  → continuous range [a, b]（int or float）
    - list[...]    → categorical choices
    - scalar       → fixed value（唔做搜尋）
    """
    name = name.lower().strip()

    if name == "trend_alpha":
        return {
            "lookback_fast": (21, 63),
            "lookback_slow": (126, 252),
            "rebalance_days": [10, 15, 21],
            "top_n": (8, 25),
            "sma_filter": [100, 150, 200],
            "atr_period": [10, 14, 20],
            "vol_target": (0.10, 0.25),
            "max_position_pct": (0.04, 0.10),
            "breakout_lookback": (30, 80),
            "use_adaptive_lookback": [True, False],
            "use_breakout_confirm": [True, False],
            "corr_filter_threshold": (0.55, 0.80),
        }

    if name in ("mean_reversion", "mean_rev", "meanrev"):
        return {
            "rsi_period": [2, 3, 4],
            "rsi_oversold": (5, 20),
            "rsi_overbought": (80, 95),
            "zscore_lookback": (10, 30),
            "zscore_entry": (-3.0, -1.5),
            "bb_lookback": (15, 30),
            "bb_std": (1.5, 3.0),
            "max_holding_days": (3, 10),
            "max_positions": (5, 15),
            "position_size_pct": (0.02, 0.05),
            "exit_rsi": (55, 75),
            "use_gap_signal": [True, False],
        }

    return {}


# 可用策略清單
AVAILABLE_STRATEGIES = ["trend_alpha", "mean_reversion"]