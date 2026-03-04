"""
strategies/base.py
==================
策略通用工具函數。

所有函數設計為 vectorized，接收 pd.Series 輸出 pd.Series。
無 look-ahead bias — 只用到 t 時刻及之前嘅數據。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


# ─────────────────────────────────────
# Technical Indicators (single series)
# ─────────────────────────────────────

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI"""
    delta = close.diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    alpha = 1.0 / period
    avg_up = up.ewm(alpha=alpha, adjust=False).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range (Wilder smoothing)"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def calc_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period, min_periods=period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calc_zscore(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Rolling z-score"""
    mean = series.rolling(lookback, min_periods=lookback).mean()
    std = series.rolling(lookback, min_periods=lookback).std()
    return (series - mean) / std.replace(0, np.nan)


def calc_bb(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """Bollinger Bands → (lower, mid, upper)"""
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def calc_donchian(
    high: pd.Series, low: pd.Series, period: int = 50
):
    """Donchian Channel → (lower, upper)"""
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    return lower, upper


def calc_momentum_score(close: pd.Series, lookback: int) -> float:
    """
    Vol-adjusted momentum score (scalar)。
    回傳最新值嘅 momentum / vol ratio。
    """
    if len(close) < lookback + 10:
        return 0.0
    ret = float(close.iloc[-1] / close.iloc[-lookback] - 1)
    vol = float(close.pct_change().iloc[-lookback:].std()) * np.sqrt(252)
    if vol < 1e-9:
        return 0.0
    return ret / vol


# ─────────────────────────────────────
# Position sizing helpers
# ─────────────────────────────────────

def inverse_vol_weights(
    atr_dict: Dict[str, float],
    total_budget: float = 1.0,
    max_weight: float = 0.10,
) -> Dict[str, float]:
    """
    Inverse-volatility weighting.
    ATR 越高 → 權重越低 → 風險平等。
    """
    if not atr_dict:
        return {}

    inv_vol = {}
    for ticker, atr in atr_dict.items():
        if atr > 1e-9:
            inv_vol[ticker] = 1.0 / atr
        else:
            inv_vol[ticker] = 0.0

    total_inv = sum(inv_vol.values())
    if total_inv < 1e-9:
        return {}

    weights = {}
    for ticker, iv in inv_vol.items():
        w = (iv / total_inv) * total_budget
        weights[ticker] = min(w, max_weight)

    # 重新 normalize
    total_w = sum(weights.values())
    if total_w > total_budget and total_w > 0:
        scale = total_budget / total_w
        weights = {k: v * scale for k, v in weights.items()}

    return weights