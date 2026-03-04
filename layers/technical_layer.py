"""
layers/technical_layer.py
=========================
技術指標計算層。

輸入 DataFrame (OHLCV)，計算常用指標並 merge 回同一個 DataFrame。
所有函數都係 vectorized，唔會 look-ahead。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def add_all_indicators(
    df: pd.DataFrame,
    sma_periods: tuple = (20, 50, 100, 200),
    ema_periods: tuple = (12, 26, 50),
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.DataFrame:
    """
    一次過加入所有常用指標。

    Parameters
    ----------
    df : pd.DataFrame
        需要有 Open, High, Low, Close, Volume columns

    Returns
    -------
    pd.DataFrame
        原始 df + 新增指標 columns
    """
    if df.empty or "Close" not in df.columns:
        return df

    df = df.copy()
    close = df["Close"].astype(float)

    # ── SMA ──
    for p in sma_periods:
        df[f"SMA_{p}"] = close.rolling(window=p, min_periods=p).mean()

    # ── EMA ──
    for p in ema_periods:
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    # ── RSI ──
    df["RSI"] = _calc_rsi(close, rsi_period)

    # ── ATR ──
    if "High" in df.columns and "Low" in df.columns:
        df["ATR"] = _calc_atr(
            df["High"].astype(float),
            df["Low"].astype(float),
            close,
            atr_period,
        )
        df["ATR_pct"] = df["ATR"] / close

    # ── Bollinger Bands ──
    mid = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    df["BB_upper"] = mid + bb_std * std
    df["BB_mid"] = mid
    df["BB_lower"] = mid - bb_std * std
    df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # ── Returns ──
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_21d"] = close.pct_change(21)
    df["return_63d"] = close.pct_change(63)

    # ── Volatility ──
    df["vol_21d"] = df["return_1d"].rolling(21).std() * np.sqrt(252)
    df["vol_63d"] = df["return_1d"].rolling(63).std() * np.sqrt(252)

    # ── Volume features ──
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        df["vol_sma_20"] = vol.rolling(20).mean()
        df["vol_ratio"] = vol / df["vol_sma_20"].replace(0, np.nan)

    # ── Momentum Score (vol-adjusted) ──
    for lb in [21, 63, 126, 252]:
        ret = close.pct_change(lb)
        vol = df["return_1d"].rolling(lb).std() * np.sqrt(252)
        df[f"mom_score_{lb}"] = ret / vol.replace(0, np.nan)

    # ── Drawdown ──
    cummax = close.cummax()
    df["drawdown"] = (close - cummax) / cummax

    return df


# ── Internal helpers ──

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    alpha = 1.0 / period
    avg_up = up.ewm(alpha=alpha, adjust=False).mean()
    avg_down = down.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()