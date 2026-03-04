"""
engine/regime_detector.py
=========================
Regime Detection Engine v1 — Rule-based Ensemble

7-Signal 集成 Regime 偵測器。
用 SPY + Universe breadth 做判斷，無需外部宏觀數據源。

Regime 輸出：
  CRISIS       — 市場恐慌 / 系統性下跌
  STRONG_TREND — 強趨勢（上升或下降皆可）
  RANGE        — 區間震盪
  RISK_ON      — 風險偏好上升、穩步上行

設計原則：寧可遲 1-2 日偵測到 regime 變化，也不要因為噪音頻繁切換。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RegimeState:
    label: str            # CRISIS / STRONG_TREND / RANGE / RISK_ON
    confidence: float     # 0.0 ~ 1.0
    signals: Dict[str, float]  # 各信號嘅原始值
    trend_breadth: float  # % of universe above SMA-200


class RegimeDetector:
    """
    Rule-based Ensemble Regime Detector

    使用 SPY (或指定 benchmark) 嘅價格數據 + Universe breadth
    來判斷當前 regime。

    Parameters
    ----------
    benchmark : str
        基準股票代碼 (default: "SPY")
    vol_short : int
        短期波動率窗口 (default: 21)
    vol_long : int
        長期波動率窗口 (default: 252)
    sma_fast : int
        快均線 (default: 50)
    sma_slow : int
        慢均線 (default: 200)
    breadth_sma : int
        Breadth 計算用嘅均線 (default: 200)
    corr_window : int
        跨資產相關性窗口 (default: 21)
    smoothing : int
        Regime 信號平滑窗口 避免 whipsaw (default: 5)
    """

    def __init__(
        self,
        benchmark: str = "SPY",
        vol_short: int = 21,
        vol_long: int = 252,
        sma_fast: int = 50,
        sma_slow: int = 200,
        breadth_sma: int = 200,
        corr_window: int = 21,
        smoothing: int = 5,
    ):
        self.benchmark = benchmark
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.breadth_sma = breadth_sma
        self.corr_window = corr_window
        self.smoothing = smoothing

        self._prev_regime: Optional[str] = None
        self._regime_counter: int = 0
        self._min_regime_hold: int = smoothing  # 最少持續 N 日

    def reset(self):
        self._prev_regime = None
        self._regime_counter = 0

    def detect(
        self,
        universe_prices: Dict[str, pd.DataFrame],
        date: Optional[pd.Timestamp] = None,
    ) -> RegimeState:
        """
        主偵測方法。

        Parameters
        ----------
        universe_prices : Dict[str, pd.DataFrame]
            已 slice 到當前日期嘅價格字典
        date : pd.Timestamp, optional
            當前日期 (用作 debug)

        Returns
        -------
        RegimeState
        """
        signals = {}

        # ---- Signal 1: Benchmark 波動率 Regime ----
        bench_df = universe_prices.get(self.benchmark)
        if bench_df is not None and len(bench_df) > self.vol_long + 10:
            close = bench_df["Close"].astype(float)
            ret = close.pct_change()

            vol_s = ret.iloc[-self.vol_short:].std() * np.sqrt(252)
            vol_l = ret.iloc[-self.vol_long:].std() * np.sqrt(252)
            vol_ratio = vol_s / vol_l if vol_l > 1e-9 else 1.0
            signals["vol_ratio"] = float(vol_ratio)

            # Annualized vol level
            signals["vol_level"] = float(vol_s)
        else:
            signals["vol_ratio"] = 1.0
            signals["vol_level"] = 0.15

        # ---- Signal 2: Benchmark Trend (Price vs SMA) ----
        if bench_df is not None and len(bench_df) > self.sma_slow + 10:
            close = bench_df["Close"].astype(float)
            sma_f = close.iloc[-self.sma_fast:].mean()
            sma_s = close.iloc[-self.sma_slow:].mean()
            current = float(close.iloc[-1])

            above_fast = 1.0 if current > sma_f else 0.0
            above_slow = 1.0 if current > sma_s else 0.0
            fast_above_slow = 1.0 if sma_f > sma_s else 0.0

            signals["bench_above_sma50"] = above_fast
            signals["bench_above_sma200"] = above_slow
            signals["sma50_above_sma200"] = fast_above_slow
        else:
            signals["bench_above_sma50"] = 1.0
            signals["bench_above_sma200"] = 1.0
            signals["sma50_above_sma200"] = 1.0

        # ---- Signal 3: Benchmark Drawdown ----
        if bench_df is not None and len(bench_df) > 50:
            close = bench_df["Close"].astype(float)
            peak = close.iloc[-252:].max() if len(close) >= 252 else close.max()
            dd = (float(close.iloc[-1]) - peak) / peak
            signals["bench_drawdown"] = float(dd)
        else:
            signals["bench_drawdown"] = 0.0

        # ---- Signal 4: Trend Breadth ----
        breadth = self._calc_breadth(universe_prices)
        signals["trend_breadth"] = breadth

        # ---- Signal 5: 21d Return Momentum ----
        if bench_df is not None and len(bench_df) > 25:
            close = bench_df["Close"].astype(float)
            ret_21 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
            signals["bench_ret_21d"] = ret_21
        else:
            signals["bench_ret_21d"] = 0.0

        # ---- Signal 6: Cross-stock Correlation ----
        avg_corr = self._calc_cross_correlation(universe_prices)
        signals["avg_cross_corr"] = avg_corr

        # ---- Signal 7: Recent Drawdown Speed (急跌偵測) ----
        if bench_df is not None and len(bench_df) > 10:
            close = bench_df["Close"].astype(float)
            ret_5d = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0.0
            signals["bench_ret_5d"] = ret_5d
        else:
            signals["bench_ret_5d"] = 0.0

        # ---- Regime Classification ----
        raw_regime = self._classify(signals)

        # 抗 whipsaw: 至少持續 N 日先允許切換
        if self._prev_regime is not None and raw_regime != self._prev_regime:
            self._regime_counter += 1
            if self._regime_counter < self._min_regime_hold:
                # 未夠日數，維持舊 regime（但 CRISIS 可以即時切換）
                if raw_regime == "CRISIS" and signals.get("bench_ret_5d", 0) < -0.05:
                    pass  # 急跌允許即時切換到 CRISIS
                else:
                    raw_regime = self._prev_regime
            else:
                self._regime_counter = 0
        else:
            self._regime_counter = 0

        self._prev_regime = raw_regime

        confidence = self._calc_confidence(signals, raw_regime)

        return RegimeState(
            label=raw_regime,
            confidence=confidence,
            signals=signals,
            trend_breadth=breadth,
        )

    def _calc_breadth(self, universe_prices: Dict[str, pd.DataFrame]) -> float:
        """計算 universe 中有幾多 % 嘅股票高於 SMA-200"""
        above = 0
        total = 0

        for ticker, df in universe_prices.items():
            if ticker == self.benchmark:
                continue
            if df is None or len(df) < self.breadth_sma + 10:
                continue
            if "Close" not in df.columns:
                continue

            close = df["Close"].astype(float)
            sma = close.iloc[-self.breadth_sma:].mean()
            current = float(close.iloc[-1])

            total += 1
            if current > sma:
                above += 1

        return above / total if total > 0 else 0.5

    def _calc_cross_correlation(
        self, universe_prices: Dict[str, pd.DataFrame]
    ) -> float:
        """計算 universe 中隨機抽樣股票嘅平均相關性"""
        # 為避免計算量太大，只取最多 30 隻
        sample_tickers = []
        for t, df in universe_prices.items():
            if df is not None and len(df) > self.corr_window + 5 and "Close" in df.columns:
                sample_tickers.append(t)
            if len(sample_tickers) >= 30:
                break

        if len(sample_tickers) < 5:
            return 0.3  # 唔夠數據，回傳中性值

        # 建構 returns matrix
        returns_dict = {}
        for t in sample_tickers:
            close = universe_prices[t]["Close"].astype(float)
            ret = close.pct_change().iloc[-self.corr_window:]
            if len(ret.dropna()) >= self.corr_window - 5:
                returns_dict[t] = ret

        if len(returns_dict) < 5:
            return 0.3

        ret_df = pd.DataFrame(returns_dict).dropna()
        if len(ret_df) < 10:
            return 0.3

        corr_matrix = ret_df.corr()
        n = len(corr_matrix)
        if n < 2:
            return 0.3

        # 平均 off-diagonal correlation
        total_corr = corr_matrix.values.sum() - n  # 減去對角線
        avg_corr = total_corr / (n * n - n)

        return float(np.clip(avg_corr, -1.0, 1.0))

    def _classify(self, signals: Dict[str, float]) -> str:
        """基於 7 個信號做 regime 分類"""
        vol_ratio = signals.get("vol_ratio", 1.0)
        vol_level = signals.get("vol_level", 0.15)
        above_sma200 = signals.get("bench_above_sma200", 1.0)
        above_sma50 = signals.get("bench_above_sma50", 1.0)
        sma50_above_200 = signals.get("sma50_above_sma200", 1.0)
        dd = signals.get("bench_drawdown", 0.0)
        breadth = signals.get("trend_breadth", 0.5)
        ret_5d = signals.get("bench_ret_5d", 0.0)
        ret_21d = signals.get("bench_ret_21d", 0.0)
        avg_corr = signals.get("avg_cross_corr", 0.3)

        crisis_score = 0.0
        trend_score = 0.0
        range_score = 0.0
        riskon_score = 0.0

        # ---- CRISIS signals ----
        if vol_level > 0.30:
            crisis_score += 2.0
        elif vol_level > 0.25:
            crisis_score += 1.0

        if vol_ratio > 1.5:
            crisis_score += 1.5
        elif vol_ratio > 1.2:
            crisis_score += 0.5

        if dd < -0.15:
            crisis_score += 2.0
        elif dd < -0.10:
            crisis_score += 1.0

        if ret_5d < -0.05:
            crisis_score += 1.5

        if avg_corr > 0.65:
            crisis_score += 1.5
        elif avg_corr > 0.50:
            crisis_score += 0.5

        if breadth < 0.30:
            crisis_score += 1.0

        # ---- STRONG_TREND signals ----
        if above_sma200 > 0.5 and sma50_above_200 > 0.5:
            trend_score += 1.5
        elif above_sma200 < 0.5 and sma50_above_200 < 0.5:
            trend_score += 1.0  # 下跌趨勢也是趨勢

        if breadth > 0.65:
            trend_score += 1.5
        elif breadth < 0.35:
            trend_score += 1.0  # 廣泛下跌 = 下跌趨勢

        if abs(ret_21d) > 0.05:
            trend_score += 1.0

        if vol_ratio < 1.0:
            trend_score += 0.5

        # ---- RANGE signals ----
        if 0.40 < breadth < 0.60:
            range_score += 1.5

        if abs(ret_21d) < 0.02:
            range_score += 1.0

        if 0.8 < vol_ratio < 1.2:
            range_score += 1.0

        if abs(dd) < 0.05:
            range_score += 0.5

        # ---- RISK_ON signals ----
        if above_sma200 > 0.5 and above_sma50 > 0.5:
            riskon_score += 1.0

        if vol_level < 0.15:
            riskon_score += 1.5
        elif vol_level < 0.18:
            riskon_score += 0.5

        if breadth > 0.55:
            riskon_score += 1.0

        if ret_21d > 0.02:
            riskon_score += 0.5

        if avg_corr < 0.30:
            riskon_score += 0.5

        # ---- Pick highest score ----
        scores = {
            "CRISIS": crisis_score,
            "STRONG_TREND": trend_score,
            "RANGE": range_score,
            "RISK_ON": riskon_score,
        }

        # CRISIS 有 priority — 即使分數咁上下，寧可誤判 crisis
        if crisis_score >= 4.0:
            return "CRISIS"

        return max(scores, key=scores.get)

    def _calc_confidence(self, signals: Dict[str, float], regime: str) -> float:
        """簡單 confidence：看信號一致性"""
        # 粗略計：有幾多信號支持呢個 regime
        if regime == "CRISIS":
            checks = [
                signals.get("vol_level", 0) > 0.25,
                signals.get("bench_drawdown", 0) < -0.10,
                signals.get("trend_breadth", 0.5) < 0.40,
                signals.get("avg_cross_corr", 0.3) > 0.50,
            ]
        elif regime == "STRONG_TREND":
            checks = [
                signals.get("bench_above_sma200", 0) > 0.5,
                signals.get("trend_breadth", 0.5) > 0.60 or signals.get("trend_breadth", 0.5) < 0.35,
                abs(signals.get("bench_ret_21d", 0)) > 0.03,
                signals.get("vol_ratio", 1) < 1.2,
            ]
        elif regime == "RISK_ON":
            checks = [
                signals.get("vol_level", 0.2) < 0.18,
                signals.get("bench_above_sma200", 0) > 0.5,
                signals.get("trend_breadth", 0.5) > 0.55,
                signals.get("bench_ret_21d", 0) > 0,
            ]
        else:  # RANGE
            checks = [
                0.40 < signals.get("trend_breadth", 0.5) < 0.60,
                abs(signals.get("bench_ret_21d", 0)) < 0.03,
                0.8 < signals.get("vol_ratio", 1) < 1.2,
                signals.get("vol_level", 0.2) < 0.22,
            ]

        return sum(checks) / len(checks) if checks else 0.5