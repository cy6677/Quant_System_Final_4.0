"""
strategies/trend_alpha.py
=========================
Engine 1: Trend Alpha Strategy

設計理念 (參考 2026 Q2 研究報告)
───────────────────────────────
- Multi-horizon momentum composite (21/63/126/252 日，長期權重更大)
- Adaptive lookback (波動率比率調整)
- Donchian breakout 確認
- Cross-stock correlation spike filter (危機保護)
- ATR-based inverse-volatility position sizing
- SMA-200 趨勢過濾
- 月度 rebalance (避免過度交易)

目標指標
────────
Sharpe 0.8-1.1 | Max DD < 20% | 平均持倉：週至月

Look-ahead bias 防護
─────────────────────
所有計算只用 .iloc[-N:] (歷史數據)，不用 .iloc[x:] (未來數據)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from backtest.universal_backtester import BaseStrategy, Order, Position
from strategies.base import (
    calc_sma,
    calc_atr,
    calc_momentum_score,
    calc_donchian,
    inverse_vol_weights,
)
from engine.regime_detector import RegimeDetector


class TrendAlphaStrategy(BaseStrategy):
    """
    Trend Alpha — 多速度動量 + 自適應 + 突破確認

    Parameters (全部可由 Optimizer 調整)
    ----------
    lookback_fast : int      快速動量窗口 (default: 42)
    lookback_slow : int      慢速動量窗口 (default: 200)
    rebalance_days : int     rebalance 頻率 (default: 21)
    top_n : int              持倉數量 (default: 15)
    sma_filter : int         趨勢過濾均線 (default: 200)
    atr_period : int         ATR 計算期 (default: 14)
    vol_target : float       目標年化波動率 (default: 0.15)
    max_position_pct : float 每隻股票最大權重 (default: 0.08)
    breakout_lookback : int  Donchian Channel 窗口 (default: 50)
    use_adaptive_lookback : bool  是否啟用自適應 lookback (default: True)
    use_breakout_confirm : bool   是否需要突破確認 (default: True)
    corr_filter_threshold : float 相關性 filter 閾值 (default: 0.65)
    """

    # Optimizer 用: 策略需要嘅最少歷史日數
    history_window = 300

    # Momentum 權重: 慢速權重 > 快速
    _MOM_LOOKBACKS = [21, 63, 126, 252]
    _MOM_WEIGHTS = [0.10, 0.20, 0.30, 0.40]

    def __init__(
        self,
        lookback_fast: int = 42,
        lookback_slow: int = 200,
        rebalance_days: int = 21,
        top_n: int = 15,
        sma_filter: int = 200,
        atr_period: int = 14,
        vol_target: float = 0.15,
        max_position_pct: float = 0.08,
        breakout_lookback: int = 50,
        use_adaptive_lookback: bool = True,
        use_breakout_confirm: bool = True,
        corr_filter_threshold: float = 0.65,
    ):
        super().__init__(name="TrendAlpha")
        self.lookback_fast = int(lookback_fast)
        self.lookback_slow = int(lookback_slow)
        self.rebalance_days = int(rebalance_days)
        self.top_n = int(top_n)
        self.sma_filter = int(sma_filter)
        self.atr_period = int(atr_period)
        self.vol_target = float(vol_target)
        self.max_position_pct = float(max_position_pct)
        self.breakout_lookback = int(breakout_lookback)
        self.use_adaptive_lookback = bool(use_adaptive_lookback)
        self.use_breakout_confirm = bool(use_breakout_confirm)
        self.corr_filter_threshold = float(corr_filter_threshold)

        # Internal state
        self._last_rebalance: Optional[pd.Timestamp] = None
        self._bar_count: int = 0
        self._regime_detector = RegimeDetector()

    def reset(self):
        self._last_rebalance = None
        self._bar_count = 0
        self._regime_detector.reset()

    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.DataFrame],
        current_portfolio_value: float,
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Order]:
        self._bar_count += 1

        # ── Warmup: 唔夠歷史數據就唔做嘢 ──
        spy_df = universe_prices.get("SPY")
        if spy_df is None or len(spy_df) < self.history_window:
            return []

        # ── 判斷係咪 rebalance 日 ──
        should_rebalance = False
        if self._last_rebalance is None:
            should_rebalance = True
        else:
            days_since = (date - self._last_rebalance).days
            if days_since >= self.rebalance_days:
                should_rebalance = True

        if not should_rebalance:
            # 非 rebalance 日：只檢查 hard stop (回撤保護)
            return self._check_hard_stops(date, universe_prices, positions)

        self._last_rebalance = date

        # ── Regime Detection ──
        regime = self._regime_detector.detect(universe_prices, date)
        regime_label = regime.label

        # Regime-based exposure multiplier
        regime_exposure = {
            "CRISIS": 0.30,       # 大幅減倉
            "STRONG_TREND": 1.00, # 全力
            "RANGE": 0.50,        # 半倉 (趨勢策略喺震盪市場唔好)
            "RISK_ON": 0.90,      # 接近全倉
        }
        exposure_mult = regime_exposure.get(regime_label, 0.70)

        # ── Cross-stock Correlation Filter ──
        avg_corr = regime.signals.get("avg_cross_corr", 0.3)
        if avg_corr > self.corr_filter_threshold:
            corr_mult = 0.5
        elif avg_corr > self.corr_filter_threshold + 0.10:
            corr_mult = 0.25
        else:
            corr_mult = 1.0

        total_exposure = exposure_mult * corr_mult

        # ── Adaptive Lookback ──
        if self.use_adaptive_lookback:
            effective_lookbacks = self._get_adaptive_lookbacks(spy_df)
        else:
            effective_lookbacks = self._MOM_LOOKBACKS.copy()

        # ── 計算每隻股票嘅 Momentum Composite Score ──
        scores = {}
        atr_values = {}

        for ticker, df in universe_prices.items():
            if ticker == "SPY":
                continue  # benchmark 唔入倉
            if df is None or len(df) < max(effective_lookbacks) + 20:
                continue
            if "Close" not in df.columns:
                continue

            # 跳過未上市 / 停牌股票
            if "pre_ipo_flag" in df.columns and df["pre_ipo_flag"].iloc[-1]:
                continue
            if "is_trading" in df.columns and not df["is_trading"].iloc[-1]:
                continue

            close = df["Close"].astype(float)
            current_price = float(close.iloc[-1])
            if current_price <= 0.5:
                continue  # 低價股直接跳過

            # 1) SMA Filter: 只做趨勢向上嘅股票
            if len(close) >= self.sma_filter:
                sma_val = float(close.iloc[-self.sma_filter:].mean())
                if current_price < sma_val:
                    continue  # 低於 SMA-200，跳過

            # 2) Multi-horizon Momentum Composite
            mom_score = 0.0
            valid_moms = 0
            for lb, w in zip(effective_lookbacks, self._MOM_WEIGHTS):
                if len(close) >= lb + 10:
                    s = calc_momentum_score(close, lb)
                    if np.isfinite(s):
                        mom_score += s * w
                        valid_moms += 1

            if valid_moms < 2:  # 至少要 2 個 lookback 有效
                continue

            # 3) Breakout Confirmation (optional)
            if self.use_breakout_confirm and len(df) > self.breakout_lookback + 5:
                high = df["High"].astype(float) if "High" in df.columns else close
                low = df["Low"].astype(float) if "Low" in df.columns else close
                dc_low, dc_high = calc_donchian(high, low, self.breakout_lookback)

                if dc_high.iloc[-1] is not None and np.isfinite(dc_high.iloc[-1]):
                    # 喺 Donchian 上半部 = breakout 確認
                    dc_range = float(dc_high.iloc[-1] - dc_low.iloc[-1])
                    if dc_range > 0:
                        dc_position = (current_price - float(dc_low.iloc[-1])) / dc_range
                        if dc_position < 0.5:
                            mom_score *= 0.5  # 未突破，信號打折
                        elif dc_position > 0.9:
                            mom_score *= 1.2  # 強突破，信號加成

            # 4) ATR for position sizing
            if "High" in df.columns and "Low" in df.columns:
                atr = calc_atr(
                    df["High"].astype(float),
                    df["Low"].astype(float),
                    close,
                    self.atr_period,
                )
                atr_val = float(atr.iloc[-1])
                if atr_val > 0:
                    atr_pct = atr_val / current_price  # ATR as % of price
                    atr_values[ticker] = atr_pct
                else:
                    atr_values[ticker] = 0.02  # fallback

            if np.isfinite(mom_score):
                scores[ticker] = mom_score

        if not scores:
            # 所有股票都無信號 → 清倉
            return self._close_all_positions(positions, "no_signal")

        # ── 排名取 Top N ──
        sorted_tickers = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        selected = sorted_tickers[: self.top_n]

        # ── Inverse-Volatility Weighting ──
        selected_atr = {t: atr_values.get(t, 0.02) for t in selected}
        target_weights = inverse_vol_weights(
            selected_atr,
            total_budget=total_exposure,
            max_weight=self.max_position_pct,
        )

        # ── 生成 Orders ──
        orders: List[Order] = []

        # Step 1: SELL 唔喺目標組合嘅持倉
        for ticker, pos in positions.items():
            if ticker not in target_weights and pos.qty > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="MARKET",
                        quantity=-pos.qty,
                        metadata={
                            "reason": f"trend_exit|regime={regime_label}",
                            "strategy": "trend_alpha",
                        },
                    )
                )

        # Step 2: BUY / ADJUST 目標組合
        for ticker, target_w in target_weights.items():
            if target_w <= 0:
                continue

            df = universe_prices.get(ticker)
            if df is None or "Close" not in df.columns:
                continue

            price = float(df["Close"].iloc[-1])
            if price <= 0:
                continue

            target_value = current_portfolio_value * target_w
            target_qty = target_value / price

            current_pos = positions.get(ticker, Position(0.0, 0.0))
            diff = target_qty - current_pos.qty

            # 只做顯著調整 (避免微調產生不必要交易成本)
            min_trade_value = current_portfolio_value * 0.005
            if abs(diff * price) < min_trade_value:
                continue

            if diff > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="MARKET",
                        quantity=diff,
                        metadata={
                            "reason": f"trend_entry|score={scores.get(ticker, 0):.2f}|regime={regime_label}",
                            "strategy": "trend_alpha",
                        },
                    )
                )
            elif diff < 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="MARKET",
                        quantity=diff,
                        metadata={
                            "reason": f"trend_reduce|regime={regime_label}",
                            "strategy": "trend_alpha",
                        },
                    )
                )

        return orders

    def _get_adaptive_lookbacks(self, spy_df: pd.DataFrame) -> List[int]:
        """根據當前波動率環境調整 momentum lookback 窗口"""
        close = spy_df["Close"].astype(float)
        if len(close) < 260:
            return self._MOM_LOOKBACKS.copy()

        ret = close.pct_change()
        vol_21 = float(ret.iloc[-21:].std()) * np.sqrt(252)
        vol_252 = float(ret.iloc[-252:].std()) * np.sqrt(252)

        ratio = vol_21 / vol_252 if vol_252 > 1e-9 else 1.0

        if ratio > 1.3:
            # 高波動 → 縮短 lookback (更快反應)
            scale = 0.6
        elif ratio > 1.1:
            scale = 0.8
        elif ratio < 0.8:
            # 低波動 → 延長 lookback (更穩定信號)
            scale = 1.3
        else:
            scale = 1.0

        adapted = [max(10, int(lb * scale)) for lb in self._MOM_LOOKBACKS]
        return adapted

    def _check_hard_stops(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.DataFrame],
        positions: Dict[str, Position],
    ) -> List[Order]:
        """非 rebalance 日：只檢查是否觸發 hard stop (回撤 > 15%)"""
        orders = []
        for ticker, pos in positions.items():
            if pos.qty <= 0:
                continue

            df = universe_prices.get(ticker)
            if df is None or "Close" not in df.columns:
                continue

            current_price = float(df["Close"].iloc[-1])
            if current_price <= 0:
                continue

            # 持倉回撤
            pnl_pct = (current_price - pos.avg_cost) / pos.avg_cost

            if pnl_pct < -0.15:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="MARKET",
                        quantity=-pos.qty,
                        metadata={
                            "reason": f"hard_stop|loss={pnl_pct:.2%}",
                            "strategy": "trend_alpha",
                        },
                    )
                )
        return orders

    def _close_all_positions(
        self, positions: Dict[str, Position], reason: str
    ) -> List[Order]:
        orders = []
        for ticker, pos in positions.items():
            if pos.qty > 0:
                orders.append(
                    Order(
                        ticker=ticker,
                        order_type="MARKET",
                        quantity=-pos.qty,
                        metadata={"reason": reason, "strategy": "trend_alpha"},
                    )
                )
        return orders