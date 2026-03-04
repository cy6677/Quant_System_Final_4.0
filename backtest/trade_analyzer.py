"""
backtest/trade_analyzer.py
==========================
交易分析模組 — E-Ratio、持倉分析、交易質量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class TradeAnalyzer:
    """
    進階交易分析。

    分析 trade_log 計算：
    - E-Ratio (Edge Ratio)
    - 平均持倉天數
    - 最佳/最差交易
    - 每月/每季分佈
    """

    def analyze(
        self,
        trade_log: pd.DataFrame,
        equity_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        分析交易記錄。

        Parameters
        ----------
        trade_log : pd.DataFrame
            columns: date, ticker, side, quantity, price, cost, notional, reason, strategy
        equity_df : pd.DataFrame, optional
            equity 曲線

        Returns
        -------
        Dict
        """
        if trade_log.empty:
            return {"error": "no trades"}

        results = {}

        # ── 基本統計 ──
        results["total_trades"] = len(trade_log)

        buys = trade_log[trade_log["side"] == "BUY"]
        sells = trade_log[trade_log["side"] == "SELL"]
        results["n_buys"] = len(buys)
        results["n_sells"] = len(sells)

        # ── 每隻股票交易次數 ──
        if "ticker" in trade_log.columns:
            ticker_counts = trade_log["ticker"].value_counts()
            results["most_traded"] = ticker_counts.head(10).to_dict()
            results["unique_tickers"] = int(ticker_counts.nunique())

        # ── 平均交易大小 ──
        if "notional" in trade_log.columns:
            results["avg_trade_size"] = float(trade_log["notional"].mean())
            results["total_notional"] = float(trade_log["notional"].sum())

        # ── 交易成本 ──
        if "cost" in trade_log.columns:
            results["total_costs"] = float(trade_log["cost"].sum())
            results["avg_cost_per_trade"] = float(trade_log["cost"].mean())

        # ── E-Ratio (簡化版) ──
        e_ratio = self._calc_e_ratio(trade_log)
        results["e_ratio"] = e_ratio

        # ── 月度分佈 ──
        if "date" in trade_log.columns:
            trade_log_copy = trade_log.copy()
            trade_log_copy["date"] = pd.to_datetime(trade_log_copy["date"])
            monthly = trade_log_copy.groupby(
                trade_log_copy["date"].dt.to_period("M")
            ).size()
            results["avg_trades_per_month"] = float(monthly.mean())
            results["max_trades_in_month"] = int(monthly.max())

        # ── 策略分佈 ──
        if "strategy" in trade_log.columns:
            strat_counts = trade_log["strategy"].value_counts().to_dict()
            results["by_strategy"] = strat_counts

        # ── 進場原因分佈 ──
        if "reason" in trade_log.columns:
            # 提取主要原因 (取 | 之前嘅部分)
            reasons = trade_log["reason"].apply(
                lambda x: str(x).split("|")[0] if pd.notna(x) else "unknown"
            )
            results["by_reason"] = reasons.value_counts().head(10).to_dict()

        return results

    def _calc_e_ratio(self, trade_log: pd.DataFrame) -> Optional[float]:
        """
        簡化版 E-Ratio。

        E-Ratio = 平均有利移動 (MFE) / 平均不利移動 (MAE)
        > 1.0 = 有 edge
        """
        # 需要更詳細嘅逐筆 P&L 數據才能計算真正嘅 E-Ratio
        # 這裡用簡化版: 基於 buy/sell 配對
        if "ticker" not in trade_log.columns:
            return None

        buys = trade_log[trade_log["side"] == "BUY"][["date", "ticker", "price"]].copy()
        sells = trade_log[trade_log["side"] == "SELL"][["date", "ticker", "price"]].copy()

        if buys.empty or sells.empty:
            return None

        # 簡單配對: 每隻股票嘅第一個 buy 配第一個 sell
        profits = []
        losses = []

        for ticker in buys["ticker"].unique():
            b = buys[buys["ticker"] == ticker].sort_values("date")
            s = sells[sells["ticker"] == ticker].sort_values("date")

            n_pairs = min(len(b), len(s))
            for j in range(n_pairs):
                buy_price = float(b.iloc[j]["price"])
                sell_price = float(s.iloc[j]["price"])
                if buy_price > 0:
                    pnl_pct = (sell_price - buy_price) / buy_price
                    if pnl_pct >= 0:
                        profits.append(pnl_pct)
                    else:
                        losses.append(abs(pnl_pct))

        avg_profit = np.mean(profits) if profits else 0.0
        avg_loss = np.mean(losses) if losses else 1e-9

        return round(float(avg_profit / avg_loss), 4) if avg_loss > 1e-9 else None