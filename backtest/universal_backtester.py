"""
backtest/universal_backtester.py
================================
核心回測引擎 v2。

特點：
- 支持任意 strategy (繼承 BaseStrategy)
- T+1 執行 (execution_delay=1)
- 交易成本模型 (佣金 + 滑價)
- 完整 trade log
- 無 look-ahead bias

修訂：
- execution_delay 默認 = 1 (T+1 Open 執行)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ═══════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════

@dataclass
class Position:
    """持倉"""
    qty: float = 0.0
    avg_cost: float = 0.0

    @property
    def market_value(self) -> float:
        return 0.0  # backtester 會用最新價計算

    def update(self, fill_qty: float, fill_price: float):
        """更新持倉 (買入或賣出)"""
        if fill_qty > 0:
            # 買入: 更新均價
            total_cost = self.avg_cost * self.qty + fill_price * fill_qty
            self.qty += fill_qty
            self.avg_cost = total_cost / self.qty if self.qty > 0 else 0.0
        elif fill_qty < 0:
            # 賣出
            self.qty += fill_qty  # fill_qty is negative
            if self.qty <= 1e-9:
                self.qty = 0.0
                self.avg_cost = 0.0


@dataclass
class Order:
    """訂單"""
    ticker: str
    order_type: str = "MARKET"         # MARKET / LIMIT
    quantity: float = 0.0              # 正=買, 負=賣
    limit_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransactionCostModel:
    """交易成本模型"""
    commission_rate: float = 0.001     # 0.1%
    slippage_bps: float = 10.0         # 10 bps
    min_commission: float = 1.0        # 最低佣金

    def calc_cost(self, price: float, quantity: float) -> float:
        """計算總成本 (佣金 + 滑價)"""
        notional = abs(price * quantity)

        # 佣金
        commission = max(notional * self.commission_rate, self.min_commission)

        # 滑價
        slippage = notional * (self.slippage_bps / 10000.0)

        return commission + slippage

    def calc_fill_price(self, price: float, quantity: float) -> float:
        """計算考慮滑價後嘅成交價"""
        slip = price * (self.slippage_bps / 10000.0)
        if quantity > 0:
            return price + slip   # 買入加滑價
        else:
            return price - slip   # 賣出減滑價


# ═══════════════════════════════════════
# Base Strategy (ABC)
# ═══════════════════════════════════════

class BaseStrategy(ABC):
    """
    所有策略嘅基類。

    子類必須實現：
    - on_bar(): 每日回調，返回 Order 列表
    - reset(): 重設內部狀態
    """

    history_window: int = 200  # 需要嘅最少歷史天數

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.DataFrame],
        current_portfolio_value: float,
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Order]:
        """
        每日回調。

        Parameters
        ----------
        date : pd.Timestamp
            當前日期
        universe_prices : Dict[str, pd.DataFrame]
            已 slice 到當前日期嘅價格數據
            每隻股票只包含到 date 為止嘅歷史數據（無 look-ahead）
        current_portfolio_value : float
            當前組合總值 (cash + positions)
        positions : Dict[str, Position]
            當前持倉
        cash : float
            當前現金

        Returns
        -------
        List[Order]
            今日要執行嘅訂單
        """
        ...

    @abstractmethod
    def reset(self):
        """重設所有內部狀態（新回測前調用）"""
        ...


# ═══════════════════════════════════════
# Universal Backtester
# ═══════════════════════════════════════

class UniversalBacktester:
    """
    事件驅動回測引擎。

    Parameters
    ----------
    initial_capital : float
        初始資金
    cost_model : TransactionCostModel
        交易成本模型
    execution_delay : int
        訂單延遲天數
        0 = 當日 Close 執行（有 bias 風險）
        1 = 次日 Open 執行（推薦）
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cost_model: Optional[TransactionCostModel] = None,
        execution_delay: int = 1,
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.execution_delay = execution_delay

        # State
        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_log: pd.DataFrame = pd.DataFrame()
        self._trade_records: List[Dict] = []
        self._pending_orders: List[Dict] = []  # delayed orders

    def run(
        self,
        strategy: BaseStrategy,
        data_dict: Dict[str, pd.DataFrame],
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        執行回測。

        Parameters
        ----------
        strategy : BaseStrategy
        data_dict : Dict[str, pd.DataFrame]
            全量價格數據（Date, Open, High, Low, Close, Volume）
        start_date : str
        end_date : str

        Returns
        -------
        pd.DataFrame
            columns: Date, equity, cash, positions_value, drawdown
        """
        # Reset
        self.cash = self.initial_capital
        self.positions = {}
        self._trade_records = []
        self._pending_orders = []
        strategy.reset()

        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # 建立統一日期軸 (用 SPY 或第一隻股票)
        all_dates = set()
        for ticker, df in data_dict.items():
            if "Date" in df.columns:
                dates = df["Date"]
                mask = (dates >= start_dt) & (dates <= end_dt)
                all_dates.update(dates[mask].tolist())

        if not all_dates:
            print("❌ 無可用交易日")
            return pd.DataFrame()

        trading_dates = sorted(all_dates)
        print(
            f"🏃 回測: {strategy.name} | "
            f"{trading_dates[0].strftime('%Y-%m-%d')} → "
            f"{trading_dates[-1].strftime('%Y-%m-%d')} | "
            f"{len(trading_dates)} 交易日"
        )

        # 預建日期索引 (加速 slice)
        date_indices = {}
        for ticker, df in data_dict.items():
            if "Date" in df.columns:
                date_indices[ticker] = df.set_index("Date").sort_index()

        equity_records = []

        for i, date in enumerate(trading_dates):
            # ── Step 1: 執行延遲訂單 ──
            if self._pending_orders:
                self._execute_pending(date, date_indices)

            # ── Step 2: Slice 數據到當前日期 ──
            universe_prices = {}
            for ticker, df_indexed in date_indices.items():
                sliced = df_indexed.loc[:date]
                if len(sliced) > 0:
                    universe_prices[ticker] = sliced.reset_index()

            # ── Step 3: 計算當前組合價值 ──
            positions_value = self._calc_positions_value(date, date_indices)
            portfolio_value = self.cash + positions_value

            # ── Step 4: 調用策略 ──
            orders = strategy.on_bar(
                date=date,
                universe_prices=universe_prices,
                current_portfolio_value=portfolio_value,
                positions=dict(self.positions),  # copy
                cash=self.cash,
            )

            # ── Step 5: 處理訂單 ──
            if orders:
                if self.execution_delay == 0:
                    # 即時執行 (用當日 Close)
                    for order in orders:
                        self._execute_order(order, date, date_indices, use_open=False)
                else:
                    # 延遲執行
                    for order in orders:
                        self._pending_orders.append({
                            "order": order,
                            "signal_date": date,
                            "execute_after": date,
                        })

            # ── Step 6: 記錄 equity ──
            positions_value = self._calc_positions_value(date, date_indices)
            equity = self.cash + positions_value

            equity_records.append({
                "Date": date,
                "equity": equity,
                "cash": self.cash,
                "positions_value": positions_value,
                "n_positions": sum(1 for p in self.positions.values() if p.qty > 0),
            })

            # 進度顯示
            if (i + 1) % 252 == 0 or i == len(trading_dates) - 1:
                pnl_pct = (equity / self.initial_capital - 1) * 100
                print(
                    f"  📅 {date.strftime('%Y-%m-%d')} | "
                    f"Equity: ${equity:,.0f} ({pnl_pct:+.1f}%) | "
                    f"Positions: {equity_records[-1]['n_positions']}"
                )

        # 組建結果
        equity_df = pd.DataFrame(equity_records)
        if not equity_df.empty:
            peak = equity_df["equity"].cummax()
            equity_df["drawdown"] = (equity_df["equity"] - peak) / peak

        self.trade_log = pd.DataFrame(self._trade_records)

        n_trades = len(self._trade_records)
        final_equity = equity_df["equity"].iloc[-1] if not equity_df.empty else self.initial_capital
        total_return = (final_equity / self.initial_capital - 1) * 100
        print(
            f"\n📊 完成 | 最終: ${final_equity:,.0f} ({total_return:+.1f}%) | "
            f"交易次數: {n_trades}"
        )

        return equity_df

    # ─────────────────────────────────────
    # Private: Order Execution
    # ─────────────────────────────────────

    def _execute_pending(self, current_date: pd.Timestamp, date_indices: Dict):
        """執行所有 pending orders"""
        remaining = []
        for pending in self._pending_orders:
            order = pending["order"]
            # 用當日 Open 執行
            self._execute_order(order, current_date, date_indices, use_open=True)
        self._pending_orders = remaining  # should be empty

    def _execute_order(
        self,
        order: Order,
        date: pd.Timestamp,
        date_indices: Dict,
        use_open: bool = True,
    ):
        """執行單個訂單"""
        ticker = order.ticker
        if ticker not in date_indices:
            return

        df = date_indices[ticker]

        # 搵當日或之前最近嘅數據
        available = df.loc[:date]
        if available.empty:
            return

        row = available.iloc[-1]
        price_col = "Open" if use_open and "Open" in df.columns else "Close"

        try:
            price = float(row[price_col])
        except (ValueError, KeyError):
            price = float(row["Close"])

        if price <= 0 or not np.isfinite(price):
            return

        qty = order.quantity

        # 計算成交價 (含滑價)
        fill_price = self.cost_model.calc_fill_price(price, qty)

        # 計算交易成本
        cost = self.cost_model.calc_cost(price, qty)

        # 計算所需資金
        trade_value = fill_price * qty  # 正=買入(扣錢), 負=賣出(加錢)

        # 買入時檢查資金
        if qty > 0:
            total_cost = trade_value + cost
            if total_cost > self.cash:
                # 資金不足 → 調整數量
                max_affordable = (self.cash - cost) / fill_price
                if max_affordable < 0.5:
                    return  # 完全買唔到
                qty = max_affordable

        # 更新持倉
        if ticker not in self.positions:
            self.positions[ticker] = Position()

        pos = self.positions[ticker]

        # 賣出時檢查持倉
        if qty < 0 and pos.qty < abs(qty):
            qty = -pos.qty  # 最多賣曬

        if abs(qty) < 1e-9:
            return

        pos.update(qty, fill_price)

        # 更新現金
        self.cash -= (fill_price * qty + cost)

        # 清理零持倉
        if pos.qty <= 1e-9:
            del self.positions[ticker]

        # 記錄交易
        self._trade_records.append({
            "date": date,
            "ticker": ticker,
            "side": "BUY" if qty > 0 else "SELL",
            "quantity": abs(qty),
            "price": fill_price,
            "cost": cost,
            "notional": abs(fill_price * qty),
            "reason": order.metadata.get("reason", ""),
            "strategy": order.metadata.get("strategy", ""),
        })

    def _calc_positions_value(
        self, date: pd.Timestamp, date_indices: Dict
    ) -> float:
        """計算所有持倉嘅市值"""
        total = 0.0
        for ticker, pos in self.positions.items():
            if pos.qty <= 0:
                continue
            if ticker not in date_indices:
                continue

            df = date_indices[ticker]
            available = df.loc[:date]
            if available.empty:
                continue

            price = float(available["Close"].iloc[-1])
            if np.isfinite(price) and price > 0:
                total += pos.qty * price

        return total


# ═══════════════════════════════════════
# Performance Analyzer
# ═══════════════════════════════════════

class PerformanceAnalyzer:
    """回測績效分析"""

    def analyze(
        self,
        equity_df: pd.DataFrame,
        trade_log: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, Any]:
        """
        分析回測結果。

        Returns
        -------
        Dict with keys:
            total_return, cagr, sharpe, sortino, calmar,
            max_drawdown, win_rate, profit_factor, n_trades, etc.
        """
        if equity_df.empty:
            return {}

        equity = equity_df["equity"].values
        dates = pd.to_datetime(equity_df["Date"])

        # ── Basic Returns ──
        total_return = float(equity[-1] / equity[0] - 1)
        n_days = (dates.iloc[-1] - dates.iloc[0]).days
        n_years = n_days / 365.25 if n_days > 0 else 1.0
        cagr = float((equity[-1] / equity[0]) ** (1.0 / n_years) - 1) if n_years > 0 else 0.0

        # ── Daily Returns ──
        daily_returns = pd.Series(equity).pct_change().dropna()

        # ── Sharpe ──
        if len(daily_returns) > 10 and daily_returns.std() > 1e-9:
            excess = daily_returns.mean() - risk_free_rate / 252
            sharpe = float(excess / daily_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # ── Sortino ──
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 5 and downside.std() > 1e-9:
            excess = daily_returns.mean() - risk_free_rate / 252
            sortino = float(excess / downside.std() * np.sqrt(252))
        else:
            sortino = 0.0

        # ── Max Drawdown ──
        peak = pd.Series(equity).cummax()
        dd = (pd.Series(equity) - peak) / peak
        max_dd = float(dd.min())

        # ── Calmar ──
        calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0

        # ── Trade Stats ──
        n_trades = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0

        if trade_log is not None and not trade_log.empty:
            n_trades = len(trade_log)

            # 簡單計算 win rate (用 round-trip)
            sells = trade_log[trade_log["side"] == "SELL"]
            if not sells.empty and "price" in sells.columns:
                # 粗略估算
                buys = trade_log[trade_log["side"] == "BUY"]
                if not buys.empty:
                    avg_buy = buys["price"].mean()
                    avg_sell = sells["price"].mean()
                    win_rate = float((sells["price"] > 0).sum() / len(sells))

        # ── Volatility ──
        annual_vol = float(daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 10 else 0.0

        results = {
            "total_return": round(total_return, 4),
            "cagr": round(cagr, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_dd, 4),
            "annual_vol": round(annual_vol, 4),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "n_days": n_days,
            "n_years": round(n_years, 2),
            "start_equity": round(float(equity[0]), 2),
            "end_equity": round(float(equity[-1]), 2),
        }

        return results