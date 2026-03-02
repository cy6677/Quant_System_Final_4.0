# backtest/universal_backtester.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Data Classes
# ============================================================


@dataclass
class Order:
    ticker: str
    order_type: str  # "MARKET", "TARGETWEIGHT", "CLOSEALL"
    quantity: float = 0.0
    target_weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    qty: float
    avg_cost: float


# ============================================================
# Strategy Base Class
# ============================================================


class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        pass

    @abstractmethod
    def on_bar(
        self,
        date: pd.Timestamp,
        universe_prices: Dict[str, pd.DataFrame],
        current_portfolio_value: float,
        positions: Dict[str, Position],
        cash: float,
    ) -> List[Order]:
        pass


# ============================================================
# Transaction Cost Model
# ============================================================


class TransactionCostModel:
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        min_commission: float = 1.0,
    ):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.slippage_pct = slippage_bps / 10_000.0
        self.min_commission = min_commission

    def calculate_cost_and_slippage(
        self, order_type: str, price: float, shares: float
    ):
        notional_value = price * abs(shares)
        commission = max(
            notional_value * self.commission_rate, self.min_commission
        )
        if shares > 0:
            executed_price = price * (1 + self.slippage_pct)
        elif shares < 0:
            executed_price = price * (1 - self.slippage_pct)
        else:
            executed_price = price
        return commission, executed_price


# ============================================================
# Performance Analyzer
# ============================================================


class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate

    def analyze(
        self,
        df: pd.DataFrame,
        trade_log: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if df.empty or len(df) < 2:
            return {}

        returns = df["equity"].pct_change().dropna()
        if returns.empty:
            return {}

        ann_factor = 252
        total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        ann_return = (
            (1 + total_return) ** (1 / years) - 1.0 if years > 0 else 0.0
        )
        ann_vol = returns.std() * math.sqrt(ann_factor)

        daily_rf = self.risk_free_rate / ann_factor
        excess_returns = returns - daily_rf
        sharpe = (
            excess_returns.mean() / returns.std() * math.sqrt(ann_factor)
            if returns.std() > 0
            else 0.0
        )

        cum_max = df["equity"].cummax()
        drawdown = (df["equity"] - cum_max) / cum_max
        max_dd = drawdown.min()

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * math.sqrt(ann_factor)
        sortino = (
            excess_returns.mean() * ann_factor / downside_std
            if downside_std > 0
            else 0.0
        )

        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

        win_rate, profit_factor = self._trade_stats(trade_log)

        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    @staticmethod
    def _trade_stats(trade_log: Optional[pd.DataFrame]) -> tuple:
        if trade_log is None or trade_log.empty:
            return 0.0, 0.0
        if "action" not in trade_log.columns:
            return 0.0, 0.0
        sells = trade_log[trade_log["action"] == "SELL"]
        if sells.empty or "pnl" not in sells.columns:
            return 0.0, 0.0
        wins = sells[sells["pnl"] > 0]
        losses = sells[sells["pnl"] <= 0]
        win_rate = len(wins) / len(sells) if len(sells) > 0 else 0.0
        gross_profit = wins["pnl"].sum() if not wins.empty else 0.0
        gross_loss = (
            abs(losses["pnl"].sum()) if not losses.empty else 1e-9
        )
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else 0.0
        )
        return win_rate, profit_factor


# ============================================================
# Universal Backtester
# ============================================================


class UniversalBacktester:
    """
    機構級回測引擎 v3 (hardened)

    修正項目
    --------
    1. Pending orders 執行成功後不再留在 queue（避免重覆執行）
    2. 賣出數量不可超過持倉（防止負持倉）
    3. 僅支援 MARKET（其他 order_type 直接忽略）
    4. trade_log 加入 executed_price/source_price 方便檢查滑點
    """

    ORDER_EXPIRY_DAYS = 5

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        cost_model: Optional[TransactionCostModel] = None,
        execution_delay: int = 0,
        allow_fractional: bool = True,
        calendar_ticker: str = "SPY",
    ):
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.execution_delay = execution_delay
        self.allow_fractional = allow_fractional
        self.calendar_ticker = calendar_ticker

        self._trade_log_records: List[Dict] = []
        self.trade_log: pd.DataFrame = pd.DataFrame()

    def _execute_order(
        self,
        order: Order,
        exec_price: float,
        current_date: pd.Timestamp,
        signal_date: pd.Timestamp,
        cash: float,
        positions: Dict[str, Position],
    ) -> Tuple[float, bool]:
        if order.order_type != "MARKET":
            return cash, False

        qty = float(order.quantity)
        if qty == 0:
            return cash, False

        current_pos = positions.get(order.ticker, Position(0.0, 0.0))

        # 防止賣出超過持倉（no short）
        if qty < 0:
            max_sellable = float(current_pos.qty)
            if max_sellable <= 0:
                return cash, False
            qty = -min(abs(qty), max_sellable)
            if qty == 0:
                return cash, False

        if not self.allow_fractional:
            qty = int(qty)
            if qty == 0:
                return cash, False

        commission, executed_price = (
            self.cost_model.calculate_cost_and_slippage(
                order.order_type, exec_price, qty
            )
        )

        if qty > 0:
            gross_cost = qty * executed_price
            total_cost = gross_cost + commission
            if cash < total_cost:
                return cash, False
            cash -= total_cost
        else:
            gross_proceeds = abs(qty) * executed_price
            cash += gross_proceeds - commission

        new_qty = current_pos.qty + qty

        if new_qty > 1e-9:
            if qty > 0:
                new_avg = (
                    current_pos.qty * current_pos.avg_cost
                    + qty * executed_price
                ) / new_qty
            else:
                new_avg = current_pos.avg_cost
            positions[order.ticker] = Position(new_qty, new_avg)
        else:
            positions.pop(order.ticker, None)

        pnl = 0.0
        if qty < 0 and current_pos.qty > 0:
            pnl = (
                (executed_price - current_pos.avg_cost) * abs(qty)
                - commission
            )

        self._trade_log_records.append(
            {
                "date": current_date,
                "signal_date": signal_date,
                "ticker": order.ticker,
                "action": "BUY" if qty > 0 else "SELL",
                "qty": qty,
                "price": executed_price,
                "source_price": exec_price,
                "commission": commission,
                "pnl": pnl,
                "reason": order.metadata.get("reason", ""),
                "order_type": order.order_type,
            }
        )
        return cash, True

    def _calc_portfolio_value(
        self,
        cash: float,
        positions: Dict[str, Position],
        data_dict: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
    ) -> float:
        value = cash
        for ticker, pos in positions.items():
            df = data_dict.get(ticker)
            if df is not None and current_date in df.index:
                value += pos.qty * df.loc[current_date, "Close"]
            else:
                value += pos.qty * pos.avg_cost
        return value

    def run(
        self,
        strategy: BaseStrategy,
        data_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        self._trade_log_records = []
        strategy.reset()

        cash = self.initial_capital
        positions: Dict[str, Position] = {}
        equity_curve: List[Dict] = []
        pending_orders: List[tuple] = []  # (Order, signal_date)

        all_dates = pd.DatetimeIndex([])
        for df in data_dict.values():
            all_dates = all_dates.union(df.index)
        all_dates = all_dates[
            (all_dates >= start_date) & (all_dates <= end_date)
        ].sort_values()

        print(f"\n🔍 測試: start_date={start_date}, end_date={end_date}")
        print(f"🔍 測試: 準備回測，總共有 {len(all_dates)} 個交易日！")

        ticker_date_idx: Dict[str, Dict[pd.Timestamp, int]] = {}
        for t, df in data_dict.items():
            ticker_date_idx[t] = {d: i for i, d in enumerate(df.index)}

        for current_date in tqdm(
            all_dates, desc=f"Backtesting {strategy.name}"
        ):
            # 0) 執行 Pending Orders
            if self.execution_delay >= 1 and pending_orders:
                new_pending = []
                for order, signal_date in pending_orders:
                    days_since = (current_date - signal_date).days
                    if days_since > self.ORDER_EXPIRY_DAYS:
                        continue

                    ticker = order.ticker
                    df = data_dict.get(ticker)
                    if df is None or current_date not in df.index:
                        new_pending.append((order, signal_date))
                        continue

                    exec_price = (
                        df.loc[current_date, "Open"]
                        if "Open" in df.columns
                        else df.loc[current_date, "Close"]
                    )

                    cash, executed = self._execute_order(
                        order=order,
                        exec_price=exec_price,
                        current_date=current_date,
                        signal_date=signal_date,
                        cash=cash,
                        positions=positions,
                    )

                    # 只保留「未執行成功」訂單，避免重覆執行
                    if not executed:
                        new_pending.append((order, signal_date))

                pending_orders = new_pending

            # 1) 準備歷史資料
            sliced_data = {}
            for t, df in data_dict.items():
                idx = ticker_date_idx[t].get(current_date)
                if idx is not None:
                    sliced_data[t] = df.iloc[: idx + 1]

            portfolio_value = self._calc_portfolio_value(
                cash, positions, data_dict, current_date
            )

            orders = strategy.on_bar(
                current_date,
                sliced_data,
                portfolio_value,
                positions,
                cash,
            )
            if orders is None:
                orders = []

            # 2) 執行當日訂單
            for order in orders:
                if not isinstance(order, Order):
                    continue
                if order.order_type != "MARKET":
                    continue

                if self.execution_delay >= 1:
                    pending_orders.append((order, current_date))
                    continue

                ticker = order.ticker
                df = data_dict.get(ticker)
                if df is None or current_date not in df.index:
                    continue

                price = df.loc[current_date, "Close"]
                cash, _ = self._execute_order(
                    order=order,
                    exec_price=price,
                    current_date=current_date,
                    signal_date=current_date,
                    cash=cash,
                    positions=positions,
                )

            # 3) 記錄 Equity
            portfolio_value = self._calc_portfolio_value(
                cash, positions, data_dict, current_date
            )
            equity_curve.append(
                {"date": current_date, "equity": portfolio_value}
            )

        if self._trade_log_records:
            self.trade_log = pd.DataFrame(self._trade_log_records)
        else:
            self.trade_log = pd.DataFrame(
                columns=[
                    "date",
                    "signal_date",
                    "ticker",
                    "action",
                    "qty",
                    "price",
                    "source_price",
                    "commission",
                    "pnl",
                    "reason",
                    "order_type",
                ]
            )

        if not equity_curve:
            return pd.DataFrame(columns=["equity"])

        df_res = pd.DataFrame(equity_curve).set_index("date")
        return df_res