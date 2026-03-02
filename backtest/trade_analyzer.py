# backtest/trade_analyzer.py
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional


class TradeAnalyzer:
    """
    機構級交易分析器（MFE / MAE / E-Ratio）v2 hardened

    升級
    ----
    1. 支援 DataFrame / List[Dict]
    2. 防呆欄位檢查
    3. 優先使用 round-trip SELL（有 pnl）做分析；否則回退 BUY entry 模式
    """

    def __init__(
        self,
        trade_log: Union[pd.DataFrame, List[Dict]],
        prices_dict: Dict[str, pd.DataFrame],
    ):
        if isinstance(trade_log, pd.DataFrame):
            self.trades = trade_log.copy()
        elif isinstance(trade_log, list):
            self.trades = pd.DataFrame(trade_log)
        else:
            self.trades = pd.DataFrame()

        self.prices = prices_dict

    def _calc_from_buy_entries(self, t_days: List[int]) -> Dict[str, float]:
        required_cols = {"action", "ticker", "date", "price"}
        if not required_cols.issubset(self.trades.columns):
            return {}

        buy_trades = self.trades[self.trades["action"] == "BUY"].copy()
        if buy_trades.empty:
            return {}

        results = {}
        for t in t_days:
            mfes, maes = [], []

            for _, trade in buy_trades.iterrows():
                ticker = trade["ticker"]
                entry_date = pd.to_datetime(trade["date"])
                entry_price = float(trade["price"])

                if ticker not in self.prices or entry_price <= 0:
                    continue

                df = self.prices[ticker]
                future_df = df.loc[entry_date:].head(t + 1)
                if len(future_df) <= 1:
                    continue

                high_col = "High" if "High" in future_df.columns else "high"
                low_col = "Low" if "Low" in future_df.columns else "low"
                if high_col not in future_df.columns or low_col not in future_df.columns:
                    continue

                max_price = future_df[high_col].max()
                min_price = future_df[low_col].min()

                mfe = (max_price - entry_price) / entry_price
                mae = (entry_price - min_price) / entry_price
                mae = max(mae, 0.0001)

                mfes.append(mfe)
                maes.append(mae)

            if mfes and maes:
                avg_mfe = np.mean(mfes)
                avg_mae = np.mean(maes)
                e_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0.0
                results[f"E{t}"] = round(e_ratio, 2)

        return results

    def _calc_from_round_trip(self, t_days: List[int]) -> Dict[str, float]:
        required_cols = {"action", "ticker", "date", "price"}
        if not required_cols.issubset(self.trades.columns):
            return {}

        df_tr = self.trades.copy()
        df_tr["date"] = pd.to_datetime(df_tr["date"], errors="coerce")
        df_tr = df_tr.dropna(subset=["date"]).sort_values("date")

        # lot-level FIFO queue
        buy_queues: Dict[str, List[Dict]] = {}
        round_trips = []  # each: ticker, entry_date, entry_price, exit_date

        for _, row in df_tr.iterrows():
            action = str(row["action"]).upper()
            ticker = row["ticker"]
            price = float(row["price"]) if pd.notna(row["price"]) else np.nan
            qty = float(row["qty"]) if "qty" in row and pd.notna(row["qty"]) else 0.0
            dt = row["date"]

            if not np.isfinite(price) or price <= 0:
                continue

            if ticker not in buy_queues:
                buy_queues[ticker] = []

            if action == "BUY" and qty > 0:
                buy_queues[ticker].append(
                    {"qty": qty, "entry_date": dt, "entry_price": price}
                )

            elif action == "SELL" and qty < 0:
                sell_qty = abs(qty)
                q = buy_queues[ticker]
                while sell_qty > 1e-9 and q:
                    lot = q[0]
                    matched = min(sell_qty, lot["qty"])
                    round_trips.append(
                        {
                            "ticker": ticker,
                            "entry_date": lot["entry_date"],
                            "entry_price": lot["entry_price"],
                            "exit_date": dt,
                            "qty": matched,
                        }
                    )
                    lot["qty"] -= matched
                    sell_qty -= matched
                    if lot["qty"] <= 1e-9:
                        q.pop(0)

        if not round_trips:
            return {}

        rt_df = pd.DataFrame(round_trips)
        results = {}

        for t in t_days:
            mfes, maes = [], []
            for _, tr in rt_df.iterrows():
                ticker = tr["ticker"]
                entry_date = pd.to_datetime(tr["entry_date"])
                entry_price = float(tr["entry_price"])
                if ticker not in self.prices or entry_price <= 0:
                    continue

                px = self.prices[ticker]
                future_df = px.loc[entry_date:].head(t + 1)
                if len(future_df) <= 1:
                    continue

                high_col = "High" if "High" in future_df.columns else "high"
                low_col = "Low" if "Low" in future_df.columns else "low"
                if high_col not in future_df.columns or low_col not in future_df.columns:
                    continue

                max_price = future_df[high_col].max()
                min_price = future_df[low_col].min()

                mfe = (max_price - entry_price) / entry_price
                mae = (entry_price - min_price) / entry_price
                mae = max(mae, 0.0001)

                mfes.append(mfe)
                maes.append(mae)

            if mfes and maes:
                avg_mfe = np.mean(mfes)
                avg_mae = np.mean(maes)
                results[f"E{t}"] = round(avg_mfe / avg_mae if avg_mae > 0 else 0.0, 2)

        return results

    def calculate_e_ratio(
        self, t_days: Optional[List[int]] = None
    ) -> Dict[str, float]:
        if t_days is None:
            t_days = [5, 10, 20]

        if self.trades.empty:
            print("⚠️ 無交易紀錄，無法計算 E-Ratio。")
            return {}

        # 優先 round-trip
        results = self._calc_from_round_trip(t_days)
        if not results:
            results = self._calc_from_buy_entries(t_days)

        if results:
            print("\n📊 交易進場優勢分析 (Edge Ratio):")
            for k, v in results.items():
                if v > 1.2:
                    evaluation = "🔥 具備強大優勢"
                elif v > 1.0:
                    evaluation = "✅ 有正向優勢"
                else:
                    evaluation = "⚠️ 無明顯優勢"
                print(f" - {k} 日 E-Ratio: {v:.2f}  ({evaluation})")
        else:
            print("⚠️ 無法計算任何 E-Ratio（可能數據不足）。")

        return results