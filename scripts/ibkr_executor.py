#!/usr/bin/env python3
"""
Daily Screener v3 (hardened)

新增
----
1. 多策略訂單 netting（同 ticker 買賣淨額化）
2. 總體買入資金上限檢查（避免超配）
3. 保留 IBKR 持倉讀取
"""
import os
import sys
import datetime
from pathlib import Path
from collections import defaultdict
import pandas as pd
import requests

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config
from layers.data_hub import DataHub
from backtest.universal_backtester import Position


def get_ib_connection(port: int = 7496):
    try:
        from ib_insync import IB
    except ImportError:
        print("⚠️ ib_insync 未安裝。")
        return None

    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=2)
        print("✅ 成功連接 IBKR!")
        return ib
    except Exception as e:
        print(f"⚠️ 無法連接 IBKR (TWS 可能未開): {e}")
        return None


def get_ib_capital(ib, default_capital: float = 100000.0) -> float:
    if ib is None:
        return default_capital

    try:
        account_values = ib.accountValues()
        for val in account_values:
            if val.tag == "NetLiquidation" and val.currency == "USD":
                capital = float(val.value)
                print(f"💰 IBKR 總資金: ${capital:,.2f}")
                return capital
    except Exception as e:
        print(f"⚠️ 讀取 IBKR 餘額失敗: {e}")

    return default_capital


def get_ib_positions(ib) -> dict:
    positions = {}
    if ib is None:
        return positions

    try:
        for pos in ib.positions():
            if pos.contract.secType == "STK":
                ticker = pos.contract.symbol
                qty = float(pos.position)
                avg_cost = float(pos.avgCost)
                if qty != 0:
                    positions[ticker] = Position(
                        qty=qty, avg_cost=avg_cost
                    )
        if positions:
            print(f"📋 已讀取 {len(positions)} 個現有持倉")
        else:
            print("📋 目前無持倉")
    except Exception as e:
        print(f"⚠️ 讀取持倉失敗: {e}")

    return positions


def send_telegram_msg(msg_text: str) -> None:
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        print("⚠️ 未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，跳過發送。")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg_text,
        "parse_mode": "HTML",
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("📲 成功將 Action List 發送至 Telegram！")
        else:
            print(f"❌ Telegram 發送失敗: {response.text}")
    except Exception as e:
        print(f"❌ Telegram 網絡錯誤: {e}")


def _net_orders(all_orders):
    """
    同 ticker 買賣淨額化:
    +qty BUY, -qty SELL
    """
    net_qty = defaultdict(float)
    reasons = defaultdict(list)

    for o in all_orders:
        q = float(o.quantity)
        if q == 0:
            continue
        net_qty[o.ticker] += q
        reasons[o.ticker].append(
            f"{o.metadata.get('strategy', 'unknown')}:{o.metadata.get('reason', '')}"
        )

    merged = []
    for ticker, qty in net_qty.items():
        if abs(qty) < 1e-9:
            continue
        action = "BUY" if qty > 0 else "SELL"
        merged.append(
            {
                "ticker": ticker,
                "action": action,
                "qty": abs(float(qty)),
                "reason": ";".join(reasons[ticker])[:300],
            }
        )
    return merged


def _risk_cap_orders(orders_for_executor, prices_dict, total_capital, cap_ratio=0.98):
    """
    用 Close 粗估 notional，總買入不得超過 total_capital * cap_ratio
    """
    buy_budget = total_capital * cap_ratio
    used = 0.0
    capped = []

    for od in orders_for_executor:
        if od["action"] != "BUY":
            capped.append(od)
            continue

        ticker = od["ticker"]
        qty = float(od["qty"])
        df = prices_dict.get(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            continue

        px = float(df["Close"].iloc[-1])
        est_cost = qty * px * 1.002  # buffer for slippage+fee

        if used + est_cost <= buy_budget:
            capped.append(od)
            used += est_cost
        else:
            remain = buy_budget - used
            if remain > 0:
                affordable_qty = int(remain / (px * 1.002))
                if affordable_qty > 0:
                    capped.append(
                        {
                            **od,
                            "qty": float(affordable_qty),
                            "reason": od.get("reason", "") + "|risk_capped_partial",
                        }
                    )
                    used += affordable_qty * px * 1.002

    return capped


def run_screener():
    print("🚀 啟動 【實戰指揮部】 (Daily Screener v3)...")
    cfg = load_config("config.yaml") if os.path.exists("config.yaml") else {}

    print("📦 正在獲取最新市場數據...")
    hub = DataHub(cfg)
    prices_dict = hub.load_price_dict()

    if not prices_dict:
        print("❌ 載入數據失敗")
        return

    calendar_df = prices_dict.get("SPY", list(prices_dict.values())[0])
    today_date = calendar_df.index[-1]
    date_str = today_date.strftime("%Y-%m-%d")
    print(f"📅 篩選基準日 (As of Date): {date_str}")

    ib = get_ib_connection(port=7496)
    total_capital = get_ib_capital(ib, default_capital=100000.0)
    existing_positions = get_ib_positions(ib)

    if ib is not None:
        try:
            ib.disconnect()
        except Exception:
            pass

    strategies_to_run = ["longterm", "a", "stat_arb"]
    all_orders = []

    capital_longterm = total_capital * 0.5
    capital_swing_total = total_capital * 0.5
    capital_swing_each = capital_swing_total / 2

    print(f"💰 總資金: ${total_capital:,.2f}")
    print(f"   📌 長線分配: ${capital_longterm:,.2f}")
    print(f"   📌 Swing 總分配: ${capital_swing_total:,.2f} (��策略 ${capital_swing_each:,.2f})")

    for strat_name in strategies_to_run:
        print(f"\n🔍 執行策略: {strat_name.upper()}...")
        try:
            from scripts.run_experiment import get_strategy_class
            StratClass = get_strategy_class(strat_name)
            strategy = StratClass()

            sliced_prices = {
                ticker: df.loc[:today_date]
                for ticker, df in prices_dict.items()
            }

            allocated_cash = capital_longterm if strat_name == "longterm" else capital_swing_each

            orders = strategy.on_bar(
                date=today_date,
                universe_prices=sliced_prices,
                current_portfolio_value=allocated_cash,
                positions=existing_positions,
                cash=allocated_cash,
            )

            if orders:
                for o in orders:
                    o.metadata["strategy"] = strat_name
                all_orders.extend(orders)
                print(f"   🎯 產生 {len(orders)} 個交易指令")
            else:
                print("   💤 無交易訊號")

        except Exception as e:
            print(f"   ❌ 策略 {strat_name} 執行失敗: {e}")

    # ========= 新增：跨策略淨額化 + 風控封頂 =========
    netted_orders = _net_orders(all_orders)
    capped_orders = _risk_cap_orders(
        netted_orders, prices_dict, total_capital, cap_ratio=0.98
    )

    tg_msg = f"<b>📊 Quant 系統每日報告 ({date_str})</b>\n\n"
    tg_msg += f"💰 最新總資金: ${total_capital:,.2f}\n"
    if existing_positions:
        tg_msg += f"📋 現有持倉: {len(existing_positions)} 隻\n"

    print("\n" + "=" * 50)
    print("📋 明日 (T+1) 開市行動清單 (Action List)")
    print("=" * 50)

    if not capped_orders:
        print("🎉 今日無事，安心訓覺！")
        tg_msg += "\n☕ 今日無任何交易訊號，安心訓覺！\n維持現有倉位。"
        send_telegram_msg(tg_msg)
        return

    tg_msg += "\n🎯 <b>明日開市行動清單 (Action List):</b>\n"
    orders_for_executor = []

    for od in capped_orders:
        action_emoji = "BUY 🟢" if od["action"] == "BUY" else "SELL 🔴"
        qty_display = round(float(od["qty"]), 2)
        qty_str = f"{qty_display} 股"
        ticker = od["ticker"]
        reason = od.get("reason", "N/A")

        print(f"[NET] {action_emoji} {ticker} - {qty_str} (Reason: {reason})")
        tg_msg += f"• [NET] {action_emoji} <b>{ticker}</b> - {qty_str} (<code>{reason}</code>)\n"

        orders_for_executor.append(
            {
                "ticker": ticker,
                "action": od["action"],
                "qty": float(od["qty"]),
                "reason": reason,
            }
        )

    send_telegram_msg(tg_msg)

    print("\n" + "=" * 50)
    print("🤖 準備將訂單發送至 IBKR TWS...")
    try:
        from scripts.ibkr_executor import execute_orders
        execute_orders(orders_for_executor)
    except Exception as e:
        print(f"❌ 自動落單失敗: {e}")


if __name__ == "__main__":
    run_screener()