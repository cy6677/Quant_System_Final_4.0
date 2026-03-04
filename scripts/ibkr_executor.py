#!/usr/bin/env python3
"""
ibkr_executor.py
================
IBKR 自動下單（上線交易用）。

⚠️ 需要：
  1. pip install ib_insync
  2. TWS / IB Gateway 運行中
  3. API 已開啟 (Edit → Global Config → API → Settings)

⚠️ 首次請用 Paper Trading (port 7497)
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import load_config, get_nested

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    HAS_IBKR = True
except ImportError:
    HAS_IBKR = False


class IBKRExecutor:
    """
    IBKR 訂單執行器。

    用法:
        executor = IBKRExecutor(cfg)
        executor.connect()
        executor.execute_orders(orders)
        executor.disconnect()
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.host = get_nested(cfg, "ibkr", "host", default="127.0.0.1")
        self.port = get_nested(cfg, "ibkr", "port", default=7497)
        self.client_id = get_nested(cfg, "ibkr", "client_id", default=1)
        self.timeout = get_nested(cfg, "ibkr", "timeout", default=30)
        self.ib: Optional[IB] = None

    def connect(self) -> bool:
        """連接 TWS / IB Gateway"""
        if not HAS_IBKR:
            print("❌ 未安裝 ib_insync: pip install ib_insync")
            return False

        try:
            self.ib = IB()
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
            )
            print(f"✅ 已連接 IBKR ({self.host}:{self.port})")
            print(f"   帳戶: {self.ib.managedAccounts()}")
            return True
        except Exception as e:
            print(f"❌ 連接失敗: {e}")
            return False

    def disconnect(self):
        """斷開連接"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            print("✅ 已斷開 IBKR")

    def get_positions(self) -> Dict[str, float]:
        """取得當前持倉"""
        if not self.ib or not self.ib.isConnected():
            return {}

        positions = {}
        for pos in self.ib.positions():
            symbol = pos.contract.symbol
            qty = float(pos.position)
            if abs(qty) > 0:
                positions[symbol] = qty

        return positions

    def get_account_value(self) -> float:
        """取得帳戶淨值"""
        if not self.ib or not self.ib.isConnected():
            return 0.0

        for item in self.ib.accountSummary():
            if item.tag == "NetLiquidation":
                return float(item.value)
        return 0.0

    def execute_orders(
        self,
        orders: List[Dict],
        dry_run: bool = True,
    ) -> List[Dict]:
        """
        執行訂單。

        Parameters
        ----------
        orders : List[Dict]
            每個 dict: {"ticker": str, "side": "BUY"/"SELL", "quantity": float}
        dry_run : bool
            True = 只打印唔執行 (預設安全模式)

        Returns
        -------
        List[Dict]
            執行結果
        """
        if not self.ib or not self.ib.isConnected():
            print("❌ 未連接 IBKR")
            return []

        results = []
        print(f"\n{'🔵 DRY RUN' if dry_run else '🔴 LIVE'} — {len(orders)} 個訂單")
        print("=" * 60)

        for order_info in orders:
            ticker = order_info["ticker"]
            side = order_info["side"].upper()
            qty = int(abs(order_info["quantity"]))

            if qty <= 0:
                continue

            contract = Stock(ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            action = side  # "BUY" or "SELL"
            order = MarketOrder(action, qty)

            print(
                f"  {'📈' if side == 'BUY' else '📉'} "
                f"{side} {qty} x {ticker}"
            )

            if dry_run:
                results.append({
                    "ticker": ticker,
                    "side": side,
                    "quantity": qty,
                    "status": "DRY_RUN",
                    "time": datetime.now().isoformat(),
                })
            else:
                try:
                    trade = self.ib.placeOrder(contract, order)
                    # 等待填單
                    timeout_count = 0
                    while not trade.isDone() and timeout_count < 30:
                        self.ib.sleep(1)
                        timeout_count += 1

                    status = trade.orderStatus.status
                    fill_price = trade.orderStatus.avgFillPrice

                    results.append({
                        "ticker": ticker,
                        "side": side,
                        "quantity": qty,
                        "status": status,
                        "fill_price": fill_price,
                        "time": datetime.now().isoformat(),
                    })

                    print(f"       → {status} @ ${fill_price:.2f}")

                except Exception as e:
                    results.append({
                        "ticker": ticker,
                        "side": side,
                        "quantity": qty,
                        "status": f"ERROR: {e}",
                        "time": datetime.now().isoformat(),
                    })
                    print(f"       → ❌ {e}")

        # 儲存執行記錄
        log_dir = Path(get_nested(self.cfg, "output", "logs_dir", default="logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"ibkr_orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with log_file.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 訂單記錄: {log_file}")

        return results


def main():
    """
    命令行入口。

    用法:
        python scripts/ibkr_executor.py --dry-run
        python scripts/ibkr_executor.py --live    ← 真金白銀！
    """
    import argparse

    parser = argparse.ArgumentParser(description="IBKR Order Executor")
    parser.add_argument("--live", action="store_true", help="真實下單（預設 dry-run）")
    parser.add_argument("--orders-file", type=str, help="訂單 JSON 檔案路徑")
    args = parser.parse_args()

    cfg = load_config()
    executor = IBKRExecutor(cfg)

    if not executor.connect():
        sys.exit(1)

    try:
        # 印帳戶資訊
        nav = executor.get_account_value()
        positions = executor.get_positions()
        print(f"\n📊 帳戶淨值: ${nav:,.0f}")
        print(f"📊 持倉: {len(positions)} 隻")
        for sym, qty in sorted(positions.items()):
            print(f"   {sym}: {qty:.0f}")

        # 載入訂單
        if args.orders_file:
            with open(args.orders_file) as f:
                orders = json.load(f)
        else:
            print("\n⚠️ 未指定訂單檔案 (--orders-file)")
            print("   範例: python scripts/ibkr_executor.py --orders-file orders.json --dry-run")
            print("\n   orders.json 格式:")
            print('   [{"ticker": "AAPL", "side": "BUY", "quantity": 10}]')
            orders = []

        if orders:
            executor.execute_orders(orders, dry_run=not args.live)

    finally:
        executor.disconnect()


if __name__ == "__main__":
    main()