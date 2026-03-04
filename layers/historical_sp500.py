"""
layers/historical_sp500.py
==========================
S&P 500 歷史成分管理。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd


FALLBACK_SP500 = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEP",
    "AFL", "AIG", "AMAT", "AMD", "AMGN", "AMP", "AMZN", "ANET", "AVGO", "AXP",
    "BA", "BAC", "BDX", "BK", "BKNG", "BLK", "BMY", "BSX", "BX",
    "C", "CAT", "CB", "CCI", "CCL", "CDW", "CEG", "CI", "CL", "CMCSA",
    "CME", "COP", "COST", "CRM", "CRWD", "CSCO", "CTAS", "CVS", "CVX",
    "D", "DASH", "DD", "DE", "DHI", "DHR", "DIS", "DUK",
    "EA", "EBAY", "EMR", "EOG", "EQT", "ETN", "EW",
    "F", "FAST", "FCX", "FDX", "FI", "FICO", "FTNT",
    "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GPN", "GS", "GWW",
    "HAL", "HCA", "HD", "HON", "HSY", "HUM",
    "IBM", "ICE", "INTC", "INTU", "ISRG", "ITW",
    "JCI", "JNJ", "JPM",
    "KDP", "KHC", "KLAC", "KMB", "KO", "KR",
    "LEN", "LHX", "LIN", "LLY", "LMT", "LOW", "LRCX", "LULU",
    "MA", "MAR", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META",
    "MMC", "MMM", "MNST", "MO", "MPC", "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI",
    "NEE", "NFLX", "NKE", "NOC", "NOW", "NSC", "NVDA", "NVO",
    "ON", "ORCL", "OXY",
    "PANW", "PAYX", "PEP", "PFE", "PG", "PGR", "PH", "PLTR", "PM", "PNC", "PSA", "PSX",
    "PYPL",
    "QCOM",
    "REGN", "ROP", "ROST", "RSG", "RTX",
    "SBUX", "SCHW", "SHW", "SLB", "SNPS", "SO", "SPG", "SPGI", "SRE", "SYK", "SYY",
    "T", "TDG", "TGT", "TJX", "TMO", "TMUS", "TRGP", "TRV", "TSLA", "TT", "TXN",
    "UBER", "UNH", "UNP", "UPS", "URI", "USB",
    "V", "VICI", "VLO", "VRSK", "VRTX", "VST", "VZ",
    "WBA", "WBD", "WELL", "WFC", "WM", "WMT",
    "XEL", "XOM",
    "ZTS",
]


class HistoricalSP500:
    """
    S&P 500 歷史成分管理。

    支持兩種初始化方式（兼容 pipeline.py）：
        HistoricalSP500(data_dir="data/sp500")
        HistoricalSP500(cachedir="data/raw")
    """

    def __init__(
        self,
        data_dir: str = "data/sp500",
        cachedir: Optional[str] = None,
    ):
        # 兼容 cachedir 參數
        if cachedir is not None:
            self.data_dir = Path(cachedir)
        else:
            self.data_dir = Path(data_dir)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._snapshots_file = self.data_dir / "sp500_snapshots.json"
        self._snapshots = self._load_snapshots()

    # ─── pipeline.py 需要嘅 download() 方法 ───

    def download(self) -> Optional[pd.DataFrame]:
        """
        下載 S&P 500 歷史成分。

        Returns
        -------
        pd.DataFrame or None
            columns: date, symbol
        """
        tickers = self.fetch_current()

        if not tickers:
            return None

        # 建立 DataFrame（當前成分 × 今日日期）
        today = datetime.now().strftime("%Y-%m-%d")
        rows = [{"date": today, "symbol": t} for t in tickers]
        df = pd.DataFrame(rows)

        # 合併歷史快照
        if self._snapshots:
            all_rows = []
            for date_str, syms in self._snapshots.items():
                for s in syms:
                    all_rows.append({"date": date_str, "symbol": s})
            if all_rows:
                hist_df = pd.DataFrame(all_rows)
                df = pd.concat([hist_df, df], ignore_index=True)
                df = df.drop_duplicates(
                    subset=["date", "symbol"], keep="last"
                )
                df = df.sort_values(["date", "symbol"]).reset_index(
                    drop=True
                )

        # 儲存
        csv_path = self.data_dir / "sp500_history.csv"
        df.to_csv(csv_path, index=False)

        return df

    # ─── 原有方法 ───

    def get_constituents(
        self, date: Optional[str] = None
    ) -> List[str]:
        if not self._snapshots:
            return FALLBACK_SP500

        if date is None:
            latest_key = max(self._snapshots.keys())
            return self._snapshots[latest_key]

        target = pd.Timestamp(date)
        valid_keys = [
            k for k in self._snapshots.keys()
            if pd.Timestamp(k) <= target
        ]

        if not valid_keys:
            return FALLBACK_SP500

        best_key = max(valid_keys)
        return self._snapshots[best_key]

    def save_snapshot(
        self, tickers: List[str], date: Optional[str] = None
    ):
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        self._snapshots[date] = sorted(set(tickers))
        self._save_snapshots()
        print(f"✅ SP500 快照已儲存: {date} ({len(tickers)} 隻)")

    def fetch_current(self) -> List[str]:
        """從 Wikipedia 抓取當前 S&P 500 成分"""
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            if tables and len(tables) > 0:
                df = tables[0]
                if "Symbol" in df.columns:
                    tickers = (
                        df["Symbol"]
                        .str.replace(".", "-", regex=False)
                        .tolist()
                    )
                    tickers = [
                        t.strip() for t in tickers if isinstance(t, str)
                    ]
                    self.save_snapshot(tickers)
                    print(
                        f"✅ 從 Wikipedia 抓取 {len(tickers)} 隻 SP500 成分"
                    )
                    return tickers
        except Exception as e:
            print(f"⚠️ Wikipedia 抓取失敗: {e}")

        return FALLBACK_SP500

    def _load_snapshots(self) -> dict:
        if self._snapshots_file.exists():
            try:
                with self._snapshots_file.open("r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_snapshots(self):
        try:
            with self._snapshots_file.open("w") as f:
                json.dump(self._snapshots, f, indent=2)
        except Exception as e:
            print(f"⚠️ 儲存失敗: {e}")