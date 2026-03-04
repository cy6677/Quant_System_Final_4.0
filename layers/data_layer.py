"""
layers/data_layer.py
====================
底層數據下載 + 快取管理。

包含：
- DataLayer: 通用數據載入（CSV 快取）
- UniverseProvider: 股票池管理
- PriceDownloader: 批量下載 + parquet 儲存
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# ═══════════════════════════════════════
# 預設股票池
# ═══════════════════════════════════════

DEFAULT_UNIVERSE = [
    "SPY",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",
    "AMD", "AVGO", "QCOM", "ON", "MCHP", "SWKS", "INTC",
    "PANW", "CRWD", "DDOG", "ZS", "FTNT", "NET",
    "JPM", "BAC", "GS", "MS", "SCHW", "C", "V", "MA",
    "UNH", "LLY", "ABBV", "MRK", "TMO", "ISRG", "JNJ",
    "WMT", "PG", "COST", "HD", "LOW", "TGT",
    "DECK", "LULU", "BURL", "NKE",
    "DHI", "LEN", "PHM", "TOL",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "CAT", "DE", "GE", "HON", "RTX", "LMT",
    "DIS", "NFLX", "CRM", "ADBE", "NOW", "UBER",
]


# ═══════════════════════════════════════
# UniverseProvider (pipeline.py 需要)
# ═══════════════════════════════════════

class UniverseProvider:
    """
    股票池提供者。

    從 config 讀取 universe 設定，回傳股票清單。
    """

    def __init__(self, config: dict):
        self.config = config

    def get_universe(self) -> List[str]:
        """回傳當前股票池"""
        universe_cfg = self.config.get("universe", {})
        mode = universe_cfg.get("mode", "custom")

        if mode == "custom":
            custom = universe_cfg.get("custom_tickers", None)
            if custom and isinstance(custom, list):
                return sorted(set(custom))

        if mode == "sp500":
            try:
                from layers.historical_sp500 import HistoricalSP500
                sp = HistoricalSP500(
                    data_dir=self.config.get("paths", {}).get(
                        "raw_data", "./data/raw"
                    )
                )
                tickers = sp.fetch_current()
                if tickers:
                    return tickers
            except Exception:
                pass

        return DEFAULT_UNIVERSE


# ═══════════════════════════════════════
# PriceDownloader (pipeline.py 需要)
# ═══════════════════════════════════════

class PriceDownloader:
    """
    批量價格下載器。
    
    下載 Yahoo Finance 數據，儲存為 .parquet。
    支持全新下載 + 增量更新。
    """

    def __init__(self, config: dict):
        self.config = config
        self.prices_dir = Path(
            config.get("paths", {}).get("prices_dir", "./data/prices")
        )
        self.prices_dir.mkdir(parents=True, exist_ok=True)

        dl_cfg = config.get("data", {}).get("download", {})
        self.start_date = dl_cfg.get("start_date", "2015-01-01")
        self.batch_size = dl_cfg.get("batch_size", 20)
        self.batch_delay = dl_cfg.get("batch_delay", 2.0)
        self.max_retries = dl_cfg.get("max_retries", 3)

    def download_all(self, tickers: List[str]) -> None:
        """
        下載指定股票嘅全部歷史數據。
        """
        if yf is None:
            print("❌ yfinance 未安裝，無法下載")
            return

        total = len(tickers)
        success = 0
        failed = []

        print(f"⬇️ 下載 {total} 隻股票 | 起始: {self.start_date}")

        for i in range(0, total, self.batch_size):
            batch = tickers[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total - 1) // self.batch_size + 1
            print(f"  📦 批次 {batch_num}/{total_batches}: {batch}")

            for ticker in batch:
                ok = self._download_single(ticker, self.start_date)
                if ok:
                    success += 1
                else:
                    failed.append(ticker)

            if i + self.batch_size < total:
                time.sleep(self.batch_delay)

        print(f"✅ 下載完成: {success}/{total} 成功")
        if failed:
            print(f"❌ 失敗: {failed}")

    def update_existing(self, tickers: List[str]) -> None:
        """
        增量更新已有數據。
        """
        if yf is None:
            print("❌ yfinance 未安裝")
            return

        print(f"🔄 增量更新 {len(tickers)} 隻股票...")

        for ticker in tickers:
            parquet_path = self.prices_dir / f"{ticker}.parquet"

            try:
                existing = pd.read_parquet(parquet_path)
                if existing.empty:
                    self._download_single(ticker, self.start_date)
                    continue

                last_date = pd.to_datetime(existing.index[-1])
                new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                if pd.Timestamp(new_start) >= pd.Timestamp.today().normalize():
                    continue  # 已是最新

                new_data = self._fetch(ticker, new_start)
                if new_data is not None and not new_data.empty:
                    combined = pd.concat([existing, new_data])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    combined.to_parquet(parquet_path)
                    print(f"  ✅ {ticker}: +{len(new_data)} rows")

            except Exception as e:
                print(f"  ⚠️ {ticker} 更新失敗: {e}")
                self._download_single(ticker, self.start_date)

    def _download_single(self, ticker: str, start: str) -> bool:
        """下載單隻股票，儲存為 parquet"""
        df = self._fetch(ticker, start)
        if df is not None and not df.empty:
            path = self.prices_dir / f"{ticker}.parquet"
            df.to_parquet(path)
            print(f"  ✅ {ticker}: {len(df)} rows → {path.name}")
            return True
        else:
            print(f"  ⚠️ {ticker}: 無數據")
            return False

    def _fetch(
        self, ticker: str, start: str, end: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """從 yfinance 下載"""
        end_str = end or pd.Timestamp.today().strftime("%Y-%m-%d")

        for attempt in range(self.max_retries):
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end_str,
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue
                    return None

                # Flatten multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 確保 index 係 DatetimeIndex
                df.index = pd.to_datetime(df.index)
                df.index.name = "Date"
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]

                # 清理
                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                if "Volume" in df.columns:
                    df["Volume"] = pd.to_numeric(
                        df["Volume"], errors="coerce"
                    ).fillna(0)

                df = df.dropna(subset=["Close"])
                return df

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"  ❌ {ticker}: {e}")
                    return None

        return None


# ═══════════════════════════════════════
# DataLayer (其他模組用)
# ═══════════════════════════════════════

class DataLayer:
    """
    通用數據載入層。
    
    支持 CSV + Parquet 雙格式。
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        prices_dir: str = "data/prices",
        start_date: str = "2015-01-01",
        batch_size: int = 20,
        batch_delay: float = 2.0,
        max_retries: int = 3,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prices_dir = Path(prices_dir)
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        self.max_retries = max_retries

    def load_ticker(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        載入單隻股票。先查 parquet，再查 CSV。
        """
        # 先查 parquet
        parquet_path = self.prices_dir / f"{ticker}.parquet"
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                df = self._normalize(df)
                return self._filter_dates(df, start, end)
            except Exception:
                pass

        # 再查 CSV
        csv_path = self.cache_dir / f"{ticker}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["Date"])
                df = self._normalize(df)
                return self._filter_dates(df, start, end)
            except Exception:
                pass

        return pd.DataFrame()

    def load_multiple(
        self,
        tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for ticker in tickers:
            df = self.load_ticker(ticker, start, end)
            if not df.empty:
                result[ticker] = df
        return result

    def get_cached_tickers(self) -> List[str]:
        """列出所有有快取嘅股票（parquet + csv）"""
        tickers = set()
        for f in self.prices_dir.glob("*.parquet"):
            tickers.add(f.stem)
        for f in self.cache_dir.glob("*.csv"):
            tickers.add(f.stem)
        return sorted(tickers)

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """統一 DataFrame 格式"""
        if df.empty:
            return df

        # 確保有 Date column
        if "Date" not in df.columns:
            if df.index.name in ("Date", "date") or isinstance(
                df.index, pd.DatetimeIndex
            ):
                df = df.reset_index()
                if "date" in df.columns:
                    df = df.rename(columns={"date": "Date"})
            else:
                return pd.DataFrame()

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.drop_duplicates(subset=["Date"], keep="last")
        df = df.sort_values("Date").reset_index(drop=True)

        # Column 名統一
        colmap = {
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close",
            "volume": "Volume",
        }
        df.rename(columns=colmap, inplace=True)

        return df

    def _filter_dates(
        self, df: pd.DataFrame, start: Optional[str], end: Optional[str]
    ) -> pd.DataFrame:
        if df.empty or "Date" not in df.columns:
            return df
        if start:
            df = df[df["Date"] >= pd.Timestamp(start)]
        if end:
            df = df[df["Date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)