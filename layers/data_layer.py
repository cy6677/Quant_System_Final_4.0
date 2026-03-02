# layers/data_layer.py
import datetime
import os
from io import StringIO
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm

try:
    from config import load_config
except ImportError:
    try:
        from config import loadconfig as load_config
    except ImportError:
        pass


class UniverseProvider:
    """提供標普 500 最新成分股名單"""

    def __init__(self, config=None):
        self.config = config or (
            load_config()
            if "load_config" in globals()
            else {
                "paths": {"universe_file": "data/universe.csv"},
                "universe": {"cache_days": 7},
            }
        )

    def _is_cache_valid(self, cache_file: str, days: int) -> bool:
        if not os.path.exists(cache_file):
            return False
        last_modified = os.path.getmtime(cache_file)
        days_old = (datetime.datetime.now().timestamp() - last_modified) / 86400
        return days_old < days

    def build_universe(self) -> pd.DataFrame:
        cache_file = self.config.get("paths", {}).get(
            "universe_file", "data/universe.csv"
        )
        cache_days = self.config.get("universe", {}).get("cache_days", 7)
        os.makedirs(
            os.path.dirname(cache_file) if os.path.dirname(cache_file) else ".",
            exist_ok=True,
        )

        if self._is_cache_valid(cache_file, cache_days):
            print("📂 使用快取股票池 (Universe)")
            return pd.read_csv(cache_file)

        print("🌐 從 Wikipedia 下載最新 S&P 500 成分股...")
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            headers = {
                "User-Agent": "Mozilla/5.0"
            }
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            df["Ticker"] = df["Symbol"].str.replace(".", "-", regex=False)
            df[["Ticker", "Security"]].to_csv(cache_file, index=False)
            return df
        except Exception as e:
            print(f"⚠️ 下載成分股失敗: {e}，嘗試載入舊快取...")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame(
                {
                    "Ticker": ["SPY", "QQQ"],
                    "Security": ["SPDR S&P 500", "Invesco QQQ"],
                }
            )

    def get_universe(self) -> List[str]:
        df = self.build_universe()
        if "Ticker" in df.columns:
            return df["Ticker"].dropna().unique().tolist()
        return []


class PriceDownloader:
    """
    多線程 Yahoo Finance 價格下載器 (支援全量與增量更新)
    hardened:
    1) OHLC 全欄位 adjusted
    2) 嚴格 QC
    3) 異常跳空打 flag（不直接刪）
    """

    def __init__(self, config=None):
        self.config = config or (
            load_config()
            if "load_config" in globals()
            else {
                "paths": {"prices_dir": "data/prices"},
                "download": {"max_workers": 10},
            }
        )
        self.prices_dir = self.config.get("paths", {}).get(
            "prices_dir", "data/prices"
        )
        os.makedirs(self.prices_dir, exist_ok=True)

    @staticmethod
    def _quality_check(df: pd.DataFrame) -> pd.DataFrame:
        required = ["Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in df.columns:
                return pd.DataFrame()

        # 基本清理
        df = df.copy()
        for c in ["Open", "High", "Low", "Close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        # 價格>0
        df = df[
            (df["Open"] > 0)
            & (df["High"] > 0)
            & (df["Low"] > 0)
            & (df["Close"] > 0)
        ]

        # OHLC 邏輯一致性
        logic_ok = (
            (df["High"] >= df[["Open", "Close"]].max(axis=1))
            & (df["Low"] <= df[["Open", "Close"]].min(axis=1))
            & (df["High"] >= df["Low"])
        )
        df = df[logic_ok]

        # Volume 不可負
        df = df[df["Volume"].fillna(0) >= 0]

        # 異常跳空 flag（保留數據，讓策略自己決定是否用）
        ret = df["Close"].pct_change()
        df["anomaly_spike_flag"] = (
            (ret > 3.0) | (ret < -0.8)
        ).fillna(False)

        return df

    def download_one(
        self, ticker: str, start_date: str = "2000-01-01"
    ) -> tuple:
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=start_date, auto_adjust=False)

            if df is None or df.empty:
                return ticker, None

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.columns = [str(c).title() for c in df.columns]

            if "Adj Close" in df.columns and "Close" in df.columns:
                adj_factor = df["Adj Close"] / df["Close"]
                adj_factor = adj_factor.fillna(1.0)
                adj_factor = adj_factor.replace([float("inf"), float("-inf")], 1.0)

                for col in ["Open", "High", "Low", "Close"]:
                    if col in df.columns:
                        df[col] = df[col] * adj_factor

                df = df.drop(columns=["Adj Close"], errors="ignore")

            keep_cols = ["Open", "High", "Low", "Close", "Volume"]
            existing_keep = [c for c in keep_cols if c in df.columns]
            df = df[existing_keep]

            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            df = self._quality_check(df)
            if df.empty:
                return ticker, None

            return ticker, df

        except Exception:
            return ticker, None

    def download_all(self, tickers: List[str]) -> None:
        max_workers = self.config.get("download", {}).get("max_workers", 10)
        failed = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ticker, df in tqdm(
                executor.map(lambda t: self.download_one(t, "2000-01-01"), tickers),
                total=len(tickers),
                desc="⬇️ 全量下載數據",
            ):
                if df is None or df.empty:
                    failed.append(ticker)
                else:
                    out = os.path.join(self.prices_dir, f"{ticker}.parquet")
                    df.to_parquet(out)

        if failed:
            print(f"⚠️ {len(failed)} 隻股票下載失敗（已下市或 Ticker 錯誤）")

    def update_existing(self, tickers: List[str]) -> None:
        max_workers = self.config.get("download", {}).get("max_workers", 10)
        failed = []
        updated_count = 0

        def _update_one(ticker: str) -> Tuple[str, bool]:
            file_path = os.path.join(self.prices_dir, f"{ticker}.parquet")
            try:
                existing_df = pd.read_parquet(file_path)
                if existing_df.empty:
                    _, new_df = self.download_one(ticker, "2000-01-01")
                    if new_df is not None and not new_df.empty:
                        new_df.to_parquet(file_path)
                        return ticker, True
                    return ticker, False

                last_date = existing_df.index[-1]
                fetch_start = (last_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
                _, new_df = self.download_one(ticker, fetch_start)

                if new_df is not None and not new_df.empty:
                    combined_df = pd.concat([existing_df, new_df])
                    combined_df = combined_df[
                        ~combined_df.index.duplicated(keep="last")
                    ].sort_index()
                    combined_df = self._quality_check(combined_df)
                    if combined_df.empty:
                        return ticker, False
                    combined_df.to_parquet(file_path)
                    return ticker, True

                return ticker, False

            except Exception:
                _, new_df = self.download_one(ticker, "2000-01-01")
                if new_df is not None and not new_df.empty:
                    new_df.to_parquet(file_path)
                    return ticker, True
                return ticker, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for ticker, success in tqdm(
                executor.map(_update_one, tickers),
                total=len(tickers),
                desc="🔄 增量更新數據",
            ):
                if success:
                    updated_count += 1
                else:
                    failed.append(ticker)

        print(f"✅ 成功更新 {updated_count} 隻股票。")
        if failed:
            print(f"⚠️ {len(failed)} 隻股票更新失敗。")