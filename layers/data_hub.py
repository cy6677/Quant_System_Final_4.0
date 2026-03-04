"""
layers/data_hub.py
==================
數據中心 — 統一入口，連接 DataLayer 同 Pipeline。

所有 script 同 strategy 都透過 DataHub 取數據。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import get_nested
from layers.data_layer import DataLayer
from layers.technical_layer import add_all_indicators


# ─── 預設股票池 ───
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


class DataHub:
    """
    統一數據入口。

    Parameters
    ----------
    cfg : dict
        config.yaml 載入嘅全局設定
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        cache_dir = get_nested(cfg, "data", "cache_dir", default="data/cache")
        start_date = get_nested(cfg, "data", "download", "start_date", default="2015-01-01")

        self._data_layer = DataLayer(
            cache_dir=cache_dir,
            start_date=start_date,
            batch_size=get_nested(cfg, "data", "download", "batch_size", default=20),
            batch_delay=get_nested(cfg, "data", "download", "batch_delay", default=2.0),
            max_retries=get_nested(cfg, "data", "download", "max_retries", default=3),
        )

    def get_universe_tickers(self) -> List[str]:
        """回傳當前 universe 嘅股票清單"""
        mode = get_nested(self.cfg, "universe", "mode", default="custom")

        if mode == "custom":
            custom = get_nested(self.cfg, "universe", "custom_tickers", default=None)
            if custom and isinstance(custom, list):
                return sorted(set(custom))

        # fallback / sp500 mode: 用快取中有嘅股票
        cached = self._data_layer.get_cached_tickers()
        if cached:
            return cached

        return DEFAULT_UNIVERSE

    def load_price_dict(
        self,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        add_indicators: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        載入所有股票嘅價格數據。

        Parameters
        ----------
        tickers : list, optional
            股票清單。None = 用 universe 設定。
        start : str, optional
            開始日期
        end : str, optional
            結束日期
        add_indicators : bool
            是否加入技術指標

        Returns
        -------
        Dict[str, pd.DataFrame]
        """
        if tickers is None:
            tickers = self.get_universe_tickers()

        print(f"📊 載入 {len(tickers)} 隻股票...")

        result = {}
        loaded = 0
        failed = 0

        for ticker in tickers:
            df = self._data_layer.load_ticker(ticker, start, end)
            if df is not None and not df.empty:
                if add_indicators:
                    df = add_all_indicators(df)
                result[ticker] = df
                loaded += 1
            else:
                failed += 1

        print(f"✅ 載入完成: {loaded} 成功, {failed} 失敗")
        return result

    def load_single(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        add_indicators: bool = False,
    ) -> pd.DataFrame:
        """載入單隻股票"""
        df = self._data_layer.load_ticker(ticker, start, end)
        if df is not None and not df.empty and add_indicators:
            df = add_all_indicators(df)
        return df if df is not None else pd.DataFrame()