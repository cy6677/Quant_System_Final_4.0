"""
data_hub.py - 統一數據入口 (Gateway) v2 hardened
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from layers.data_layer import PriceDownloader

try:
    from layers.technical_layer import TechnicalIndicator
except ImportError:

    class TechnicalIndicator:
        def __init__(self, close_df, ohlcv_df):
            pass


try:
    from config import load_config
except ImportError:
    try:
        from config import loadconfig as load_config
    except ImportError:
        pass

try:
    from engine.pipeline import QuantPipeline
except ImportError:
    pass


class PriceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.prices_dir = Path(
            config.get("paths", {}).get("prices_dir", "data/prices")
        )

        try:
            self.pipeline = QuantPipeline("config.yaml")
        except Exception:
            self.pipeline = None

        # key: (align_dates, fill_method) -> prices_dict
        self._prices_cache: Dict[Tuple[bool, str], Dict[str, pd.DataFrame]] = {}

    def _invalidate_cache(self):
        self._prices_cache = {}

    def download(
        self,
        symbols: Optional[List[str]] = None,
        force: bool = False,
    ) -> None:
        if self.pipeline:
            if symbols:
                self.pipeline.tickers = symbols
            self.pipeline.update_data(force_full_download=force)
        else:
            downloader = PriceDownloader(self.config)
            if force or symbols is None:
                symbols = symbols or []
                downloader.download_all(symbols)
            else:
                downloader.update_existing(symbols)

        self._invalidate_cache()

    def _get_prices_dict(
        self,
        tickers: Optional[List[str]] = None,
        align_dates: bool = True,
        fill_method: str = "ffill",
    ) -> Dict[str, pd.DataFrame]:
        if self.pipeline is None:
            return {}

        # 有指定 tickers 時，不使用全量 cache（避免錯配）
        if tickers is not None:
            return self.pipeline.load_prices(
                tickers=tickers,
                align_dates=align_dates,
                fill_method=fill_method,
            )

        cache_key = (align_dates, fill_method)
        if cache_key not in self._prices_cache:
            self._prices_cache[cache_key] = self.pipeline.load_prices(
                align_dates=align_dates,
                fill_method=fill_method,
            )
        return self._prices_cache[cache_key]

    def load(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        prices_dict = self._get_prices_dict()
        if not prices_dict:
            return pd.DataFrame()

        close_series = {
            ticker: df["Close"]
            for ticker, df in prices_dict.items()
            if "Close" in df.columns
        }
        df_close = pd.DataFrame(close_series)

        if start:
            df_close = df_close[df_close.index >= pd.to_datetime(start)]
        if end:
            df_close = df_close[df_close.index <= pd.to_datetime(end)]

        df_close.index.name = "date"
        df_close.columns.name = "Symbol"
        return df_close

    def load_ohlcv(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        prices_dict = self._get_prices_dict()
        if not prices_dict:
            return pd.DataFrame()

        dfs = []
        for ticker, df in prices_dict.items():
            temp = df.copy()
            temp["Symbol"] = ticker
            dfs.append(temp)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs)
        combined.index.name = "date"
        combined = (
            combined.reset_index()
            .set_index(["date", "Symbol"])
            .sort_index()
        )

        if start:
            combined = combined.loc[
                combined.index.get_level_values("date")
                >= pd.to_datetime(start)
            ]
        if end:
            combined = combined.loc[
                combined.index.get_level_values("date")
                <= pd.to_datetime(end)
            ]

        return combined


class DataHub:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or (
            load_config() if "load_config" in globals() else {}
        )
        self.price = PriceManager(self.config)

    def load_price(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.price.load(start, end)

    def load_ohlcv(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.price.load_ohlcv(start, end)

    def build_technical(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> TechnicalIndicator:
        close_df = self.price.load(start, end)
        ohlcv_df = self.price.load_ohlcv(start, end)
        return TechnicalIndicator(close_df, ohlcv_df)

    def load_price_dict(
        self, tickers: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        return self.price._get_prices_dict(tickers=tickers)