import os
import datetime
from typing import List, Dict, Optional, Set
import pandas as pd

try:
    from config import load_config, ensure_dirs
except ImportError:
    try:
        from config import loadconfig as load_config, ensuredirs as ensure_dirs
    except ImportError:
        pass

try:
    from layers.data_layer import UniverseProvider, PriceDownloader
except ImportError:

    class UniverseProvider:
        def __init__(self, config):
            pass

        def get_universe(self):
            return ["SPY", "AAPL", "MSFT"]

    class PriceDownloader:
        def __init__(self, config):
            pass

        def download_all(self, missing):
            pass

        def update_existing(self, to_update):
            pass

try:
    from layers.historical_sp500 import HistoricalSP500
except ImportError:

    class HistoricalSP500:
        def __init__(self, cachedir):
            pass

        def download(self):
            return None


class QuantPipeline:
    """
    機構級量化數據管線 (Data Pipeline Engine) v2 hardened

    升級
    ----
    1. is_trading 改為 vectorized 計算（加速）
    2. 加上市前標記 pre_ipo_flag，避免策略誤讀
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = load_config(config_path) if "load_config" in globals() else {}

        try:
            if "ensure_dirs" in globals():
                ensure_dirs(self.config)
        except Exception:
            pass

        self.universe_provider = UniverseProvider(self.config)
        self.price_downloader = PriceDownloader(self.config)

        self.historical_universe: Optional[Dict[pd.Timestamp, List[str]]] = None
        self.historical_df: Optional[pd.DataFrame] = None
        self.tickers: List[str] = []

        self.benchmark_ticker = self.config.get("backtest", {}).get(
            "benchmark", "SPY"
        )

        if self.config.get("universe", {}).get("use_historical", False):
            self._load_historical_universe()

    def _load_historical_universe(self) -> None:
        try:
            raw_dir = self.config.get("paths", {}).get("raw_data", "./data/raw")
            hist = HistoricalSP500(cachedir=raw_dir)
            df = hist.download()
            if (
                df is not None
                and "date" in df.columns
                and "symbol" in df.columns
            ):
                self.historical_df = df
                df["date"] = pd.to_datetime(df["date"])
                self.historical_universe = (
                    df.groupby("date")["symbol"].apply(list).to_dict()
                )
                print("✅ 成功載入歷史成分股 (Point-in-Time Universe)")
            else:
                print("⚠️ 歷史成分股格式不符，停用 point-in-time，改用最新成分股。")
        except Exception as e:
            print(f"⚠️ 載入歷史成分股失敗: {e}，改用最新成��股。")

    def update_data(
        self, min_ratio: float = 0.9, force_full_download: bool = False
    ) -> List[str]:
        print("🔍 正在獲取股票池 (Universe)...")
        try:
            self.tickers = self.universe_provider.get_universe()
        except Exception as e:
            print(f"⚠️ 獲取 Universe 失敗: {e}，將使用預設名單")
            self.tickers = [self.benchmark_ticker, "AAPL", "MSFT"]

        if self.benchmark_ticker not in self.tickers:
            self.tickers.append(self.benchmark_ticker)

        prices_dir = self.config.get("paths", {}).get("prices_dir", "./data/prices")
        os.makedirs(prices_dir, exist_ok=True)

        existing_files = set(os.listdir(prices_dir))
        missing_tickers = []
        outdated_tickers = []

        today = pd.Timestamp(datetime.date.today())
        threshold_days = 3

        print(f"🔄 檢查 {len(self.tickers)} 隻股票的數據新鮮度...")

        if force_full_download:
            missing_tickers = self.tickers
        else:
            for t in self.tickers:
                file_path = os.path.join(prices_dir, f"{t}.parquet")
                if f"{t}.parquet" not in existing_files:
                    missing_tickers.append(t)
                else:
                    try:
                        df = pd.read_parquet(file_path, columns=["Close"])
                        if df.empty:
                            missing_tickers.append(t)
                            continue
                        last_date = pd.to_datetime(df.index[-1])
                        days_diff = (today - last_date).days
                        if days_diff > threshold_days:
                            outdated_tickers.append(t)
                    except Exception:
                        missing_tickers.append(t)

        if missing_tickers:
            print(f"⬇️ 發現 {len(missing_tickers)} 隻股票缺少數據，開始全新下載...")
            try:
                self.price_downloader.download_all(missing_tickers)
            except AttributeError:
                print("⚠️ PriceDownloader 尚未實作 download_all 方法")

        if outdated_tickers:
            print(f"🔄 發現 {len(outdated_tickers)} 隻股票數據過舊 (> {threshold_days} 日)，開始增量更新...")
            try:
                if hasattr(self.price_downloader, "update_existing"):
                    self.price_downloader.update_existing(outdated_tickers)
                else:
                    self.price_downloader.download_all(outdated_tickers)
            except Exception as e:
                print(f"⚠️ 更新過舊數據失敗: {e}")

        if not missing_tickers and not outdated_tickers:
            print("✅ 所有股票數據已是最新狀態，無需下載。")

        return self.tickers

    def load_prices(
        self,
        tickers: Optional[List[str]] = None,
        align_dates: bool = True,
        fill_method: str = "ffill",
    ) -> Dict[str, pd.DataFrame]:
        if tickers is None:
            if not self.tickers:
                self.tickers = self.universe_provider.get_universe()
            tickers = self.tickers

        if self.benchmark_ticker not in tickers:
            tickers.append(self.benchmark_ticker)

        prices_dir = self.config.get("paths", {}).get("prices_dir", "./data/prices")
        raw_prices: Dict[str, pd.DataFrame] = {}

        print(f"📦 正在載入 {len(tickers)} 隻股票的價格數據至記憶體...")
        all_dates_set: Set[pd.Timestamp] = set()

        for t in tickers:
            path = os.path.join(prices_dir, f"{t}.parquet")
            if not os.path.exists(path):
                continue

            try:
                df = pd.read_parquet(path)
            except Exception:
                print(f"⚠️ 無法讀取 {path}，檔案可能損毀，已跳過。")
                continue

            if df.empty:
                continue

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            colmap = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df.rename(columns=colmap, inplace=True)

            if "Close" in df.columns:
                raw_prices[t] = df
                if align_dates:
                    all_dates_set.update(df.index)

        if align_dates and raw_prices:
            print("🔧 正在進行全局日期對齊 (Global Date Alignment) 與停牌清理...")

            if self.benchmark_ticker in raw_prices:
                global_index = raw_prices[self.benchmark_ticker].index
            else:
                global_index = pd.DatetimeIndex(sorted(list(all_dates_set)))

            aligned_prices: Dict[str, pd.DataFrame] = {}

            for t, df in raw_prices.items():
                original_index = df.index.copy()
                first_valid = original_index.min() if len(original_index) else None

                aligned_df = df.reindex(global_index)

                if fill_method == "ffill":
                    cols_to_ffill = ["Open", "High", "Low", "Close"]
                    existing_cols = [c for c in cols_to_ffill if c in aligned_df.columns]
                    aligned_df[existing_cols] = aligned_df[existing_cols].ffill()

                if "Volume" in aligned_df.columns:
                    aligned_df["Volume"] = aligned_df["Volume"].fillna(0)

                # vectorized is_trading
                aligned_df["is_trading"] = False
                original_mask = aligned_df.index.isin(original_index)
                if "Volume" in aligned_df.columns:
                    aligned_df["is_trading"] = original_mask & (aligned_df["Volume"] > 0)
                else:
                    aligned_df["is_trading"] = original_mask

                # 上市前標記（防止策略誤用）
                aligned_df["pre_ipo_flag"] = False
                if first_valid is not None:
                    aligned_df.loc[aligned_df.index < first_valid, "pre_ipo_flag"] = True

                aligned_df = aligned_df.dropna(subset=["Close"])

                if not aligned_df.empty:
                    aligned_prices[t] = aligned_df

            print(f"✅ 成功載入並對齊 {len(aligned_prices)} 隻股票數據！")
            return aligned_prices

        else:
            for t in raw_prices:
                if "Volume" in raw_prices[t].columns:
                    raw_prices[t]["is_trading"] = raw_prices[t]["Volume"] > 0
                else:
                    raw_prices[t]["is_trading"] = True
                raw_prices[t]["pre_ipo_flag"] = False

            print(f"✅ 成功載入 {len(raw_prices)} 隻股票數據 (無對齊)！")
            return raw_prices


quant_pipeline = QuantPipeline