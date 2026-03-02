import pandas as pd
import numpy as np
from pathlib import Path


class TechnicalIndicator:
    def __init__(self, price_close_df, ohlcv_df=None):
        self.price_close_df = price_close_df          # pivot table: date x ticker
        self.ohlcv_df = ohlcv_df                      # multi-index (date, ticker)
        self.indicators = {}                           # name -> DataFrame (date x ticker)

    def add_rsi(self, length=14):
        """計算 RSI (純 Pandas 實現)，回傳 date x ticker DataFrame"""
        rsi_list = []
        for ticker in self.price_close_df.columns:
            series = self.price_close_df[ticker].astype(float).dropna()
            if len(series) < length + 1:
                continue
            
            # 計算每日變化
            delta = series.diff()
            up = delta.where(delta > 0, 0.0)
            down = -delta.where(delta < 0, 0.0)
            
            # 使用 Wilder's Smoothing (RMA) 計算平均
            alpha = 1.0 / length
            roll_up = up.ewm(alpha=alpha, adjust=False).mean()
            roll_down = down.ewm(alpha=alpha, adjust=False).mean()
            
            rs = roll_up / roll_down
            rsi = 100.0 - (100.0 / (1.0 + rs))
            
            rsi.name = ticker
            rsi_list.append(rsi)
            
        if rsi_list:
            rsi_df = pd.concat(rsi_list, axis=1)
        else:
            rsi_df = pd.DataFrame(index=self.price_close_df.index)
            
        self.indicators[f'RSI_{length}'] = rsi_df
        return rsi_df

    def add_bbands(self, length=20, std=2):
        """計算布林通道 (純 Pandas 實現)"""
        df = self.price_close_df.astype(float)
        
        # 中軌 (Simple Moving Average)
        bbm = df.rolling(window=length).mean()
        # 標準差 (Standard Deviation)
        bstd = df.rolling(window=length).std(ddof=0)
        
        # 上下軌
        bbl = bbm - (std * bstd)
        bbu = bbm + (std * bstd)
        
        self.indicators[f'BBL_{length}'] = bbl
        self.indicators[f'BBM_{length}'] = bbm
        self.indicators[f'BBU_{length}'] = bbu
        
        # 將三者合併返回，方便外部呼叫
        return pd.concat([bbl, bbm, bbu], axis=1, keys=['BBL', 'BBM', 'BBU'])

    def add_atr(self, length=14):
        """計算 ATR (純 Pandas 實現)，回傳 date x ticker DataFrame"""
        if self.ohlcv_df is None:
            raise ValueError("ATR 需要 OHLCV 數據")

        high = self.ohlcv_df['High'].unstack(level=1).astype(float)
        low = self.ohlcv_df['Low'].unstack(level=1).astype(float)
        close = self.ohlcv_df['Close'].unstack(level=1).astype(float)

        atr_list = []
        for ticker in close.columns:
            if ticker in high and ticker in low:
                h = high[ticker].dropna()
                l = low[ticker].dropna()
                c = close[ticker].dropna()
                common_idx = h.index.intersection(l.index).intersection(c.index)
                if len(common_idx) < length + 1:
                    continue
                h = h.loc[common_idx]
                l = l.loc[common_idx]
                c = c.loc[common_idx]
                
                # 計算 True Range (TR)
                tr1 = h - l
                tr2 = (h - c.shift(1)).abs()
                tr3 = (l - c.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # ATR 就是 TR 嘅 Wilder's Smoothing
                atr = tr.ewm(alpha=1.0/length, adjust=False).mean()
                atr.name = ticker
                atr_list.append(atr)

        if atr_list:
            atr_df = pd.concat(atr_list, axis=1)
        else:
            atr_df = pd.DataFrame(index=close.index)

        self.indicators[f'ATR_{length}'] = atr_df
        return atr_df

    def add_adx(self, length=14):
        """計算 ADX (純 Pandas 實現)，回傳 date x ticker DataFrame"""
        if self.ohlcv_df is None:
            raise ValueError("ADX 需要 OHLCV 數據")

        high = self.ohlcv_df['High'].unstack(level=1).astype(float)
        low = self.ohlcv_df['Low'].unstack(level=1).astype(float)
        close = self.ohlcv_df['Close'].unstack(level=1).astype(float)

        adx_list = []
        for ticker in close.columns:
            if ticker in high and ticker in low:
                h = high[ticker].dropna()
                l = low[ticker].dropna()
                c = close[ticker].dropna()
                common_idx = h.index.intersection(l.index).intersection(c.index)
                if len(common_idx) < length + 1:
                    continue
                h = h.loc[common_idx]
                l = l.loc[common_idx]
                c = c.loc[common_idx]
                
                # 計算 +DM 同 -DM
                up_move = h - h.shift(1)
                down_move = l.shift(1) - l
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                
                plus_dm = pd.Series(plus_dm, index=h.index)
                minus_dm = pd.Series(minus_dm, index=h.index)
                
                # 計算 TR
                tr1 = h - l
                tr2 = (h - c.shift(1)).abs()
                tr3 = (l - c.shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                alpha = 1.0 / length
                atr = tr.ewm(alpha=alpha, adjust=False).mean()
                
                # 計算 +DI, -DI, DX
                plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
                minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
                
                dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
                adx = dx.ewm(alpha=alpha, adjust=False).mean()
                
                adx.name = ticker
                adx_list.append(adx)

        if adx_list:
            adx_df = pd.concat(adx_list, axis=1)
        else:
            adx_df = pd.DataFrame(index=close.index)

        self.indicators[f'ADX_{length}'] = adx_df
        return adx_df

    def add_vwap(self):
        """計算 VWAP (純 Pandas 實現)，回傳 date x ticker DataFrame"""
        if self.ohlcv_df is None:
            raise ValueError("VWAP 需要 OHLCV 數據")

        high = self.ohlcv_df['High'].unstack(level=1).astype(float)
        low = self.ohlcv_df['Low'].unstack(level=1).astype(float)
        close = self.ohlcv_df['Close'].unstack(level=1).astype(float)
        volume = self.ohlcv_df['Volume'].unstack(level=1).astype(float)

        vwap_list = []
        for ticker in close.columns:
            if ticker in high and ticker in low and ticker in volume:
                h = high[ticker].dropna()
                l = low[ticker].dropna()
                c = close[ticker].dropna()
                v = volume[ticker].dropna()
                common_idx = h.index.intersection(l.index).intersection(c.index).intersection(v.index)
                if len(common_idx) < 1:
                    continue
                h = h.loc[common_idx]
                l = l.loc[common_idx]
                c = c.loc[common_idx]
                v = v.loc[common_idx]
                
                # 典型價格
                tp = (h + l + c) / 3.0
                
                # 累積 VWAP 計算
                cv_tp = (tp * v).cumsum()
                cv = v.cumsum()
                vwap = cv_tp / cv
                
                vwap.name = ticker
                vwap_list.append(vwap)

        if vwap_list:
            vwap_df = pd.concat(vwap_list, axis=1)
        else:
            vwap_df = pd.DataFrame(index=close.index)

        self.indicators['VWAP'] = vwap_df
        return vwap_df

    def save_indicators(self, out_dir):
        """儲存每個指標為獨立 parquet 檔案"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self.indicators.items():
            if df is None or df.empty:
                raise ValueError(f"Indicator {name} is None/empty")
            df.to_parquet(out_dir / f"{name}.parquet")

    def to_unified_frame(self):
        """合併所有指標為一個 MultiIndex DataFrame (indicator, ticker)"""
        frames = []
        for name, df in self.indicators.items():
            if df.empty:
                continue
            # 轉成 MultiIndex columns: (indicator, ticker)
            df_multi = df.copy()
            df_multi.columns = pd.MultiIndex.from_product([[name], df.columns])
            frames.append(df_multi)
        if not frames:
            return pd.DataFrame()
        unified = pd.concat(frames, axis=1)
        return unified

    def save_unified(self, out_path):
        """儲存合併後嘅指標至單一 parquet 檔案"""
        unified = self.to_unified_frame()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        unified.to_parquet(out_path)
        return unified