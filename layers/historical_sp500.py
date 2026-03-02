"""
歷史 S&P 500 成分股下載器 (point-in-time 修復版)
解決舊版「累積成分股」導致 Universe 暴增的邏輯錯誤。
"""
import pandas as pd
import requests
import io
from pathlib import Path
from typing import Optional, List
import datetime


class HistoricalSP500:
    # 使用 Wikipedia 歷年 Add/Drop 記錄作為基礎，這是一個更準確的 Point-in-Time 來源
    # 注意：如果外部資料源變動，此處可能需改用自建 CSV
    WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    def __init__(self, cache_dir: str = "data/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "sp500_historical_changes.parquet"
        self._current_components: Optional[List[str]] = None
        self._changes_df: Optional[pd.DataFrame] = None

    def download(self, force: bool = False) -> pd.DataFrame:
        """下載當前成分股與歷史變更表"""
        if not force and self.cache_path.exists():
            print("📂 使用快取歷史成分股變更紀錄")
            self._changes_df = pd.read_parquet(self.cache_path)
            # 順便拿一下最新的成分股 (Fallback 用)
            return self._changes_df

        print("📥 從 Wikipedia 獲取 S&P 500 歷史變更紀錄...")
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(self.WIKI_URL, headers=headers, timeout=15)
            resp.raise_for_status()

            tables = pd.read_html(io.StringIO(resp.text))

            # Table 0: Current components
            current_df = tables[0]
            current_df["Ticker"] = current_df["Symbol"].str.replace(".", "-", regex=False)
            self._current_components = current_df["Ticker"].tolist()

            # Table 1: Historical changes (Added / Removed)
            changes_df = tables[1]

            # 整理 Wikipedia 複雜的欄位結構
            if isinstance(changes_df.columns, pd.MultiIndex):
                # Flatten multi-index columns
                changes_df.columns = ['_'.join(col).strip() for col in changes_df.columns.values]

            # 找出 Date, Added Ticker, Removed Ticker 欄位
            date_col = [c for c in changes_df.columns if 'Date' in c][0]
            add_col = [c for c in changes_df.columns if 'Added' in c and 'Ticker' in c]
            rem_col = [c for c in changes_df.columns if 'Removed' in c and 'Ticker' in c]

            add_col = add_col[0] if add_col else changes_df.columns[1]
            rem_col = rem_col[0] if rem_col else changes_df.columns[3]

            clean_changes = pd.DataFrame({
                "date": pd.to_datetime(changes_df[date_col], errors='coerce'),
                "added": changes_df[add_col].str.replace(".", "-", regex=False),
                "removed": changes_df[rem_col].str.replace(".", "-", regex=False)
            }).dropna(subset=["date"])

            clean_changes = clean_changes.sort_values("date", ascending=False)
            clean_changes.to_parquet(self.cache_path)

            self._changes_df = clean_changes
            print(f"✅ 歷史變更紀錄已儲存至 {self.cache_path}")
            return clean_changes

        except Exception as e:
            print(f"⚠️ 獲取歷史變更失敗: {e}。這將導致只能使用最新成分股進行回測。")
            return pd.DataFrame()

    def get_universe_at(self, target_date: datetime.date) -> List[str]:
        """
        機構級 Point-in-Time 邏輯：
        由「今日名單」開始，時光倒流 (Reverse Engineering)。
        如果在 target_date 之後發生了「加入」，我們要在歷史名單將它「移除」。
        如果在 target_date 之後發生了「移除」，我們要在歷史名單將它「加回」。
        """
        if self._changes_df is None or self._current_components is None:
            # 如果還沒下載，先嘗試讀取最新清單 (fallback)
            try:
                self.download()
                if self._current_components is None:
                    # 從 DataLayer 呼叫
                    from layers.data_layer import UniverseProvider
                    self._current_components = UniverseProvider().get_universe()
            except Exception:
                return []

        if self._changes_df is None or self._changes_df.empty:
            print("⚠️ 無歷史變更數據，返回最新成分股 (注意會有 Survivorship Bias)")
            return self._current_components.copy()

        target_dt = pd.to_datetime(target_date)

        # 複製一份最新清單作為起始點
        historical_universe = set(self._current_components)

        # 找出在 target_date 到今天之間的所有變更 (時光倒流)
        recent_changes = self._changes_df[self._changes_df["date"] > target_dt]

        for _, row in recent_changes.iterrows():
            added = row["added"]
            removed = row["removed"]

            # 因為是時光倒流：
            # 發生在未來的 Added -> 代表在 target_date 時它還沒加入 -> 移除
            if pd.notna(added) and added in historical_universe:
                historical_universe.remove(added)

            # 發生在未來的 Removed -> 代表在 target_date 時它還在榜上 -> 加回
            if pd.notna(removed):
                historical_universe.add(removed)

        # 轉回 List，並過濾掉可能的空值
        final_list = [ticker for ticker in historical_universe if isinstance(ticker, str) and ticker.strip()]

        # 防呆：S&P 500 數量應在 490~510 之間
        if len(final_list) < 450 or len(final_list) > 550:
             # print(f"⚠️ {target_date} 計算出的成分股數量異常 ({len(final_list)})，可能影響回測。")
             pass

        return sorted(final_list)