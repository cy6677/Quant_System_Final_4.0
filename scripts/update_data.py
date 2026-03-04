#!/usr/bin/env python3
import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from engine.pipeline import QuantPipeline
except ImportError as e:
    print(f"❌ 嚴重錯誤: 無法載入 pipeline.py ({e})")
    print(f"🔍 系統當前搜尋路徑: {PROJECT_ROOT}")
    sys.exit(1)


def run_update(force_full: bool = False):
    print("🚀 啟動量化數據管線 (Quant Pipeline)...")
    try:
        pipeline = QuantPipeline(config_path="config.yaml")
        # ✅ P1 FIX: 傳入 force_full_download 參數
        tickers = pipeline.update_data(
            min_ratio=0.9, force_full_download=force_full
        )
        print(f"✅ 數據管線更新完成！共 {len(tickers)} 隻股票。")
    except Exception as e:
        print(f"❌ 數據更新失敗: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="量化系統數據更新器")
    parser.add_argument(
        "--force", action="store_true", help="強制重新下載所有數據"
    )
    args = parser.parse_args()

    run_update(force_full=args.force)