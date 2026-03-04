"""
config.py
=========
Config loader — 支持 YAML 同 JSON 格式。
"""
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(
    path: str = "config.json",
    fallback: str = "config.yaml",
) -> Dict[str, Any]:
    """
    載入設定檔。優先 JSON，fallback 到 YAML。
    """
    p = Path(path)

    if p.exists():
        return _load_file(p)

    fp = Path(fallback)
    if fp.exists():
        return _load_file(fp)

    print(f"⚠️ 找唔到設定檔 {path} 或 {fallback}，使用預設值")
    return _default_config()


def _load_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            print(f"✅ 已載入設定: {path}")
            return cfg or {}
        except ImportError:
            print("⚠️ 未安裝 PyYAML，嘗試 JSON fallback")
            return _default_config()

    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"✅ 已載入設定: {path}")
        return cfg

    else:
        raise ValueError(f"不支持嘅設定檔格式: {suffix}")


def ensure_dirs(config: Dict[str, Any]) -> None:
    """
    確保所有必要嘅資料夾存在。
    """
    paths = config.get("paths", {})

    dirs_to_create = [
        paths.get("raw_data", "./data/raw"),
        paths.get("prices_dir", "./data/prices"),
        paths.get("results_dir", "./results"),
        paths.get("logs_dir", "./logs"),
        paths.get("cache_dir", "./data/cache"),
    ]

    # 兼容舊格式
    data_cfg = config.get("data", {})
    if "cache_dir" in data_cfg:
        dirs_to_create.append(data_cfg["cache_dir"])

    output_cfg = config.get("output", {})
    for key in ["results_dir", "logs_dir"]:
        if key in output_cfg:
            dirs_to_create.append(output_cfg[key])

    for d in dirs_to_create:
        if d:
            os.makedirs(d, exist_ok=True)

    print("✅ 所有資料夾已確認存在")


def get_nested(cfg: Dict, *keys, default=None) -> Any:
    """安全取得巢狀 config 值。"""
    current = cfg
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return default
    return current


def _default_config() -> Dict[str, Any]:
    """預設設定"""
    return {
        "paths": {
            "raw_data": "./data/raw",
            "prices_dir": "./data/prices",
            "results_dir": "./results",
            "logs_dir": "./logs",
            "cache_dir": "./data/cache",
        },
        "data": {
            "cache_dir": "data/cache",
            "sp500_dir": "data/sp500",
            "download": {
                "start_date": "2015-01-01",
                "batch_size": 20,
                "batch_delay": 2.0,
                "max_retries": 3,
            },
        },
        "backtest": {
            "initial_capital": 100000.0,
            "commission_rate": 0.001,
            "slippage_bps": 10.0,
            "execution_delay": 1,
            "benchmark": "SPY",
        },
        "universe": {
            "use_historical": False,
            "mode": "custom",
        },
        "optimizer": {
            "default_metric": "robust_sharpe",
            "n_trials": 50,
            "timeout_sec": 1800,
            "consistency_penalty": 0.15,
        },
        "risk": {
            "daily_loss_trigger": 0.02,
            "drawdown_l2": 0.12,
            "drawdown_l3": 0.18,
            "cooldown_days": 30,
            "max_single_position": 0.08,
            "max_total_exposure": 1.00,
        },
        "output": {
            "results_dir": "results",
            "logs_dir": "logs",
            "log_level": "INFO",
        },
    }