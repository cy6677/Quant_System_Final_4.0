import json
import os
import copy


def ensuredirs(path_or_obj):
    """遞迴建立所有路徑嘅父目錄"""
    if isinstance(path_or_obj, str):
        dirname = os.path.dirname(path_or_obj)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        return
    if isinstance(path_or_obj, dict):
        for v in path_or_obj.values():
            ensuredirs(v)
        return
    if isinstance(path_or_obj, (list, tuple)):
        for v in path_or_obj:
            ensuredirs(v)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_env_keys(config: dict) -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    resolved = {}
    for key, value in config.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_env_keys(value)
        elif isinstance(key, str) and key.endswith("_env"):
            real_key = key[:-4]
            env_val = os.environ.get(value, "")
            if not env_val:
                print(f"⚠️  環境變量 {value} 未設定，{real_key} 將為空字串")
            resolved[real_key] = env_val
        else:
            resolved[key] = value
    return resolved


def _validate_config(cfg: dict) -> dict:
    """
    輕量 schema 驗證（避免引入額外依賴）
    """
    # 必要節點
    cfg.setdefault("paths", {})
    cfg.setdefault("backtest", {})
    cfg.setdefault("universe", {})
    cfg.setdefault("download", {})

    # 型別與範圍修正
    cfg["backtest"]["initial_capital"] = float(
        cfg["backtest"].get("initial_capital", 100000.0)
    )
    if cfg["backtest"]["initial_capital"] <= 0:
        cfg["backtest"]["initial_capital"] = 100000.0

    cfg["backtest"]["calendar_ticker"] = str(
        cfg["backtest"].get("calendar_ticker", "SPY")
    ).upper()

    cfg["backtest"]["allow_fractional"] = bool(
        cfg["backtest"].get("allow_fractional", True)
    )

    cfg["backtest"]["commission_rate"] = float(
        cfg["backtest"].get("commission_rate", 0.001)
    )
    cfg["backtest"]["slippage"] = float(cfg["backtest"].get("slippage", 0.001))
    cfg["backtest"]["min_commission"] = float(
        cfg["backtest"].get("min_commission", 1.0)
    )

    cfg["download"]["max_workers"] = int(cfg["download"].get("max_workers", 10))
    if cfg["download"]["max_workers"] < 1:
        cfg["download"]["max_workers"] = 1

    # 路徑
    default_paths = {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "prices_dir": "data/prices",
        "universe_file": "data/universe.csv",
        "clean_universe_file": "data/clean_universe.csv",
        "log_file": "logs/system.log",
        "failed_file": "data/failed_tickers.csv",
    }
    for k, v in default_paths.items():
        cfg["paths"].setdefault(k, v)

    return cfg


def load_config(config_path: str = "config.yaml"):
    """
    統一優先讀 config.yaml，找不到再 fallback config.json
    """
    default_config = {
        "data": {
            "price": {
                "start_date": "2015-01-01",
                "end_date": None,
            }
        },
        "backtest": {
            "initial_capital": 100000.0,
            "calendar_ticker": "SPY",
            "allow_fractional": True,
            "commission_rate": 0.001,
            "slippage": 0.001,
            "min_commission": 1.0,
            "benchmark": "SPY",
        },
        "universe": {
            "extra_etfs": ["QQQ", "AAPL", "NVDA", "TSLA", "MSFT"],
            "use_historical": False,
            "cache_days": 7,
        },
        "data_quality": {
            "use_clean_universe": False,
            "max_missing_pct": 0.2,
            "max_spike_count": 5,
        },
        "paths": {
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "prices_dir": "data/prices",
            "universe_file": "data/universe.csv",
            "clean_universe_file": "data/clean_universe.csv",
            "log_file": "logs/system.log",
            "failed_file": "data/failed_tickers.csv",
        },
        "download": {"max_workers": 10},
        "cost_model": {},
    }

    # 自動 fallback
    candidates = [config_path]
    if config_path == "config.yaml":
        candidates.append("config.json")

    loaded = False
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith(".yaml") or path.endswith(".yml"):
                try:
                    import yaml
                except ImportError:
                    print("⚠️ 未安裝 PyYAML，無法讀取 yaml，將嘗試 json。")
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}
            else:
                with open(path, "r", encoding="utf-8") as f:
                    file_config = json.load(f)
            default_config = _deep_merge(default_config, file_config)
            loaded = True
            break
        except Exception as e:
            print(f"Warning: 無法讀取 {path}: {e}")

    if not loaded:
        print("ℹ️ 未找到 config 檔，使用預設配置。")

    default_config = _resolve_env_keys(default_config)
    default_config = _validate_config(default_config)
    return default_config


def loadconfig(_config_path=None):
    return load_config(_config_path or "config.yaml")