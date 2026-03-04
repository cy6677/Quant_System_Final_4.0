"""
backtest/optimizer.py
=====================
參數優化器 — Optuna + Walk-Forward Optimization (WFO)

特點：
- Bayesian optimization (TPE sampler)
- Consistency penalty (防止過擬合)
- Walk-Forward / Rolling-window validation
- 自動儲存最佳參數
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from backtest.universal_backtester import (
    UniversalBacktester,
    TransactionCostModel,
    PerformanceAnalyzer,
)


class StrategyOptimizer:
    """
    Optuna-based 策略參數優化器。

    Parameters
    ----------
    strategy_class : type
        策略類 (BaseStrategy 嘅子類)
    param_space : dict
        參數搜尋空間
    prices_dict : dict
        價格數據
    start_date : str
    end_date : str
    metric : str
        優化目標 (sharpe / robust_sharpe / calmar / sortino)
    n_trials : int
        搜尋次數
    cost_model : TransactionCostModel
    optuna_timeout_sec : int
        超時秒數
    consistency_penalty : float
        Consistency penalty 權重
    """

    def __init__(
        self,
        strategy_class,
        param_space: Dict,
        prices_dict: Dict[str, pd.DataFrame],
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
        metric: str = "robust_sharpe",
        n_trials: int = 50,
        cost_model: Optional[TransactionCostModel] = None,
        optuna_timeout_sec: int = 1800,
        consistency_penalty: float = 0.15,
    ):
        if optuna is None:
            raise ImportError("需要安裝 optuna: pip install optuna")

        self.strategy_class = strategy_class
        self.param_space = param_space
        self.prices_dict = prices_dict
        self.start_date = start_date
        self.end_date = end_date
        self.metric = metric
        self.n_trials = n_trials
        self.cost_model = cost_model or TransactionCostModel()
        self.timeout = optuna_timeout_sec
        self.consistency_penalty = consistency_penalty

    def optimize_single_period(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        save_best_to: Optional[Path] = None,
    ) -> Dict:
        """
        單期優化: Train 搵最佳參數 → Test 驗證。

        Returns
        -------
        Dict with:
            best_params, best_metric_insample, oos_metrics,
            consistency_adjusted_score
        """
        def objective(trial):
            params = self._sample_params(trial)
            score = self._evaluate(params, train_start, train_end)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True,
            )

        best_params = study.best_params
        best_value = study.best_value

        # ── Consistency Penalty ──
        # 用 top 10% 嘅 trials 嘅 std 做 penalty
        values = [t.value for t in study.trials if t.value is not None]
        if len(values) >= 5:
            top_k = sorted(values, reverse=True)[:max(3, len(values) // 10)]
            std_top = np.std(top_k)
            penalty = self.consistency_penalty * std_top
            adjusted = best_value - penalty
        else:
            penalty = 0.0
            adjusted = best_value

        # ── OOS Test ──
        oos_metrics = self._full_backtest(best_params, test_start, test_end)

        result = {
            "best_params": best_params,
            "best_metric_insample": round(float(best_value), 4),
            "consistency_penalty": round(float(penalty), 4),
            "consistency_adjusted_score": round(float(adjusted), 4),
            "oos_metrics": oos_metrics,
            "n_trials": len(study.trials),
            "metric": self.metric,
        }

        if save_best_to:
            save_best_to = Path(save_best_to)
            save_best_to.parent.mkdir(parents=True, exist_ok=True)
            with save_best_to.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 最佳參數已儲存: {save_best_to}")

        return result

    def optimize_walk_forward(
        self,
        window_train_years: int = 2,
        window_test_years: int = 1,
        step_years: int = 1,
        save_dir: Optional[Path] = None,
    ) -> Dict:
        """
        Walk-Forward Optimization。

        將整個期間分成多個 train/test window，逐段優化同驗證。

        Returns
        -------
        Dict with fold details + overall OOS metrics
        """
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        folds = []
        current = start

        while True:
            train_start = current
            train_end = train_start + pd.DateOffset(years=window_train_years)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(years=window_test_years)

            if test_end > end:
                test_end = end
            if test_start >= end:
                break

            folds.append({
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
            })

            current += pd.DateOffset(years=step_years)

        if not folds:
            print("⚠️ 無法建立 WFO folds")
            return {}

        print(f"🔄 WFO: {len(folds)} folds")

        fold_results = []
        all_oos_equity = []

        for i, fold in enumerate(folds):
            print(f"\n── Fold {i + 1}/{len(folds)} ──")
            print(f"   Train: {fold['train_start']} → {fold['train_end']}")
            print(f"   Test:  {fold['test_start']} → {fold['test_end']}")

            result = self.optimize_single_period(
                train_start=fold["train_start"],
                train_end=fold["train_end"],
                test_start=fold["test_start"],
                test_end=fold["test_end"],
            )

            fold_results.append({
                "fold": i + 1,
                **fold,
                **result,
            })

        # ── 彙總 OOS 結果 ──
        oos_sharpes = [
            fr["oos_metrics"].get("sharpe", 0)
            for fr in fold_results
            if fr.get("oos_metrics")
        ]
        oos_returns = [
            fr["oos_metrics"].get("total_return", 0)
            for fr in fold_results
            if fr.get("oos_metrics")
        ]
        oos_dds = [
            fr["oos_metrics"].get("max_drawdown", 0)
            for fr in fold_results
            if fr.get("oos_metrics")
        ]

        overall = {
            "n_folds": len(folds),
            "fold_results": fold_results,
            "overall_oos_metrics": {
                "sharpe": round(float(np.mean(oos_sharpes)), 4) if oos_sharpes else 0,
                "sharpe_std": round(float(np.std(oos_sharpes)), 4) if oos_sharpes else 0,
                "total_return": round(float(np.mean(oos_returns)), 4) if oos_returns else 0,
                "max_drawdown": round(float(np.min(oos_dds)), 4) if oos_dds else 0,
                "positive_folds": sum(1 for r in oos_returns if r > 0),
                "negative_folds": sum(1 for r in oos_returns if r <= 0),
            },
        }

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            with (save_dir / "walk_forward_result.json").open("w", encoding="utf-8") as f:
                json.dump(overall, f, indent=2, default=str)
            print(f"\n💾 WFO 結果已儲存: {save_dir / 'walk_forward_result.json'}")

        return overall

    # ─────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────

    def _sample_params(self, trial) -> Dict:
        """從 param_space 取樣一組參數"""
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, list):
                params[name] = trial.suggest_categorical(name, space)
            elif isinstance(space, tuple) and len(space) == 2:
                lo, hi = space
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = trial.suggest_int(name, lo, hi)
                elif isinstance(lo, float) or isinstance(hi, float):
                    params[name] = trial.suggest_float(name, float(lo), float(hi))
                else:
                    params[name] = trial.suggest_int(name, int(lo), int(hi))
            else:
                params[name] = space  # fixed value
        return params

    def _evaluate(self, params: Dict, start: str, end: str) -> float:
        """用一組參數做回測，回傳 metric 值"""
        try:
            strategy = self.strategy_class(**params)
        except Exception:
            return -999.0

        bt = UniversalBacktester(
            initial_capital=100_000.0,
            cost_model=self.cost_model,
            execution_delay=1,
        )

        try:
            equity_df = bt.run(
                strategy=strategy,
                data_dict=self.prices_dict,
                start_date=start,
                end_date=end,
            )
        except Exception:
            return -999.0

        if equity_df is None or equity_df.empty:
            return -999.0

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze(equity_df, trade_log=bt.trade_log)

        if self.metric == "robust_sharpe":
            sharpe = metrics.get("sharpe", 0)
            dd = abs(metrics.get("max_drawdown", -1))
            # Penalize deep drawdowns
            dd_penalty = max(0, dd - 0.15) * 2.0
            return sharpe - dd_penalty

        return metrics.get(self.metric, 0)

    def _full_backtest(self, params: Dict, start: str, end: str) -> Dict:
        """完整回測，回傳所有 metrics"""
        try:
            strategy = self.strategy_class(**params)
        except Exception:
            return {}

        bt = UniversalBacktester(
            initial_capital=100_000.0,
            cost_model=self.cost_model,
            execution_delay=1,
        )

        try:
            equity_df = bt.run(
                strategy=strategy,
                data_dict=self.prices_dict,
                start_date=start,
                end_date=end,
            )
        except Exception:
            return {}

        if equity_df is None or equity_df.empty:
            return {}

        analyzer = PerformanceAnalyzer()
        return analyzer.analyze(equity_df, trade_log=bt.trade_log)