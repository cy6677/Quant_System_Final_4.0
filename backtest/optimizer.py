from __future__ import annotations

import json
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.universal_backtester import (
    UniversalBacktester,
    PerformanceAnalyzer,
    TransactionCostModel,
)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Optuna 未安裝。將自動降級為 Random Search。建議執行: pip install optuna"
    )
    _OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrialResult:
    params: Dict[str, Any]
    metric: float
    metrics: Optional[Dict[str, float]] = None
    equity_df: Optional[pd.DataFrame] = None


class StrategyOptimizer:
    """
    機構級策略參數優化器 v5 (hardened)

    升級
    ----
    1. tuple=list 規則維持清晰
    2. robust_sharpe 加入 turnover penalty
    3. WFO / single period 加入 IS-OOS consistency penalty
    4. 每個 trial 記錄完整 metrics
    5. optuna 加 timeout 防卡死
    """

    def __init__(
        self,
        strategy_class: type,
        param_space: Dict[str, Union[List[Any], Tuple[float, float], Any]],
        prices_dict: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        fixed_params: Optional[Dict[str, Any]] = None,
        metric: str = "robust_sharpe",
        n_trials: int = 50,
        initial_capital: float = 100_000.0,
        cost_model: Optional[TransactionCostModel] = None,
        allow_fractional: bool = True,
        calendar_ticker: str = "SPY",
        optuna_timeout_sec: int = 1200,
    ):
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.prices_dict = prices_dict
        self.start_date = start_date
        self.end_date = end_date
        self.fixed_params = fixed_params or {}
        self.metric = metric.lower()
        self.n_trials = n_trials
        self.initial_capital = initial_capital
        self.cost_model = cost_model or TransactionCostModel()
        self.allow_fractional = allow_fractional
        self.calendar_ticker = calendar_ticker
        self.optuna_timeout_sec = optuna_timeout_sec

        self.trials: List[TrialResult] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_metric: float = -np.inf

    def _suggest_params_optuna(self, trial) -> Dict[str, Any]:
        params = {}
        for key, value in self.param_space.items():
            if isinstance(value, tuple) and len(value) == 2:
                if isinstance(value[0], float) or isinstance(value[1], float):
                    params[key] = trial.suggest_float(
                        key, float(value[0]), float(value[1])
                    )
                else:
                    params[key] = trial.suggest_int(
                        key, int(value[0]), int(value[1])
                    )
            elif isinstance(value, list):
                params[key] = trial.suggest_categorical(key, value)
            else:
                params[key] = value
        params.update(self.fixed_params)
        return params

    def _sample_params_random(self) -> Dict[str, Any]:
        params = {}
        for key, spec in self.param_space.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                low, high = spec
                if isinstance(low, int) and isinstance(high, int):
                    params[key] = random.randint(low, high)
                else:
                    params[key] = round(random.uniform(float(low), float(high)), 4)
            elif isinstance(spec, list):
                params[key] = random.choice(spec)
            else:
                params[key] = spec
        params.update(self.fixed_params)
        return params

    def _run_single_backtest(
        self, params: Dict[str, Any], start_date: str, end_date: str
    ) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        try:
            strategy = self.strategy_class(**params)
            if hasattr(strategy, "reset"):
                strategy.reset()

            backtester = UniversalBacktester(
                initial_capital=self.initial_capital,
                cost_model=self.cost_model,
                allow_fractional=self.allow_fractional,
                calendar_ticker=self.calendar_ticker,
            )

            warmup_days = int(getattr(strategy, "history_window", 400)) + 50
            warmup_start = (
                pd.to_datetime(start_date) - pd.Timedelta(days=warmup_days)
            ).strftime("%Y-%m-%d")

            sliced_prices = {}
            for ticker, df in self.prices_dict.items():
                sliced_df = df.loc[warmup_start:end_date]
                if not sliced_df.empty and len(sliced_df) > 10:
                    sliced_prices[ticker] = sliced_df

            if not sliced_prices:
                return {}, None

            equity_df = backtester.run(
                strategy=strategy,
                data_dict=sliced_prices,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.warning(f"[Backtest Failed] params={params} | error={e}")
            return {}, None

        if equity_df is None or equity_df.empty:
            return {}, None

        trade_log_df = backtester.trade_log

        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze(
            equity_df,
            trade_log=trade_log_df if not trade_log_df.empty else None,
        )

        if not trade_log_df.empty and "qty" in trade_log_df.columns:
            metrics["total_trades"] = int((trade_log_df["qty"] < 0).sum())
            metrics["total_fills"] = len(trade_log_df)
        else:
            metrics["total_trades"] = 0
            metrics["total_fills"] = 0

        # turnover proxy (fills / trading_days)
        trading_days = max(len(equity_df), 1)
        metrics["turnover_proxy"] = metrics.get("total_fills", 0) / trading_days

        return metrics, equity_df

    def _extract_metric(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return -np.inf

        if self.metric == "robust_sharpe":
            base_score = metrics.get("sharpe", 0.0)

            max_dd = abs(metrics.get("max_drawdown", 0.0))
            dd_penalty = max(0.0, (max_dd - 0.15) * 8.0)

            num_fills = metrics.get("total_fills", 0)
            if num_fills < 30:
                trade_penalty = 3.0
            elif num_fills < 80:
                trade_penalty = 1.0
            elif num_fills > 1000:
                trade_penalty = (num_fills - 1000) * 0.002
            else:
                trade_penalty = 0.0

            sortino = metrics.get("sortino", 0.0)
            sortino_bonus = max(0.0, sortino * 0.2)

            pf = metrics.get("profit_factor", 0.0)
            pf_penalty = max(0.0, (1.3 - pf) * 2.0) if pf > 0 else 2.0

            calmar = metrics.get("calmar", 0.0)
            calmar_bonus = max(0.0, calmar * 0.1)

            # 新增 turnover penalty
            turnover = metrics.get("turnover_proxy", 0.0)
            turnover_penalty = max(0.0, (turnover - 0.6) * 1.5)

            final = (
                base_score
                - dd_penalty
                - trade_penalty
                - pf_penalty
                - turnover_penalty
                + sortino_bonus
                + calmar_bonus
            )
            return float(final)

        val = metrics.get(self.metric, -np.inf)
        if self.metric == "max_drawdown":
            return -abs(float(val))
        return float(val)

    @staticmethod
    def _apply_consistency_penalty(
        is_score: float,
        oos_metrics: Dict[str, float],
        metric_name: str,
        penalty_scale: float = 0.7,
    ) -> float:
        if not oos_metrics:
            return is_score - 1.0

        metric_name = metric_name.lower()
        if metric_name == "robust_sharpe":
            oos_base = oos_metrics.get("sharpe", 0.0)
        else:
            oos_base = oos_metrics.get(metric_name, 0.0)

        gap = abs(is_score - float(oos_base))
        return float(is_score - penalty_scale * gap)

    def _optimize(
        self, train_start: str, train_end: str, n_trials: int
    ) -> Tuple[Optional[Dict[str, Any]], float]:

        if _OPTUNA_AVAILABLE:

            def objective(trial):
                params = self._suggest_params_optuna(trial)
                metrics, _ = self._run_single_backtest(
                    params, train_start, train_end
                )
                score = self._extract_metric(metrics)

                self.trials.append(
                    TrialResult(
                        params=params,
                        metric=score,
                        metrics=metrics if metrics else None,
                        equity_df=None,
                    )
                )
                return score

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=10, n_warmup_steps=5
                ),
            )
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True,
                n_jobs=1,
                timeout=self.optuna_timeout_sec,
            )

            if study.best_trial is None:
                return None, -np.inf

            best_params = dict(study.best_trial.params)
            best_params.update(self.fixed_params)

            for key, spec in self.param_space.items():
                if not isinstance(spec, (list, tuple)):
                    best_params[key] = spec

            return best_params, float(study.best_trial.value)

        else:
            best_metric = -np.inf
            best_params = None
            for i in range(n_trials):
                params = self._sample_params_random()
                metrics, equity_df = self._run_single_backtest(
                    params, train_start, train_end
                )
                value = self._extract_metric(metrics)
                self.trials.append(
                    TrialResult(
                        params=params,
                        metric=value,
                        metrics=metrics if metrics else None,
                        equity_df=equity_df,
                    )
                )
                if value > best_metric:
                    best_metric = value
                    best_params = params
                if (i + 1) % 10 == 0:
                    print(
                        f"  Random Search: {i + 1}/{n_trials} done, "
                        f"best={best_metric:.4f}"
                    )
            return best_params, best_metric

    def optimize_single_period(
        self,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        save_best_to: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        print(f"\n{'=' * 60}")
        print(f"📊 訓練期: {train_start} → {train_end}")
        print(f"🧪 測試期: {test_start} → {test_end}")
        print(f"🎯 優化指標: {self.metric}")
        print(f"🔢 試驗次數: {self.n_trials}")
        print(f"{'=' * 60}\n")

        best_params, best_metric_is = self._optimize(
            train_start, train_end, self.n_trials
        )

        oos_metrics = {}
        final_score = best_metric_is

        if best_params:
            print(f"\n✅ IS 最佳分數: {best_metric_is:.4f}")
            print(f"📋 最佳參數: {json.dumps(best_params, indent=2, default=str)}")

            oos_metrics, _ = self._run_single_backtest(
                best_params, test_start, test_end
            )
            final_score = self._apply_consistency_penalty(
                is_score=best_metric_is,
                oos_metrics=oos_metrics,
                metric_name=self.metric,
                penalty_scale=0.7,
            )

            if oos_metrics:
                print(f"\n🧪 OOS 結果:")
                for k, v in oos_metrics.items():
                    if isinstance(v, (int, float)):
                        print(f"  {k:20s}: {v:.4f}")
                print(f"⚖️ Consistency-adjusted score: {final_score:.4f}")

        result = {
            "mode": "single_period",
            "metric": self.metric,
            "optimizer": "optuna" if _OPTUNA_AVAILABLE else "random",
            "best_params": best_params,
            "best_metric_insample": float(best_metric_is),
            "consistency_adjusted_score": float(final_score),
            "oos_metrics": oos_metrics,
        }

        if save_best_to is not None:
            path = Path(save_best_to)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 結果已儲存至 {path}")

        return result

    def optimize_walk_forward(
        self,
        window_train_years: int = 3,
        window_test_years: int = 1,
        step_years: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        step_years = step_years or window_test_years

        calendar = self.prices_dict.get(self.calendar_ticker)
        if calendar is None or calendar.empty:
            raise ValueError(f"搵唔到日曆股票 {self.calendar_ticker} 嘅數據！")

        all_dates = calendar.index
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        dates_in_range = all_dates[(all_dates >= start) & (all_dates <= end)]

        if len(dates_in_range) == 0:
            raise ValueError("該時間範圍內沒有交易日！")

        segments = []
        all_test_equity = []
        cur_train_start = dates_in_range[0]

        fold = 0
        while True:
            cur_train_end = cur_train_start + pd.DateOffset(
                years=window_train_years
            )
            cur_test_start = cur_train_end + pd.Timedelta(days=1)
            cur_test_end = (
                cur_test_start
                + pd.DateOffset(years=window_test_years)
                - pd.Timedelta(days=1)
            )

            if cur_train_end >= end:
                break

            train_s = cur_train_start.strftime("%Y-%m-%d")
            train_e = min(cur_train_end, end).strftime("%Y-%m-%d")
            test_s = cur_test_start.strftime("%Y-%m-%d")
            test_e = min(cur_test_end, end).strftime("%Y-%m-%d")

            if pd.to_datetime(test_s) > pd.to_datetime(test_e):
                break

            fold += 1
            print(f"\n{'=' * 60}")
            print(f"🔄 WFO Fold {fold}")
            print(f"   訓練: {train_s} → {train_e}")
            print(f"   測試: {test_s} → {test_e}")
            print(f"{'=' * 60}")

            best_params_seg, best_metric_seg = self._optimize(
                train_s, train_e, self.n_trials
            )

            if best_params_seg is None:
                print("   ⚠️ 呢個 fold 搵唔到有效參數，跳過")
                cur_train_start += pd.DateOffset(years=step_years)
                continue

            test_metrics, test_equity = self._run_single_backtest(
                best_params_seg, test_s, test_e
            )

            adjusted_score = self._apply_consistency_penalty(
                is_score=best_metric_seg,
                oos_metrics=test_metrics,
                metric_name=self.metric,
                penalty_scale=0.7,
            )

            print(f"   IS Score: {best_metric_seg:.4f}")
            print(f"   Adj Score: {adjusted_score:.4f}")
            if test_metrics:
                print(f"   OOS Sharpe: {test_metrics.get('sharpe', 'N/A')}")
                print(f"   OOS Return: {test_metrics.get('total_return', 'N/A')}")

            segments.append(
                {
                    "fold": fold,
                    "train_start": train_s,
                    "train_end": train_e,
                    "test_start": test_s,
                    "test_end": test_e,
                    "best_metric_in_sample": float(best_metric_seg),
                    "consistency_adjusted_score": float(adjusted_score),
                    "best_params": best_params_seg,
                    "test_metrics": test_metrics,
                }
            )

            if test_equity is not None and not test_equity.empty:
                returns_series = test_equity["equity"].pct_change().iloc[1:]
                all_test_equity.append(returns_series)

            cur_train_start += pd.DateOffset(years=step_years)

        result: Dict[str, Any] = {
            "mode": "walk_forward",
            "metric": self.metric,
            "optimizer": "optuna" if _OPTUNA_AVAILABLE else "random",
            "n_folds": len(segments),
            "segments": segments,
        }

        if all_test_equity:
            combined_returns = pd.concat(all_test_equity)
            combined_returns = combined_returns[
                ~combined_returns.index.duplicated(keep="first")
            ]
            combined_returns = combined_returns.sort_index()

            running_capital = self.initial_capital
            equity_values = []
            for ret in combined_returns:
                running_capital *= 1 + ret
                equity_values.append(running_capital)

            combined_df = pd.DataFrame(
                {"equity": equity_values}, index=combined_returns.index
            )

            analyzer = PerformanceAnalyzer()
            overall_metrics = analyzer.analyze(combined_df)
            result["overall_oos_metrics"] = overall_metrics
            result["final_capital"] = running_capital
            result["wf_return"] = (
                running_capital - self.initial_capital
            ) / self.initial_capital

            print(f"\n{'=' * 60}")
            print(f"📊 Walk-Forward 整體 OOS 結果:")
            for k, v in overall_metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k:20s}: {v:.4f}")
            print(f"  {'final_capital':20s}: ${running_capital:,.2f}")
            print(f"{'=' * 60}")

        if save_dir is not None:
            path = Path(save_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "walk_forward_result.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 WFO 結果已儲存至 {path}")

        return result


strategy_optimizer = StrategyOptimizer
trial_result = TrialResult