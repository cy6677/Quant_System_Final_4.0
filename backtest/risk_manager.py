"""
backtest/risk_manager.py
========================
機構級風控模組 — 三級斷路器 + 倉位限制

設計原則：
- 斷路器係最後防線，唔係日常風控
- 觸發後有冷靜期，唔會即日反覆觸發
- 可以被策略主動 bypass (但會 log 警告)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import pandas as pd
import numpy as np


@dataclass
class CircuitBreakerState:
    """斷路器當前狀態"""
    level: int = 0           # 0=正常, 1/2/3
    triggered_date: Optional[pd.Timestamp] = None
    cooldown_until: Optional[pd.Timestamp] = None
    exposure_cap: float = 1.0   # 1.0 = 100%


class RiskManager:
    """
    三級斷路器 + 即時風控

    Level 1: 日虧損 > daily_loss_pct → 減倉 reduce_l1
    Level 2: 回撤 > drawdown_l2 → 減倉 reduce_l2
    Level 3: 回撤 > drawdown_l3 → 全平 + cooldown_days 冷靜期

    Parameters
    ----------
    daily_loss_pct : float
        Level 1 觸發條件 (default: 0.02 = 2%)
    drawdown_l2 : float
        Level 2 觸發條件 (default: 0.12 = 12%)
    drawdown_l3 : float
        Level 3 觸發條件 (default: 0.18 = 18%)
    reduce_l1 : float
        Level 1 減倉比例 (default: 0.30)
    reduce_l2 : float
        Level 2 目標曝露 (default: 0.50)
    cooldown_days : int
        Level 3 冷靜期 (default: 30)
    max_single_position_pct : float
        單隻股票最大佔比 (default: 0.08)
    max_total_exposure : float
        最大總曝露 (default: 1.0)
    """

    def __init__(
        self,
        daily_loss_pct: float = 0.02,
        drawdown_l2: float = 0.12,
        drawdown_l3: float = 0.18,
        reduce_l1: float = 0.30,
        reduce_l2: float = 0.50,
        cooldown_days: int = 30,
        max_single_position_pct: float = 0.08,
        max_total_exposure: float = 1.0,
    ):
        self.daily_loss_pct = daily_loss_pct
        self.drawdown_l2 = drawdown_l2
        self.drawdown_l3 = drawdown_l3
        self.reduce_l1 = reduce_l1
        self.reduce_l2 = reduce_l2
        self.cooldown_days = cooldown_days
        self.max_single_position_pct = max_single_position_pct
        self.max_total_exposure = max_total_exposure

        self._peak_equity: float = 0.0
        self._prev_equity: float = 0.0
        self._state = CircuitBreakerState()
        self._log: List[Dict] = []

    def reset(self):
        self._peak_equity = 0.0
        self._prev_equity = 0.0
        self._state = CircuitBreakerState()
        self._log = []

    def update(
        self,
        date: pd.Timestamp,
        current_equity: float,
    ) -> CircuitBreakerState:
        """
        每日更新風控狀態。回傳當前 exposure cap。

        Parameters
        ----------
        date : pd.Timestamp
        current_equity : float

        Returns
        -------
        CircuitBreakerState
        """
        # 初始化
        if self._peak_equity == 0:
            self._peak_equity = current_equity
            self._prev_equity = current_equity
            return self._state

        # 更新 peak
        self._peak_equity = max(self._peak_equity, current_equity)

        # 冷靜期檢查
        if (
            self._state.cooldown_until is not None
            and date < self._state.cooldown_until
        ):
            self._state.exposure_cap = 0.0  # 完全唔做嘢
            return self._state

        # 重設冷靜期
        if (
            self._state.cooldown_until is not None
            and date >= self._state.cooldown_until
        ):
            self._state = CircuitBreakerState()  # 重啟
            self._peak_equity = current_equity     # 重設 peak
            self._log.append({
                "date": date, "event": "cooldown_end", "level": 0,
            })

        # 計算 drawdown
        drawdown = (current_equity - self._peak_equity) / self._peak_equity
        daily_return = (
            (current_equity - self._prev_equity) / self._prev_equity
            if self._prev_equity > 0
            else 0.0
        )

        self._prev_equity = current_equity

        # ── Level 3: Hard Stop ──
        if abs(drawdown) > self.drawdown_l3:
            self._state.level = 3
            self._state.triggered_date = date
            self._state.cooldown_until = date + pd.Timedelta(
                days=self.cooldown_days
            )
            self._state.exposure_cap = 0.0
            self._log.append({
                "date": date,
                "event": "LEVEL3_HARD_STOP",
                "drawdown": drawdown,
                "equity": current_equity,
            })
            return self._state

        # ── Level 2: Soft Stop ──
        if abs(drawdown) > self.drawdown_l2:
            self._state.level = 2
            self._state.triggered_date = date
            self._state.exposure_cap = self.reduce_l2
            return self._state

        # ── Level 1: Daily Loss ──
        if daily_return < -self.daily_loss_pct:
            self._state.level = 1
            self._state.triggered_date = date
            self._state.exposure_cap = 1.0 - self.reduce_l1
            return self._state

        # ── Recovery ──
        if self._state.level > 0 and abs(drawdown) < self.drawdown_l2 * 0.5:
            self._state = CircuitBreakerState()

        return self._state

    def check_position_limit(
        self, ticker: str, position_value: float, total_equity: float
    ) -> bool:
        """檢查單隻股票佔比是否超標"""
        if total_equity <= 0:
            return False
        pct = position_value / total_equity
        return pct <= self.max_single_position_pct

    @property
    def log(self) -> List[Dict]:
        return self._log.copy()