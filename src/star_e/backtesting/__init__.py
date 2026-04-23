"""Backtesting module for strategy evaluation."""

from star_e.backtesting.engine import BacktestEngine
from star_e.backtesting.walk_forward import WalkForwardValidator

__all__ = [
    "BacktestEngine",
    "WalkForwardValidator",
]
