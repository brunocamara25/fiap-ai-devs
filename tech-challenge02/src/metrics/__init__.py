"""
Pacote metrics para métricas de desempenho e risco.
"""
from src.metrics.performance import (
    calculate_metrics,
    calculate_sortino_ratio,
    calculate_treynor_ratio,
    calculate_beta
)

from src.metrics.risk import (
    calculate_volatility,
    calculate_var,
    calculate_cvar,
    calculate_drawdown,
    calculate_diversification
)
