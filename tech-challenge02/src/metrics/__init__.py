"""
Pacote metrics para métricas de desempenho e risco.

Este pacote fornece um conjunto de métricas para análise de portfólios,
incluindo medidas de desempenho ajustadas ao risco e métricas de risco.

Módulos:
    performance: Métricas de desempenho e retorno ajustado ao risco
    risk: Métricas de risco e diversificação
"""

from src.metrics.performance import (
    calculate_metrics,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_beta,
    calculate_treynor_ratio
)

from src.metrics.risk import (
    calculate_volatility,
    calculate_var,
    calculate_cvar,
    calculate_drawdown,
    calculate_tail_risk,
    calculate_diversification_ratio
)

__all__ = [
    # Performance metrics
    'calculate_metrics',
    'calculate_sortino_ratio',
    'calculate_information_ratio',
    'calculate_calmar_ratio',
    'calculate_beta',
    'calculate_treynor_ratio',
    
    # Risk metrics
    'calculate_volatility',
    'calculate_var',
    'calculate_cvar',
    'calculate_drawdown',
    'calculate_tail_risk',
    'calculate_diversification_ratio'
]