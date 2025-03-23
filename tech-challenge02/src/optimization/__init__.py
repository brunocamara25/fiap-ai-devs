"""
Pacote optimization para otimização de portfólio.

Este pacote contém módulos para representação e análise de portfólios,
definição de restrições e funções objetivo para otimização.
"""
from src.optimization.portfolio import Portfolio
from src.optimization.constraints import (
    weights_sum_to_one,
    enforce_weights_sum_to_one,
    weights_within_bounds,
    enforce_weights_within_bounds,
    apply_all_constraints
)
from src.optimization.objective import (
    sharpe_ratio_objective,
    sortino_ratio_objective,
    treynor_ratio_objective,
    volatility_objective,
    var_objective,
    cvar_objective,
    return_objective,
    pareto_front_objective,
    get_objective_function
)

__all__ = [
    # Portfolio
    'Portfolio',

    # Constraints
    'weights_sum_to_one',
    'enforce_weights_sum_to_one',
    'weights_within_bounds',
    'enforce_weights_within_bounds',
    'apply_all_constraints',

    # Objective
    'sharpe_ratio_objective',
    'sortino_ratio_objective',
    'treynor_ratio_objective',
    'volatility_objective',
    'var_objective',
    'cvar_objective',
    'return_objective',
    'pareto_front_objective',
    'get_objective_function'
]
