"""
Módulo com funções objetivo para otimização de portfólio.

Este módulo implementa funções objetivo para otimização de portfólio de investimentos,
incluindo maximização de retorno ajustado ao risco, minimização de risco,
e objetivos multi-critério.

"""
from typing import Callable, Dict, Tuple, Union
import numpy as np
import pandas as pd

from src.metrics.performance import (
    calculate_metrics, calculate_sortino_ratio, calculate_treynor_ratio
)
from src.metrics.risk import (
    calculate_var, calculate_cvar, calculate_volatility
)


def sharpe_ratio_objective(weights: np.ndarray, 
                          returns: pd.DataFrame, 
                          cov_matrix: pd.DataFrame, 
                          risk_free_rate: float) -> float:

    portfolio_return, portfolio_volatility, sharpe = calculate_metrics(
        weights, returns, cov_matrix, risk_free_rate
    )
    # Retornamos o valor negativo porque os otimizadores geralmente minimizam
    return -sharpe


def sortino_ratio_objective(weights: np.ndarray, 
                           returns: pd.DataFrame, 
                           risk_free_rate: float,
                           target_return: float = 0.0) -> float:

    sortino = calculate_sortino_ratio(weights, returns, risk_free_rate, target_return)
    return -sortino


def treynor_ratio_objective(weights: np.ndarray, 
                           returns: pd.DataFrame, 
                           market_returns: pd.Series,
                           risk_free_rate: float = 0.0) -> float:

    treynor = calculate_treynor_ratio(weights, returns, market_returns, risk_free_rate)
    return -treynor


def volatility_objective(weights: np.ndarray, 
                        cov_matrix: pd.DataFrame) -> float:

    return calculate_volatility(weights, cov_matrix)


def var_objective(weights: np.ndarray, 
                 returns: pd.DataFrame, 
                 confidence_level: float = 0.95) -> float:

    var = calculate_var(weights, returns, confidence_level)
    return var


def cvar_objective(weights: np.ndarray, 
                  returns: pd.DataFrame, 
                  confidence_level: float = 0.95) -> float:

    cvar = calculate_cvar(weights, returns, confidence_level)
    return cvar


def return_objective(weights: np.ndarray, 
                    returns: pd.DataFrame) -> float:

    portfolio_return = np.sum(returns.mean() * weights) * 252
    return -portfolio_return


def pareto_front_objective(weights: np.ndarray, 
                          returns: pd.DataFrame, 
                          cov_matrix: pd.DataFrame) -> Tuple[float, float]:

    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    
    # Para otimizadores que minimizam, inverte-se o retorno
    return -portfolio_return, portfolio_risk


def get_objective_function(objective_name: str) -> Callable:

    objectives = {
        'sharpe': sharpe_ratio_objective,
        'sortino': sortino_ratio_objective,
        'treynor': treynor_ratio_objective,
        'volatility': volatility_objective,
        'var': var_objective,
        'cvar': cvar_objective,
        'return': return_objective,
        'pareto': pareto_front_objective
    }
    
    if objective_name not in objectives:
        raise ValueError(f"Função objetivo '{objective_name}' não reconhecida. "
                        f"Opções válidas: {list(objectives.keys())}")
    
    return objectives[objective_name]