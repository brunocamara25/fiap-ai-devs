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
    """
    Função objetivo para maximizar o índice de Sharpe.
    
    .. math::
        SR = \\frac{R_p - R_f}{\\sigma_p}
    
    onde:
    
    - :math:`R_p` é o retorno esperado do portfólio
    - :math:`R_f` é a taxa livre de risco
    - :math:`\\sigma_p` é a volatilidade (desvio padrão) do portfólio
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
    cov_matrix : pd.DataFrame
        Matriz de covariância dos retornos.
    risk_free_rate : float
        Taxa livre de risco anualizada.
        
    Returns
    -------
    float
        Valor do índice de Sharpe (negativo para minimização).
    """
    portfolio_return, portfolio_volatility, sharpe = calculate_metrics(
        weights, returns, cov_matrix, risk_free_rate
    )
    # Retornamos o valor negativo porque os otimizadores geralmente minimizam
    return -sharpe


def sortino_ratio_objective(weights: np.ndarray,
                           returns: pd.DataFrame,
                           risk_free_rate: float,
                           target_return: float = 0.0) -> float:
    """
    Função objetivo para maximizar o índice de Sortino.
    
    .. math::
        Sortino = \\frac{R_p - R_f}{\\sigma_d}
    
    onde:
    
    - :math:`R_p` é o retorno esperado do portfólio
    - :math:`R_f` é a taxa livre de risco
    - :math:`\\sigma_d` é o desvio padrão dos retornos negativos (abaixo do alvo)
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
    risk_free_rate : float
        Taxa livre de risco anualizada.
    target_return : float, optional
        Retorno alvo mínimo aceitável (geralmente 0).
        
    Returns
    -------
    float
        Valor negativo do índice de Sortino (para minimização).
    """
    sortino = calculate_sortino_ratio(weights, returns, risk_free_rate, target_return)
    return -sortino


def treynor_ratio_objective(weights: np.ndarray,
                           returns: pd.DataFrame,
                           market_returns: pd.Series,
                           risk_free_rate: float = 0.0) -> float:
    """
    Função objetivo para maximizar o índice de Treynor.
    
    .. math::
        Treynor = \\frac{R_p - R_f}{\\beta_p}
    
    onde:
    
    - :math:`R_p` é o retorno esperado do portfólio
    - :math:`R_f` é a taxa livre de risco
    - :math:`\\beta_p` é o beta do portfólio em relação ao mercado
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
    market_returns : pd.Series
        Série com retornos do mercado.
    risk_free_rate : float, optional
        Taxa livre de risco anualizada.
        
    Returns
    -------
    float
        Valor negativo do índice de Treynor (para minimização).
    """
    treynor = calculate_treynor_ratio(weights, returns, market_returns, risk_free_rate)
    return -treynor


def volatility_objective(weights: np.ndarray,
                        cov_matrix: pd.DataFrame) -> float:
    """
    Função objetivo para minimizar a volatilidade do portfólio.
    
    .. math::
        \\sigma_p = \\sqrt{w^T \\Sigma w \\times 252}
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    cov_matrix : pd.DataFrame
        Matriz de covariância dos retornos.
        
    Returns
    -------
    float
        Volatilidade anualizada do portfólio.
    """
    return calculate_volatility(weights, cov_matrix)


def var_objective(weights: np.ndarray,
                 returns: pd.DataFrame,
                 confidence_level: float = 0.95) -> float:
    """
    Função objetivo para minimizar o Value at Risk (VaR) do portfólio.
    
    .. math::
        VaR_{\\alpha} = -\\inf\\{l \\in \\mathbb{R}: P(L \\leq l) \\geq \\alpha\\}
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
    confidence_level : float, optional
        Nível de confiança para o cálculo do VaR (ex: 0.95 para 95%).
        
    Returns
    -------
    float
        Valor positivo do VaR (para minimização).
    """
    var = calculate_var(weights, returns, confidence_level)
    return var


def cvar_objective(weights: np.ndarray,
                  returns: pd.DataFrame,
                  confidence_level: float = 0.95) -> float:
    """
    Função objetivo para minimizar o Conditional Value at Risk (CVaR) do portfólio.
    
    .. math::
        CVaR_{\\alpha} = -\\mathbb{E}[L|L \\leq -VaR_{\\alpha}]
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
    confidence_level : float, optional
        Nível de confiança para o cálculo do CVaR (ex: 0.95 para 95%).
        
    Returns
    -------
    float
        Valor positivo do CVaR (para minimização).
    """
    cvar = calculate_cvar(weights, returns, confidence_level)
    return cvar


def return_objective(weights: np.ndarray,
                    returns: pd.DataFrame) -> float:
    """
    Função objetivo para maximizar o retorno esperado do portfólio.
    
    .. math::
        R_p = \\sum_i w_i \\mu_i \\times 252
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
        
    Returns
    -------
    float
        Valor negativo do retorno esperado anualizado (para minimização).
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    return -portfolio_return


def pareto_front_objective(weights: np.ndarray,
                          returns: pd.DataFrame,
                          cov_matrix: pd.DataFrame) -> Tuple[float, float]:
    """
    Função objetivo para otimização multi-objetivo visando o Pareto Front.
    
    Esta função retorna retorno e risco como objetivos separados.
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    returns : pd.DataFrame
        DataFrame com os retornos históricos dos ativos.
    cov_matrix : pd.DataFrame
        Matriz de covariância dos retornos.
        
    Returns
    -------
    Tuple[float, float]
        (retorno, risco) para otimização multi-objetivo.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

    # Para otimizadores que minimizam, inverte-se o retorno
    return -portfolio_return, portfolio_risk


def get_objective_function(objective_name: str) -> Callable:
    """
    Retorna a função objetivo correspondente ao nome fornecido.
    
    Parameters
    ----------
    objective_name : str
        Nome da função objetivo.
        
    Returns
    -------
    Callable
        Função objetivo correspondente.
        
    Raises
    ------
    ValueError
        Se o nome da função objetivo não for reconhecido.
    """
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
