"""
Módulo com métricas de risco para avaliação de portfólios.

Este módulo implementa métricas de risco para análise de portfólios,
incluindo volatilidade, value at risk, conditional value at risk,
e métricas de drawdown.

"""
import numpy as np
import pandas as pd
from scipy import stats

def calculate_volatility(weights, cov_matrix):

    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def calculate_tail_risk(weights, returns):

    portfolio_returns = np.dot(returns, weights)
    
    return {
        'skewness': stats.skew(portfolio_returns),
        'kurtosis': stats.kurtosis(portfolio_returns)
    }

def calculate_var(weights, returns, confidence_level=0.95):

    portfolio_returns = np.dot(returns, weights)
    return -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

def calculate_cvar(weights, returns, confidence_level=0.95):

    portfolio_returns = np.dot(returns, weights)
    var = calculate_var(weights, returns, confidence_level)
    
    # Evitar problemas quando não há retornos abaixo do VaR
    tail_returns = portfolio_returns[portfolio_returns <= -var]
    return -tail_returns.mean() if len(tail_returns) > 0 else var

def calculate_drawdown(weights, returns):

    portfolio_returns = np.dot(returns, weights)
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - rolling_max) / rolling_max
    
    return {
        'max_drawdown': drawdowns.min(),
        'avg_drawdown': drawdowns[drawdowns < 0].mean() if any(drawdowns < 0) else 0
    }

def calculate_diversification_ratio(weights, cov_matrix):

    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vols = np.sum(weights * asset_vols)
    portfolio_vol = calculate_volatility(weights, cov_matrix)
    
    return weighted_vols / portfolio_vol if portfolio_vol > 0 else 0