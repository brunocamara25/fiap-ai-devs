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
    """
    Calcula a volatilidade anualizada de um portfólio.
    
    A volatilidade é definida como:
    \[ \sigma_p = \sqrt{w^T \Sigma w \times 252} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        
    Retorna:
        float: Volatilidade anualizada do portfólio.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def calculate_tail_risk(weights, returns):
    """
    Calcula métricas de risco de cauda do portfólio.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        
    Retorna:
        dict: Dicionário com métricas de risco de cauda (skewness, kurtosis).
    """
    portfolio_returns = np.dot(returns, weights)

    return {
        'skewness': stats.skew(portfolio_returns),
        'kurtosis': stats.kurtosis(portfolio_returns)
    }

def calculate_var(weights, returns, confidence_level=0.95):
    """
    Calcula o Value at Risk (VaR) histórico do portfólio.
    
    O VaR é definido como:
    \[ VaR_{\alpha} = -\inf\{l \in \mathbb{R}: P(L \leq l) \geq \alpha\} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        confidence_level (float): Nível de confiança, entre 0 e 1 (padrão: 0.95).
        
    Retorna:
        float: VaR no nível de confiança especificado.
    """
    portfolio_returns = np.dot(returns, weights)
    return -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

def calculate_cvar(weights, returns, confidence_level=0.95):
    """
    Calcula o Conditional Value at Risk (CVaR)/Expected Shortfall do portfólio.
    
    O CVaR é definido como:
    \[ CVaR_{\alpha} = -\mathbb{E}[L|L \leq -VaR_{\alpha}] \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        confidence_level (float): Nível de confiança (padrão: 0.95).
        
    Retorna:
        float: CVaR no nível de confiança especificado.
    """
    portfolio_returns = np.dot(returns, weights)
    var = calculate_var(weights, returns, confidence_level)

    # Evitar problemas quando não há retornos abaixo do VaR
    tail_returns = portfolio_returns[portfolio_returns <= -var]
    return -tail_returns.mean() if len(tail_returns) > 0 else var

def calculate_drawdown(weights, returns):
    """
    Calcula o máximo drawdown e drawdown médio do portfólio.
    
    O drawdown em um tempo t é definido como:
    \[ DD_t = \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        
    Retorna:
        dict: Dicionário com 'max_drawdown' e 'avg_drawdown'.
    """
    portfolio_returns = np.dot(returns, weights)
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - rolling_max) / rolling_max

    return {
        'max_drawdown': drawdowns.min(),
        'avg_drawdown': drawdowns[drawdowns < 0].mean() if any(drawdowns < 0) else 0
    }

def calculate_diversification_ratio(weights, cov_matrix):
    """
    Calcula o Maximum Diversification Ratio do portfólio.
    
    O DR é definido como:
    \[ DR = \frac{\sum_{i=1}^n w_i \sigma_i}{\sigma_p} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        
    Retorna:
        float: Ratio de diversificação do portfólio.
    """
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vols = np.sum(weights * asset_vols)
    portfolio_vol = calculate_volatility(weights, cov_matrix)

    return weighted_vols / portfolio_vol if portfolio_vol > 0 else 0
