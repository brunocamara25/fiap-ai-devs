"""
Módulo com métricas de risco para avaliação de portfólios.
"""
import numpy as np
import pandas as pd

def calculate_volatility(weights, cov_matrix):
    """
    Calcula a volatilidade anualizada de um portfólio.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        
    Retorna:
        float: Volatilidade anualizada do portfólio.
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

def calculate_var(weights, returns, confidence_level=0.95):
    """
    Calcula o Value at Risk (VaR) histórico do portfólio.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        confidence_level (float): Nível de confiança, entre 0 e 1 (padrão: 0.95).
        
    Retorna:
        float: Valor percentil correspondente ao VaR.
    """
    # Calcular os retornos do portfólio
    portfolio_returns = np.dot(returns, weights)
    
    # Calcular o VaR como o percentil
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    return var

def calculate_cvar(weights, returns, confidence_level=0.95):
    """
    Calcula o Conditional Value at Risk (CVaR)/Expected Shortfall do portfólio.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        confidence_level (float): Nível de confiança, entre 0 e 1 (padrão: 0.95).
        
    Retorna:
        float: CVaR do portfólio.
    """
    # Calcular os retornos do portfólio
    portfolio_returns = np.dot(returns, weights)
    
    # Calcular o VaR
    var = calculate_var(weights, returns, confidence_level)
    
    # Calcular o CVaR
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    return cvar

def calculate_drawdown(weights, returns):
    """
    Calcula o máximo drawdown do portfólio.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        
    Retorna:
        tuple: (max_drawdown, drawdown_duration) - Máximo drawdown e sua duração.
    """
    # Calcular os retornos do portfólio
    portfolio_returns = np.dot(returns, weights)
    
    # Calcular o valor acumulado do portfólio
    wealth_index = (1 + portfolio_returns).cumprod()
    
    # Calcular o valor máximo atual
    previous_peaks = wealth_index.cummax()
    
    # Calcular drawdowns
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    # Máximo drawdown
    max_drawdown = drawdowns.min()
    
    # Duração do drawdown
    is_drawdown = drawdowns < 0
    if not any(is_drawdown):
        return 0, 0
    
    # Contar períodos consecutivos de drawdown
    count = 0
    max_count = 0
    for d in is_drawdown:
        if d:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    
    return max_drawdown, max_count

def calculate_diversification(cov_matrix):
    """
    Calcula a diversificação média do portfólio.
    
    Parâmetros:
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        
    Retorna:
        float: Correlação média do portfólio.
    """
    correlations = cov_matrix.corr()
    avg_correlation = correlations.mean().mean()
    return avg_correlation 