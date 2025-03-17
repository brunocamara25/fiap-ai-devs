import numpy as np
import pandas as pd

def calculate_metrics(weights, returns, cov_matrix, risk_free_rate):
    """Calcular métricas do portfólio"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe_ratio

def calculate_diversification(cov_matrix):
    """Calcular a diversificação média do portfólio"""
    correlations = cov_matrix.corr()
    avg_correlation = correlations.mean().mean()
    return avg_correlation

def calculate_sortino_ratio(weights, returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calcula o Índice de Sortino para um portfólio.
    
    Parâmetros:
    - weights: array com os pesos do portfólio.
    - returns: DataFrame com os retornos históricos das ações.
    - risk_free_rate: taxa livre de risco (padrão: 0.0).
    - target_return: retorno alvo (padrão: 0.0).
    
    Retorna:
    - sortino_ratio: Índice de Sortino calculado.
    """
    # Retorno esperado do portfólio
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Anualizado
    
    # Retornos do portfólio
    portfolio_returns = np.dot(returns, weights)
    
    # Desvio padrão dos retornos abaixo do retorno alvo (volatilidade negativa)
    downside_deviation = np.sqrt(np.mean(np.minimum(0, portfolio_returns - target_return) ** 2))
    
    # Índice de Sortino
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    return sortino_ratio

def calculate_beta(weights, returns, market_returns):
    # Calcular os retornos do portfólio
    portfolio_returns = np.dot(returns, weights)
    
    # Converter para Series e alinhar índices
    portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
    market_returns = pd.Series(market_returns, index=returns.index).reindex(returns.index).fillna(method='ffill').fillna(method='bfill')
    
    # Remover valores ausentes
    aligned_data = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    portfolio_returns = aligned_data.iloc[:, 0]
    market_returns = aligned_data.iloc[:, 1]
    
    # Calcular a covariância e a variância do mercado
    covariance = np.cov(portfolio_returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    
    # Calcular o beta
    beta = covariance / market_variance
    return beta

def calculate_treynor_ratio(weights, returns, cov_matrix, risk_free_rate, market_returns):
    portfolio_return, _, _ = calculate_metrics(weights, returns, cov_matrix, risk_free_rate)
    beta = calculate_beta(weights, returns, market_returns)
    treynor_ratio = (portfolio_return - risk_free_rate) / beta
    return treynor_ratio

def calculate_var(weights, returns, confidence_level=0.95):
    portfolio_returns = np.dot(returns, weights)
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    return var