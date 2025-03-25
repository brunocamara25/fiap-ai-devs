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
    
    A volatilidade mede a dispersão dos retornos (risco total):
    Fórmula: Volatilidade = raiz(pesos × matriz covariância × pesos × 252)
    
    O valor 252 é usado para anualizar (número aproximado de dias de negociação em um ano).
    
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
    
    O VaR responde à pergunta: "Qual é a perda máxima esperada com X% de confiança?"
    
    Exemplo: VaR de 2% com 95% de confiança significa que, em 95% do tempo,
    a perda diária não deve exceder 2% do valor do portfólio.
    
    Fórmula simplificada: VaR = percentil(retornos do portfólio, 100 - nível de confiança × 100)
    
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
    
    O CVaR mede a perda média esperada nos piores cenários, além do VaR.
    
    Exemplo: Se o VaR(95%) for 2%, o CVaR(95%) seria a perda média esperada
    nos 5% piores casos (quando as perdas excedem o VaR).
    
    Fórmula simplificada: CVaR = média dos retornos que são piores que o VaR
    
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
    
    O drawdown mede a queda do valor do portfólio desde um pico até um vale:
    
    Fórmula: Drawdown = (Valor atual - Pico anterior) / Pico anterior
    
    Exemplo: Se o portfólio atingiu um pico de R$100 e depois caiu para R$80,
    o drawdown seria de -20%.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        
    Retorna:
        dict: Dicionário com 'drawdown_series' (série completa), 'max_drawdown' e 'avg_drawdown'.
    """
    if isinstance(returns, pd.DataFrame):
        portfolio_returns = (returns * weights).sum(axis=1)
    else:
        portfolio_returns = np.dot(returns, weights)
    
    # Garantir que portfolio_returns é uma pandas Series
    if not isinstance(portfolio_returns, pd.Series):
        # Se tivermos índices no DataFrame original, usamos; caso contrário, criamos um índice numérico
        if isinstance(returns, pd.DataFrame):
            portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
        else:
            portfolio_returns = pd.Series(portfolio_returns)
    
    # Calcular retornos cumulativos
    cum_returns = (1 + portfolio_returns).cumprod()
    
    # Calcular máximo acumulado (peak)
    running_max = cum_returns.cummax()
    
    # Calcular drawdowns
    drawdown = (cum_returns - running_max) / running_max
    
    result = {
        'drawdown_series': drawdown,
        'max_drawdown': abs(drawdown.min()),
        'avg_drawdown': abs(drawdown.mean())
    }
    
    return result

def calculate_diversification_ratio(weights, cov_matrix):
    """
    Calcula a Razão de Diversificação do portfólio.
    
    Esta métrica mede o grau de diversificação efetiva do portfólio:
    
    Fórmula: Razão de Diversificação = Soma(pesos × volatilidades individuais) / Volatilidade do portfólio
    
    Interpretação:
    - Valor = 1: Sem benefício de diversificação (como um portfólio de um único ativo)
    - Valor > 1: Maior diversificação (redução de risco através da diversificação)
    - Valores típicos entre 1.2 e 2.0 para portfólios bem diversificados
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        
    Retorna:
        float: Razão de diversificação do portfólio.
    """
    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vols = np.sum(weights * asset_vols)
    portfolio_vol = calculate_volatility(weights, cov_matrix)

    return weighted_vols / portfolio_vol if portfolio_vol > 0 else 0
