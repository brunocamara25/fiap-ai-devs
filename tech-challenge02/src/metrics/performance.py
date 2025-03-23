"""
Módulo com métricas de desempenho para avaliação de portfólios.

Este módulo implementa métricas clássicas e modernas de avaliação de desempenho
de portfólios, considerando diferentes aspectos de risco e retorno.

"""
import numpy as np
import pandas as pd

def calculate_metrics(weights, returns, cov_matrix, risk_free_rate):
    """
    Calcular métricas básicas do portfólio.
    
    A função calcula as métricas fundamentais de um portfólio:
    - Retorno anualizado: r'[ R_p = \sum_{i=1}^{n} w_i \mu_i \times 252 ]'
    - Volatilidade anualizada: r'[ \sigma_p = \sqrt{w^T \Sigma w \times 252} ]'
    - Índice de Sharpe: r'[ SR = \frac{R_p - R_f}{\sigma_p} ]'
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        risk_free_rate (float): Taxa livre de risco.
        
    Retorna:
        tuple: (portfolio_return, portfolio_vol, sharpe_ratio) - Retorno, 
        volatilidade e índice de Sharpe anualizados.
    """
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)  # Normalização dos pesos

    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

    return portfolio_return, portfolio_vol, sharpe_ratio

def calculate_information_ratio(weights, returns, benchmark_returns):
    """
    Calcula o Information Ratio do portfólio.
    
    O Information Ratio mede o retorno ativo (excess return) por unidade de risco ativo:
    \[ IR = \frac{R_p - R_b}{\sigma_{p-b}} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        benchmark_returns (pd.Series): Retornos do benchmark.
        
    Retorna:
        float: Information Ratio do portfólio.
    """
    portfolio_returns = np.dot(returns, weights)
    active_returns = portfolio_returns - benchmark_returns

    # Anualização
    active_return = np.mean(active_returns) * 252
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)

    return active_return / tracking_error if tracking_error > 0 else 0

def calculate_sortino_ratio(weights, returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calcula o Índice de Sortino para um portfólio.
    
    O Índice de Sortino é uma modificação do Índice de Sharpe que considera apenas
    a volatilidade negativa (downside risk):
    \[ Sortino = \frac{R_p - R_f}{\sigma_d} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        risk_free_rate (float): Taxa livre de risco (padrão: 0.0).
        target_return (float): Retorno alvo (padrão: 0.0).
    
    Retorna:
        float: Índice de Sortino anualizado.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_returns = np.dot(returns, weights)

    # Desvio padrão dos retornos abaixo do target (downside risk)
    downside_returns = np.minimum(portfolio_returns - target_return, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

    return (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

def calculate_calmar_ratio(weights, returns, risk_free_rate=0.0, window=252):
    """
    Calcula o Calmar Ratio do portfólio.
    
    O Calmar Ratio é definido como a razão entre o retorno excedente anualizado
    e o máximo drawdown absoluto:
    \[ Calmar = \frac{R_p - R_f}{|MaxDD|} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        risk_free_rate (float): Taxa livre de risco (padrão: 0.0).
        window (int): Janela de tempo em dias para cálculo (padrão: 252 dias).
    
    Retorna:
        float: Calmar Ratio do portfólio.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_returns = np.dot(returns, weights)

    # Calcular drawdown máximo
    cum_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cum_returns.rolling(window=window, min_periods=1).max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min())

    return (portfolio_return - risk_free_rate) / max_drawdown if max_drawdown > 0 else 0

def calculate_beta(weights, returns, market_returns):
    """
    Calcula o beta (sensibilidade ao mercado) de um portfólio.
    
    O beta mede a volatilidade relativa de um portfólio em relação ao mercado:
    \[ \beta_p = \frac{Cov(R_p, R_m)}{Var(R_m)} \]
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        market_returns (pd.Series): Retornos históricos do mercado.
        
    Retorna:
        float: Beta do portfólio.
    """
    portfolio_returns = np.dot(returns, weights)

    # Alinhar dados
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'market': market_returns
    }).dropna()

    # Calcular beta usando covariância/variância
    covariance = np.cov(aligned_data['portfolio'], aligned_data['market'])[0, 1]
    market_variance = np.var(aligned_data['market'])

    return covariance / market_variance if market_variance > 0 else 0

def calculate_treynor_ratio(weights, returns, market_returns, risk_free_rate=0.0):
    """
    Calcula o Índice de Treynor para um portfólio.
    
    O Índice de Treynor mede o excesso de retorno por unidade de risco sistemático (beta):
    \[ Treynor = \frac{R_p - R_f}{\beta_p} \]
    
    Esta métrica é especialmente útil para avaliar portfólios bem diversificados,
    onde o risco não-sistemático foi minimizado.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        market_returns (pd.Series): Retornos históricos do mercado.
        risk_free_rate (float): Taxa livre de risco (padrão: 0.0).
        
    Retorna:
        float: Índice de Treynor anualizado.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    beta = calculate_beta(weights, returns, market_returns)

    # Evitar divisão por beta zero ou muito pequeno
    if abs(beta) < 1e-6:
        return float('inf') if portfolio_return > risk_free_rate else float('-inf')

    return (portfolio_return - risk_free_rate) / beta
