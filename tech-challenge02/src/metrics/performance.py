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
    - Retorno anualizado: Média ponderada dos retornos dos ativos multiplicada por 252
      Fórmula: R_portfólio = (soma dos pesos × retornos médios) × 252
    
    - Volatilidade anualizada: Raiz quadrada da variância do portfólio anualizada
      Fórmula: Volatilidade = raiz(pesos × matriz covariância × pesos × 252)
    
    - Índice de Sharpe: Retorno excedente dividido pela volatilidade
      Fórmula: Sharpe = (Retorno do portfólio - Taxa livre de risco) / Volatilidade
    
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
    
    O Information Ratio mede quanto o portfólio supera o benchmark por unidade de risco:
    Fórmula: IR = (Retorno do portfólio - Retorno do benchmark) / Tracking Error
    
    Onde o Tracking Error é o desvio padrão da diferença entre os retornos do 
    portfólio e do benchmark.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        benchmark_returns (pd.Series): Retornos do benchmark.
        
    Retorna:
        float: Information Ratio do portfólio.
    """
    # Garantir que os índices estejam alinhados
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Verificar se benchmark_returns é uma Series ou DataFrame
    if isinstance(benchmark_returns, pd.DataFrame):
        benchmark_col = benchmark_returns.columns[0]
        benchmark_returns = benchmark_returns[benchmark_col]
    
    # Alinhar datas entre portfólio e benchmark
    common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common_dates) == 0:
        return 0  # Sem datas comuns, retorna 0
    
    aligned_portfolio = portfolio_returns.loc[common_dates]
    aligned_benchmark = benchmark_returns.loc[common_dates]
    
    # Cálculo com dados alinhados
    active_returns = aligned_portfolio - aligned_benchmark

    # Anualização
    active_return = np.mean(active_returns) * 252
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)

    return active_return / tracking_error if tracking_error > 0 else 0

def calculate_sortino_ratio(weights, returns, risk_free_rate=0.0, target_return=0.0):
    """
    Calcula o Índice de Sortino para um portfólio.
    
    O Índice de Sortino é similar ao Sharpe, mas considera apenas o risco de queda:
    Fórmula: Sortino = (Retorno do portfólio - Taxa livre de risco) / Downside Deviation
    
    Onde Downside Deviation mede apenas a volatilidade dos retornos negativos 
    (abaixo do retorno alvo), ignorando os retornos positivos.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        risk_free_rate (float): Taxa livre de risco (padrão: 0.0).
        target_return (float): Retorno alvo (padrão: 0.0).
    
    Retorna:
        float: Índice de Sortino anualizado.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    
    # Calcular retornos do portfólio
    if isinstance(returns, pd.DataFrame):
        portfolio_returns = (returns * weights).sum(axis=1)
    else:
        portfolio_returns = np.dot(returns, weights)
    
    # Desvio padrão dos retornos abaixo do target (downside risk)
    downside_returns = np.minimum(portfolio_returns - target_return, 0)
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)

    return (portfolio_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0

def calculate_calmar_ratio(weights, returns, risk_free_rate=0.0, window=252):
    """
    Calcula o Calmar Ratio do portfólio.
    
    O Calmar Ratio relaciona o retorno com o máximo drawdown (maior queda):
    Fórmula: Calmar = (Retorno do portfólio - Taxa livre de risco) / Máximo Drawdown
    
    Um Calmar Ratio maior indica melhor retorno por unidade de risco de queda extrema.
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        risk_free_rate (float): Taxa livre de risco (padrão: 0.0).
        window (int): Janela de tempo em dias para cálculo (padrão: 252 dias).
    
    Retorna:
        float: Calmar Ratio anualizado.
    """
    portfolio_return = np.sum(returns.mean() * weights) * 252
    
    # Calcular retornos do portfólio
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
    
    # Calcular preços cumulativos
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.rolling(window=window, min_periods=1).max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    return (portfolio_return - risk_free_rate) / max_drawdown if max_drawdown > 0 else 0

def calculate_beta(weights, returns, market_returns):
    """
    Calcula o beta (β) do portfólio em relação ao mercado.
    
    O beta mede a sensibilidade do portfólio às movimentações do mercado:
    
    Fórmula: Beta = Covariância(retornos do portfólio, retornos do mercado) / Variância(retornos do mercado)
    
    Interpretação:
    - Beta = 1.0: O portfólio tende a se mover na mesma proporção que o mercado
    - Beta > 1.0: O portfólio é mais volátil que o mercado (ex: beta 1.2 significa 20% mais volátil)
    - Beta < 1.0: O portfólio é menos volátil que o mercado
    - Beta negativo: O portfólio tende a se mover na direção oposta ao mercado
    
    Parâmetros:
        weights (np.ndarray): Pesos do portfólio.
        returns (pd.DataFrame): Retornos históricos das ações.
        market_returns (pd.Series): Retornos históricos do mercado.
    
    Retorna:
        float: Beta do portfólio.
    """
    # Garantir que os índices estejam alinhados
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Verificar se market_returns é uma Series ou DataFrame
    if isinstance(market_returns, pd.DataFrame):
        market_col = market_returns.columns[0]
        market_returns = market_returns[market_col]
    
    # Alinhar datas entre portfólio e mercado
    common_dates = portfolio_returns.index.intersection(market_returns.index)
    if len(common_dates) == 0:
        return 1.0  # Sem datas comuns, retorna beta neutro (1.0)
    
    aligned_portfolio = portfolio_returns.loc[common_dates]
    aligned_market = market_returns.loc[common_dates]
    
    # Calcular beta usando covariância/variância com dados alinhados
    covariance = np.cov(aligned_portfolio, aligned_market)[0, 1]
    market_variance = np.var(aligned_market)

    return covariance / market_variance if market_variance > 0 else 0

def calculate_treynor_ratio(weights, returns, market_returns, risk_free_rate=0.0):
    """
    Calcula o Índice de Treynor para um portfólio.
    
    O Índice de Treynor mede o retorno excedente por unidade de risco sistemático (beta):
    
    Fórmula: Treynor = (Retorno do portfólio - Taxa livre de risco) / Beta
    
    Esta métrica é importante para:
    - Avaliar portfólios bem diversificados (onde o risco não-sistemático foi minimizado)
    - Comparar desempenho considerando apenas o risco não-diversificável
    - Identificar portfólios que geram mais retorno por unidade de risco de mercado
    
    Um Treynor maior indica melhor desempenho ajustado ao risco sistemático.
    
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
