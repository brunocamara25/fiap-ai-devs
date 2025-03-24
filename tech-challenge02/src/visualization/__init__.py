"""
Pacote de visualização para análise de portfólios de investimento.

Este pacote contém funções para visualização de dados e resultados relacionados à
análise e otimização de portfólios de investimento.
"""

from .plots import (
    # Visualizações essenciais
    plot_correlation_matrix,
    plot_efficient_frontier,
    plot_portfolio_allocation,
    plot_cumulative_returns,
    plot_drawdowns,
    
    # Visualizações para algoritmo genético
    plot_ga_evolution,
    plot_pareto_front,
    
    # Função de dashboard completo
    create_dashboard
)

# Definição da função create_interactive_dashboard com parâmetros adicionais
def create_interactive_dashboard(weights, tickers, risk_free_rate, investment=None, 
                                show_correlation=True, show_risk=True, show_ga_evolution=True,
                                returns=None, cov_matrix=None, start_date=None, end_date=None,
                                pareto_front=None, best_history=None):
    """
    Wrapper da função create_dashboard com parâmetros adicionais para controlar quais visualizações exibir.
    
    Parameters
    ----------
    weights : np.ndarray
        Pesos otimizados do portfólio
    tickers : list
        Lista de tickers/códigos dos ativos
    risk_free_rate : float
        Taxa livre de risco anualizada
    investment : float, optional
        Valor do investimento
    show_correlation : bool, default=True
        Se True, exibe análise de correlação
    show_risk : bool, default=True
        Se True, exibe análise de risco
    show_ga_evolution : bool, default=True
        Se True, exibe evolução do algoritmo genético
    returns : pd.DataFrame, optional
        DataFrame com os retornos diários. Se None, baixa os dados
    cov_matrix : pd.DataFrame, optional
        Matriz de covariância. Se None, calcula a partir dos retornos
    start_date : str or datetime, optional
        Data inicial para baixar dados (se returns=None)
    end_date : str or datetime, optional
        Data final para baixar dados (se returns=None)
    pareto_front : array-like, optional
        Conjunto de soluções pareto-ótimas para problemas multiobjetivo
    best_history : array-like, optional
        Histórico de evolução das melhores soluções do algoritmo genético
    """
    from src.data.loader import download_data
    import pandas as pd
    import streamlit as st
    import datetime
    import numpy as np
    
    # Verificar se temos todos os dados necessários
    if weights is None:
        st.error("Pesos do portfólio não fornecidos.")
        return
    
    # Se não foram fornecidos retornos e matriz de covariância, baixar dados
    if returns is None or cov_matrix is None:
        # Baixar dados para obter retornos e matriz de covariância
        with st.spinner("Carregando dados para visualizações..."):
            # Definir período de dados
            if start_date is None or end_date is None:
                # Se não foram especificados, usar os últimos 2 anos
                end_date = pd.Timestamp.today()
                start_date = end_date - pd.DateOffset(years=2)
                
            # Converter para string se forem datetime
            if isinstance(start_date, (datetime.date, pd.Timestamp)):
                start_date = start_date.strftime("%Y-%m-%d")
            if isinstance(end_date, (datetime.date, pd.Timestamp)):
                end_date = end_date.strftime("%Y-%m-%d")
            
            # Baixar dados
            st.info(f"Baixando dados de {start_date} até {end_date}")
            data = download_data(tickers, start_date, end_date)
            
            if data is None or data.empty:
                st.error("Não foi possível baixar os dados para o período especificado.")
                return
                
            # Calcular retornos e matriz de covariância
            returns = data.pct_change().dropna()
            cov_matrix = returns.cov()
    
    # Verificar se temos todos os dados necessários após download
    if returns is None or cov_matrix is None:
        st.error("Dados insuficientes para criar o dashboard.")
        return
    
    # Verificação e correção de incompatibilidade de dimensões
    try:
        st.info(f"Verificando dados: {len(weights)} pesos, {len(returns.columns)} ativos, {returns.shape[0]} dias, {cov_matrix.shape}")
        
        # Verificar se o número de pesos é igual ao número de colunas em returns
        if len(weights) != len(returns.columns):
            st.warning(f"Incompatibilidade de dimensões: {len(weights)} pesos vs {len(returns.columns)} colunas. Tentando corrigir...")
            
            # Verificar se os tickers correspondem às colunas de returns
            if set(tickers) != set(returns.columns):
                st.warning("Tickers fornecidos não correspondem às colunas de retornos.")
                
                # Tentar usar apenas as colunas correspondentes aos tickers
                if set(tickers).issubset(set(returns.columns)):
                    st.info("Filtrando retornos para usar apenas os tickers selecionados.")
                    returns = returns[tickers]
                    cov_matrix = returns.cov()
        
        # Se ainda houver incompatibilidade, desista
        if len(weights) != len(returns.columns):
            st.error(f"Não foi possível resolver a incompatibilidade de dimensões: {len(weights)} pesos vs {len(returns.columns)} colunas")
            
            # Ajustar pesos ou extrair uma submatriz de cov_matrix
            if len(weights) > len(returns.columns):
                st.warning("Ajustando pesos para corresponder aos retornos disponíveis.")
                weights = weights[:len(returns.columns)]
                weights = weights / np.sum(weights)  # Renormalizar
            else:
                st.warning("Não há ativos suficientes nos dados. Usando apenas os disponíveis.")
                # Não podemos continuar
                return
    except Exception as e:
        st.error(f"Erro ao verificar dimensões: {str(e)}")
        return
    
    # Criar dashboard
    try:
        create_dashboard(
            returns=returns,
            cov_matrix=cov_matrix,
            weights=weights,
            tickers=tickers[:len(weights)],  # Garantir que tickers e weights tenham o mesmo tamanho
            risk_free_rate=risk_free_rate,
            investment=investment,
            pareto_front=pareto_front,
            best_history=best_history
        )
    except Exception as e:
        st.error(f"Erro ao criar o dashboard: {str(e)}")
        st.error("Detalhes do problema:")
        st.write(f"Dimensões - Pesos: {len(weights)}, Tickers: {len(tickers)}, Colunas de retornos: {len(returns.columns)}, Matriz de covariância: {cov_matrix.shape}")

__all__ = [
    # Visualizações essenciais
    "plot_correlation_matrix",
    "plot_efficient_frontier",
    "plot_portfolio_allocation",
    "plot_cumulative_returns",
    "plot_drawdowns",

    # Visualizações para algoritmo genético
    "plot_ga_evolution",
    "plot_pareto_front",

    # Função de dashboard completo
    "create_interactive_dashboard"
]
