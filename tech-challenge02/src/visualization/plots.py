"""
Módulo de visualização para análise de portfólios de investimento.

Este módulo fornece funções para visualização de dados e resultados relacionados à
análise e otimização de portfólios de investimento, incluindo:
- Matrizes de correlação
- Fronteira eficiente
- Alocação de ativos
- Séries temporais de retornos e drawdowns
- Evolução do algoritmo genético

O módulo é projetado para integração com Streamlit e utiliza
principalmente Plotly para gráficos interativos.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import uuid

from src.metrics.performance import calculate_metrics, calculate_sortino_ratio
from src.metrics.risk import calculate_var, calculate_cvar, calculate_drawdown

# Paleta de cores padrão para consistência visual
COLORS = px.colors.qualitative.Plotly
ACCENT_COLOR = "#2E5090"  # Azul para destaque
RISK_COLOR = "#D62728"    # Vermelho para risco
RETURN_COLOR = "#2CA02C"  # Verde para retorno

# ==============================================================================
# FUNÇÕES DE VISUALIZAÇÃO ESSENCIAIS
# ==============================================================================

def plot_correlation_matrix(returns, title="Matriz de Correlação", interactive=True):
    """
    Cria um heatmap da matriz de correlação entre ativos.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos das ações
    title : str, optional
        Título do gráfico
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo
        
    Returns
    -------
    fig
        Figura de matplotlib ou plotly
    """
    correlation_matrix = returns.corr()

    if interactive:
        fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(correlation_matrix.values, 2),
                hovertemplate='%{y} - %{x}<br>Correlação: %{z:.4f}<extra></extra>',
                colorbar=dict(title='Correlação')
            ))

        fig.update_layout(
            title=title,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed'
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.title(title, fontsize=14)
        plt.tight_layout()

        return fig

def plot_efficient_frontier(returns, cov_matrix, risk_free_rate=0.0, num_portfolios=100, interactive=True):
    """
    Plota a fronteira eficiente com diferentes portfólios.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos das ações
    cov_matrix : pd.DataFrame
        Matriz de covariância
    risk_free_rate : float, optional
        Taxa livre de risco
    num_portfolios : int, optional
        Número de portfólios a simular
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo
        
    Returns
    -------
    fig
        Figura de matplotlib ou plotly
    """
    # Gerar pesos aleatórios para portfólios
    np.random.seed(42)
    all_weights = []
    ret_arr = []
    vol_arr = []
    sharpe_arr = []

    # Simular portfólios aleatórios
    num_assets = len(returns.columns)
    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights / np.sum(weights)
        all_weights.append(weights)

        # Calcular métricas
        portfolio_return, portfolio_vol, sharpe = calculate_metrics(
            weights, returns, cov_matrix, risk_free_rate)

        ret_arr.append(portfolio_return)
        vol_arr.append(portfolio_vol)
        sharpe_arr.append(sharpe)

    # Encontrar portfólio de menor volatilidade
    min_vol_idx = np.argmin(vol_arr)
    min_vol_ret = ret_arr[min_vol_idx]
    min_vol_vol = vol_arr[min_vol_idx]

    # Encontrar portfólio de máximo Sharpe
    max_sharpe_idx = np.argmax(sharpe_arr)
    max_sharpe_ret = ret_arr[max_sharpe_idx]
    max_sharpe_vol = vol_arr[max_sharpe_idx]

    if interactive:
        fig = go.Figure()

        # Adicionar nuvem de portfólios simulados
        fig.add_trace(go.Scatter(
            x=vol_arr,
            y=ret_arr,
            mode='markers',
            marker=dict(
                size=6,
                color=sharpe_arr,
                colorscale='Viridis',
                colorbar=dict(title='Índice Sharpe'),
                line=dict(width=1)
            ),
            text=[f"Sharpe: {s:.3f}" for s in sharpe_arr],
            hovertemplate="Retorno: %{y:.2%}<br>Volatilidade: %{x:.2%}<br>%{text}<extra></extra>",
            name='Portfólios Simulados'
        ))

        # Adicionar portfólio de mínima volatilidade
        fig.add_trace(go.Scatter(
            x=[min_vol_vol],
            y=[min_vol_ret],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='star'
            ),
            hovertemplate="Mínima Volatilidade<br>Retorno: %{y:.2%}<br>Volatilidade: %{x:.2%}<br>",
            name='Mínima Volatilidade'
        ))

        # Adicionar portfólio de máximo Sharpe
        fig.add_trace(go.Scatter(
            x=[max_sharpe_vol],
            y=[max_sharpe_ret],
            mode='markers',
            marker=dict(
                color='green',
                size=12,
                symbol='star'
            ),
            hovertemplate="Máximo Sharpe<br>Retorno: %{y:.2%}<br>Volatilidade: %{x:.2%}<br>",
            name='Máximo Sharpe'
        ))

        # Adicionar Capital Market Line (CML)
        x_cml = [0, max_sharpe_vol * 1.5]
        slope = (max_sharpe_ret - risk_free_rate) / max_sharpe_vol
        y_cml = [risk_free_rate, risk_free_rate + slope * x_cml[1]]

        fig.add_trace(go.Scatter(
            x=x_cml,
            y=y_cml,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='CML'
        ))

        fig.update_layout(
            title='Fronteira Eficiente de Markowitz',
            xaxis_title='Volatilidade Anualizada',
            yaxis_title='Retorno Anualizado',
            yaxis=dict(tickformat='.1%'),
            xaxis=dict(tickformat='.1%'),
            legend=dict(x=0.01, y=0.99),
            hovermode='closest'
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Índice Sharpe')
        plt.title('Fronteira Eficiente de Markowitz', fontsize=14)
        plt.xlabel('Volatilidade Anualizada')
        plt.ylabel('Retorno Anualizado')
        plt.tight_layout()

        return fig

def plot_portfolio_allocation(weights, tickers, title="Alocação do Portfólio", interactive=True):
    """
    Cria um gráfico de pizza para visualizar a alocação do portfólio.
    
    Parameters
    ----------
    weights : np.ndarray
        Pesos dos ativos no portfólio
    tickers : list
        Lista com os nomes/códigos dos ativos
    title : str, optional
        Título do gráfico
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo
        
    Returns
    -------
    fig
        Figura de matplotlib ou plotly
    """
    # Validar entradas
    if len(weights) != len(tickers):
        raise ValueError("O número de pesos deve ser igual ao número de tickers")

    # Normalizar pesos
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    # Criar DataFrame para os pesos
    df = pd.DataFrame({
        'Ativo': tickers,
        'Peso': [w * 100 for w in weights]  # Converter para porcentagem
    })

    # Ordenar por peso, do maior para o menor
    df = df.sort_values('Peso', ascending=False).reset_index(drop=True)

    if interactive:
        fig = px.pie(
            df, 
            values='Peso', 
            names='Ativo', 
            title=title,
            color_discrete_sequence=COLORS,
            hover_data={'Peso': ':.2f%'}
        )

        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Peso: %{value:.2f}%<extra></extra>'
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(
            df['Peso'], 
            labels=df['Ativo'], 
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        ax.set_title(title, fontsize=14)
        ax.axis('equal')
        plt.tight_layout()

        return fig

def plot_cumulative_returns(returns, weights=None, benchmark_returns=None, interactive=True):
    """
    Plota os retornos acumulados dos ativos e do portfólio.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários dos ativos
    weights : np.ndarray, optional
        Pesos dos ativos no portfólio
    benchmark_returns : pd.Series, optional
        Retornos do benchmark
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo
        
    Returns
    -------
    fig
        Figura de matplotlib ou plotly
    """
    # Calcular retornos acumulados
    cum_returns = (1 + returns).cumprod()

    # Calcular retorno do portfólio, se pesos forem fornecidos
    if weights is not None:
        portfolio_returns = returns.dot(weights)
        portfolio_cum_returns = (1 + portfolio_returns).cumprod()

    # Calcular retorno acumulado do benchmark, se fornecido
    if benchmark_returns is not None:
        benchmark_cum_returns = (1 + benchmark_returns).cumprod()

    if interactive:
        fig = go.Figure()

        # Adicionar linhas para cada ativo
        for column in cum_returns.columns:
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[column],
                mode='lines',
                name=column,
                opacity=0.7,
                hovertemplate="Data: %{x}<br>" + f"{column}: %{{y:.2f}}<br>Retorno: %{{customdata:.2%}}<extra></extra>",
                customdata=cum_returns[column].values - 1
            ))

        # Adicionar linha para o portfólio, se fornecido
        if weights is not None:
            fig.add_trace(go.Scatter(
                x=portfolio_cum_returns.index,
                y=portfolio_cum_returns.values,
                mode='lines',
                name='Portfólio',
                line=dict(color='black', width=3),
                hovertemplate="Data: %{x}<br>Portfólio: %{y:.2f}<br>Retorno: %{customdata:.2%}<extra></extra>",
                customdata=portfolio_cum_returns.values - 1
            ))

        # Adicionar linha para o benchmark, se fornecido
        if benchmark_returns is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_cum_returns.index,
                y=benchmark_cum_returns.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate="Data: %{x}<br>Benchmark: %{y:.2f}<br>Retorno: %{customdata:.2%}<extra></extra>",
                customdata=benchmark_cum_returns.values - 1
            ))

        fig.update_layout(
            title='Retornos Acumulados',
            xaxis_title='Data',
            yaxis_title='Valor',
            hovermode='x unified'
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        cum_returns.plot(ax=ax, alpha=0.7)

        if weights is not None:
            portfolio_cum_returns.plot(ax=ax, color='black', linewidth=3, label='Portfólio')

        if benchmark_returns is not None:
            benchmark_cum_returns.plot(ax=ax, color='red', linewidth=2, linestyle=':', label='Benchmark')

        plt.title('Retornos Acumulados', fontsize=14)
        plt.xlabel('Data')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        return fig

def plot_drawdowns(returns, weights, interactive=True):
    """
    Plota os drawdowns do portfólio ao longo do tempo.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários dos ativos
    weights : np.ndarray
        Pesos dos ativos no portfólio
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo
        
    Returns
    -------
    fig
        Figura de matplotlib ou plotly
    """
    # Calcular retorno do portfólio
    portfolio_returns = returns.dot(weights)

    # Calcular drawdowns
    dd_info = calculate_drawdown(weights, returns)
    drawdowns = dd_info.get('drawdown_series', pd.Series())

    if interactive:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values,
            mode='lines',
            name='Drawdown',
            line=dict(color=RISK_COLOR),
            fill='tozeroy',
            fillcolor=f'rgba(214, 39, 40, 0.3)',
            hovertemplate="Data: %{x}<br>Drawdown: %{y:.2%}<extra></extra>"
        ))

        fig.update_layout(
            title='Drawdowns do Portfólio',
            xaxis_title='Data',
            yaxis_title='Drawdown (%)',
            yaxis=dict(tickformat='.0%')
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color=RISK_COLOR)
        ax.plot(drawdowns.index, drawdowns.values, color=RISK_COLOR)
        plt.title('Drawdowns do Portfólio', fontsize=14)
        plt.xlabel('Data')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

# ==============================================================================
# FUNÇÕES DE VISUALIZAÇÃO DO ALGORITMO GENÉTICO
# ==============================================================================

def plot_ga_evolution(best_history, interactive=True):
    """
    Plota a evolução do fitness ao longo das gerações do algoritmo genético.
    
    Parameters
    ----------
    best_history : list
        Histórico do melhor fitness por geração
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo
        
    Returns
    -------
    fig
        Figura de matplotlib ou plotly
    """
    generations = list(range(1, len(best_history) + 1))

    if interactive:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=generations,
            y=best_history,
            mode='lines+markers',
            name='Melhor Fitness',
            line=dict(color=ACCENT_COLOR, width=2),
            marker=dict(size=6),
            hovertemplate="Geração %{x}<br>Fitness: %{y:.4f}<extra></extra>"
        ))

        fig.update_layout(
            title='Evolução do Algoritmo Genético',
            xaxis_title='Geração',
            yaxis_title='Fitness'
        )

        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(generations, best_history, 'o-', color=ACCENT_COLOR, linewidth=2, markersize=6)
        plt.title('Evolução do Algoritmo Genético', fontsize=14)
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig

def plot_pareto_front(pareto_front_history, risk_free_rate=0.0, interactive=True):
    """
    Plota a evolução da fronteira de Pareto (fronteira eficiente).

    Parameters
    ----------
    pareto_front_history : list
        Lista de listas de tuplas (weights, (return, risk)).
    risk_free_rate : float, optional
        Taxa livre de risco.
    interactive : bool, optional
        Se True, usa Plotly para gráfico interativo.

    Returns
    -------
    fig
        Figura de matplotlib ou plotly.
    """
    if interactive:
        import plotly.graph_objects as go
        fig = go.Figure()

        for generation, pareto_front in enumerate(pareto_front_history):
            # Extract returns and risks from the current generation's pareto front
            rets = [score[0] for _, score in pareto_front]
            vols = [score[1] for _, score in pareto_front]
            sharpes = [(r - risk_free_rate) / v if v > 0 else 0 for r, v in zip(rets, vols)]

            # Add scatter plot for the current generation
            fig.add_trace(go.Scatter(
                x=vols,
                y=rets,
                mode='markers+lines',
                name=f'Geração {generation + 1}',
                marker=dict(size=8, color=sharpes, colorscale='Viridis', colorbar=dict(title='Índice Sharpe')),
                line=dict(dash='dash'),
                hovertemplate="Volatilidade: %{x:.2%}<br>Retorno: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title='Evolução do Pareto Front',
            xaxis_title='Risco (Volatilidade)',
            yaxis_title='Retorno',
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            legend_title='Gerações'
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        for generation, pareto_front in enumerate(pareto_front_history):
            # Extract returns and risks from the current generation's pareto front
            rets = [score[0] for _, score in pareto_front]
            vols = [score[1] for _, score in pareto_front]
            sharpes = [(r - risk_free_rate) / v if v > 0 else 0 for r, v in zip(rets, vols)]

            # Add scatter plot for the current generation
            scatter = ax.scatter(vols, rets, label=f'Geração {generation + 1}', alpha=0.6, c=sharpes, cmap='viridis')

        ax.set_xlabel('Risco (Volatilidade)')
        ax.set_ylabel('Retorno')
        ax.set_title('Evolução do Pareto Front')
        ax.legend()
        ax.grid(True)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Índice Sharpe')

        return fig

# ==============================================================================
# DASHBOARD COMPLETO
# ==============================================================================

def create_dashboard(returns, cov_matrix, weights, tickers, risk_free_rate,
                    investment=None, benchmark_returns=None, pareto_front=None, best_history=None):
    """
    Cria um dashboard interativo com análises de portfólio para Streamlit.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários
    cov_matrix : pd.DataFrame
        Matriz de covariância
    weights : np.ndarray
        Pesos do portfólio otimizado
    tickers : list
        Lista de tickers
    risk_free_rate : float
        Taxa livre de risco
    investment : float, optional
        Valor do investimento
    benchmark_returns : pd.Series, optional
        Retornos do benchmark
    pareto_front : list, optional
        Fronteira de Pareto do algoritmo genético
    best_history : list, optional
        Histórico do melhor fitness por geração
    """
    # Calcular métricas básicas
    ret, vol, sharpe = calculate_metrics(weights, returns, cov_matrix, risk_free_rate)
    sortino = calculate_sortino_ratio(weights, returns, risk_free_rate)
    var_95 = calculate_var(weights, returns, 0.95)
    cvar_95 = calculate_cvar(weights, returns, 0.95)
    dd_info = calculate_drawdown(weights, returns)
    max_dd = dd_info.get('max_drawdown', 0)

    # Criar abas para organizar o dashboard
    tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "📈 Análise de Retornos", "🔍 Análise de Risco"])

    with tab1:
        # Visão geral
        st.header("Visão Geral do Portfólio")

        # Exibir métricas em colunas
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Métricas do Portfólio")
            metrics_df = pd.DataFrame({
                'Métrica': ['Retorno Anualizado', 'Volatilidade Anualizada', 'Índice Sharpe',
                          'Índice Sortino', 'VaR (95%)', 'CVaR (95%)', 'Max Drawdown'],
                'Valor': [f"{ret:.2%}", f"{vol:.2%}", f"{sharpe:.2f}", 
                        f"{sortino:.2f}", f"{var_95:.2%}", f"{cvar_95:.2%}", f"{max_dd:.2%}"]
            })
            st.dataframe(metrics_df, hide_index=True)

            # Projeção de investimento, se fornecido
            if investment is not None:
                st.subheader("💰 Projeção de Investimento")
                monthly_return = (1 + ret) ** (1/12) - 1
                projection_df = pd.DataFrame({
                    'Horizonte': ['1 mês', '3 meses', '6 meses', '1 ano'],
                    'Retorno Esperado': [
                        f"{(1 + monthly_return) - 1:.2%}",
                        f"{(1 + monthly_return) ** 3 - 1:.2%}",
                        f"{(1 + monthly_return) ** 6 - 1:.2%}",
                        f"{(1 + ret) - 1:.2%}"
                    ],
                    'Valor Projetado': [
                        f"${investment * (1 + monthly_return):,.2f}",
                        f"${investment * (1 + monthly_return) ** 3:,.2f}",
                        f"${investment * (1 + monthly_return) ** 6:,.2f}",
                        f"${investment * (1 + ret):,.2f}"
                    ]
                })
                st.dataframe(projection_df, hide_index=True)

        with col2:
            # Gráfico de alocação
            fig = plot_portfolio_allocation(weights, tickers)
            st.plotly_chart(fig, use_container_width=True)

            # Detalhes da alocação em tabela
            if investment is not None:
                allocation_df = pd.DataFrame({
                    'Ativo': tickers,
                    'Peso': [f"{w*100:.2f}%" for w in weights],
                    'Valor': [f"${w*investment:,.2f}" for w in weights]
                })
                st.dataframe(allocation_df, hide_index=True)

        # Evolução do algoritmo genético, se disponível
        if best_history is not None:
            st.subheader("🧬 Evolução do Algoritmo")
            fig = plot_ga_evolution(best_history)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Análise de retornos
        st.header("Análise de Retornos")

        # Retornos acumulados
        st.subheader("Retornos Acumulados")
        fig = plot_cumulative_returns(returns, weights, benchmark_returns)
        st.plotly_chart(fig, use_container_width=True)

        # Fronteira eficiente
        st.subheader("Fronteira Eficiente")
        if pareto_front is not None:
            fig = plot_pareto_front(pareto_front, risk_free_rate)
        else:
            fig = plot_efficient_frontier(returns, cov_matrix, risk_free_rate)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Análise de risco
        st.header("Análise de Risco")

        # Drawdowns
        st.subheader("Drawdowns")
        fig = plot_drawdowns(returns, weights)
        st.plotly_chart(fig, use_container_width=True)

        # Matriz de correlação
        st.subheader("Matriz de Correlação")
        fig = plot_correlation_matrix(returns)
        st.plotly_chart(fig, use_container_width=True)

    # Botão para baixar alocação como CSV
    allocation_df = pd.DataFrame({
        'Ativo': tickers,
        'Peso (%)': [w * 100 for w in weights]
    })

    if investment is not None:
        allocation_df['Valor Alocado ($)'] = [w * investment for w in weights]

    csv = allocation_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar Alocação como CSV",
        data=csv,
        file_name="portfolio_allocation.csv",
        mime="text/csv",
        key=f"download_{uuid.uuid4().hex}"
    )