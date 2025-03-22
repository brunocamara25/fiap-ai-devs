"""
Módulo com funções para visualização de dados e resultados de otimização de portfólio.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import numpy as np
import uuid

from src.metrics.performance import calculate_metrics
from src.metrics.risk import calculate_diversification, calculate_var

def display_stock_prices(data):
    """
    Exibe preços das ações na interface Streamlit.
    
    Parâmetros:
        data (pd.DataFrame): DataFrame com os preços ajustados das ações.
    """
    st.subheader("Preços Atuais das Ações")
    latest_prices = data.iloc[-1]
    price_df = pd.DataFrame({
        'Ação': latest_prices.index,
        'Preço': latest_prices.values
    })
    st.dataframe(price_df.style.format({'Preço': '${:.2f}'}))

def display_progress_chart(best_history):
    """
    Exibe gráfico de progresso da otimização.
    
    Parâmetros:
        best_history (list): Lista com o histórico do melhor índice por geração.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_history, 'b-')
    ax.set_xlabel('Geração')
    ax.set_ylabel('Melhor Índice')
    ax.set_title('Progresso da Otimização')
    ax.grid(True)
    st.pyplot(fig)

def display_allocation_chart(best_weights, selected_tickers):
    """
    Exibe gráfico de alocação do portfólio em forma de pizza.
    
    Parâmetros:
        best_weights (np.ndarray): Pesos do melhor portfólio.
        selected_tickers (list): Lista de códigos das ações selecionadas.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%')
    ax.set_title('Melhor Alocação de Portfólio Atual')
    st.pyplot(fig)

def plot_pareto_front(pareto_front_history):
    """
    Exibe o gráfico de evolução do Pareto Front ao longo das gerações.
    
    Parâmetros:
        pareto_front_history (list): Lista com o histórico do Pareto Front por geração.
    """
    st.subheader("Evolução do Pareto Front")
    fig, ax = plt.subplots(figsize=(10, 6))
    for generation, pareto_front in enumerate(pareto_front_history):
        returns = [score[0] for _, score in pareto_front]
        risks = [score[1] for _, score in pareto_front]
        ax.scatter(risks, returns, label=f'Geração {generation + 1}', alpha=0.6)
    ax.set_xlabel('Risco (Volatilidade)')
    ax.set_ylabel('Retorno')
    ax.set_title('Evolução do Pareto Front')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def display_initial_data(data, train_cov_matrix):
    """
    Exibe dados iniciais do portfólio, como preços e diversificação.
    
    Parâmetros:
        data (pd.DataFrame): DataFrame com os preços ajustados das ações.
        train_cov_matrix (pd.DataFrame): Matriz de covariância dos retornos de treinamento.
    """
    latest_prices = data.iloc[-1]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Preços Atuais das Ações")
        st.dataframe(pd.DataFrame({'Ação': latest_prices.index, 'Preço': latest_prices.values}).style.format({'Preço': '${:.2f}'}))
        avg_correlation = calculate_diversification(train_cov_matrix)
        st.markdown(f"**Correlação Média do Portfólio:** {avg_correlation:.2f}")

    with col2:
        st.subheader("Retorno das Ações")
        fig, ax = plt.subplots(figsize=(10, 6))
        (train_cov_matrix + 1).cumprod().plot(ax=ax)
        ax.set_title("Retornos Acumulados")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

def update_progress_display(generation, num_generations, progress_bar, status_text, metrics_text, best_weights, train_data, train_cov_matrix, risk_free_rate, evaluation_method, best_sharpe):
    """
    Atualiza a exibição de progresso e métricas durante a otimização.
    
    Parâmetros:
        generation (int): Geração atual.
        num_generations (int): Número total de gerações.
        progress_bar: Barra de progresso do Streamlit.
        status_text: Container de texto para status.
        metrics_text: Container de texto para métricas.
        best_weights (np.ndarray): Pesos do melhor portfólio.
        train_data (pd.DataFrame): Dados de treinamento.
        train_cov_matrix (pd.DataFrame): Matriz de covariância dos retornos de treinamento.
        risk_free_rate (float): Taxa livre de risco.
        evaluation_method (str): Método de avaliação.
        best_sharpe (float): Melhor índice Sharpe encontrado.
    """
    progress_bar.progress((generation + 1) / num_generations)
    status_text.text(f"Geração {generation + 1}/{num_generations}")
    if best_weights is not None:
        ret, vol, _ = calculate_metrics(best_weights, train_data, train_cov_matrix, risk_free_rate)
        metrics_text.markdown(f"""
        **Melhor Portfólio Atual:**
        - Retorno Anual Esperado: {ret:.2%}
        - Volatilidade Anual Esperada: {vol:.2%}
        - Índice {evaluation_method}: {best_sharpe:.4f}
        """)

def update_progress_chart(progress_chart, best_history):
    """
    Atualiza o gráfico de progresso da otimização.
    
    Parâmetros:
        progress_chart: Container para o gráfico.
        best_history (list): Lista com o histórico do melhor índice por geração.
    """
    with progress_chart.container():
        st.subheader("Progresso da Otimização")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(best_history, 'b-')
        ax.set_xlabel('Geração')
        ax.set_ylabel('Melhor Índice')
        ax.set_title('Progresso da Otimização')
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

def update_allocation_chart(allocation_chart, best_weights, selected_tickers):
    """
    Atualiza o gráfico de alocação do portfólio.
    
    Parâmetros:
        allocation_chart: Container para o gráfico.
        best_weights (np.ndarray): Pesos do melhor portfólio.
        selected_tickers (list): Lista de códigos das ações selecionadas.
    """
    if best_weights is not None:
        with allocation_chart.container():
            st.subheader("Alocação do Portfólio")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%')
            ax.set_title('Melhor Alocação de Portfólio Atual')
            st.pyplot(fig)
            plt.close(fig)

def display_final_results(best_weights, test_data, test_cov_matrix, risk_free_rate, investment, selected_tickers, data, train_cov_matrix, returns, evaluation_method, pareto_front_history, best_history, benchmark_returns):
    """
    Exibe os resultados finais da otimização de portfólio.
    
    Parâmetros:
        best_weights (np.ndarray): Pesos do melhor portfólio.
        test_data (pd.DataFrame): Dados de teste.
        test_cov_matrix (pd.DataFrame): Matriz de covariância dos retornos de teste.
        risk_free_rate (float): Taxa livre de risco.
        investment (float): Valor do investimento.
        selected_tickers (list): Lista de códigos das ações selecionadas.
        data (pd.DataFrame): DataFrame com os preços ajustados das ações.
        train_cov_matrix (pd.DataFrame): Matriz de covariância dos retornos de treinamento.
        returns (pd.DataFrame): DataFrame com os retornos diários.
        evaluation_method (str): Método de avaliação.
        pareto_front_history (list): Lista com o histórico do Pareto Front por geração.
        best_history (list): Lista com o histórico do melhor índice por geração.
        benchmark_returns (float): Retorno do benchmark.
    """
    # Avaliar no conjunto de teste
    test_return, test_vol, test_sharpe = calculate_metrics(best_weights, test_data, test_cov_matrix, risk_free_rate)
    
    # Seção: Desempenho no Conjunto de Teste
    st.header("📊 Desempenho no Conjunto de Teste")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **Retorno:** {test_return:.2%}")
        st.markdown(f"- **Volatilidade:** {test_vol:.2%}")
        st.markdown(f"- **Índice {evaluation_method}:** {test_sharpe:.4f}")
    with col2:
        st.subheader("Gráfico de Alocação (Teste)")
        plot_interactive_allocation(best_weights, selected_tickers)

    # Seção: Portfólio Final
    st.header("🎯 Portfólio Final")
    final_return, final_vol, final_sharpe = calculate_metrics(best_weights, returns, train_cov_matrix, risk_free_rate)
    display_portfolio_metrics(final_return, final_vol, final_sharpe, investment, best_weights, selected_tickers, data, evaluation_method)

    # Seção: Gráficos Finais
    st.header("📈 Gráficos Finais")
    display_final_charts(pareto_front_history=pareto_front_history, best_history=best_history, evaluation_method=evaluation_method)

    # Exportação de Resultados
    st.header("📤 Exportar Resultados")
    latest_prices = data.iloc[-1]
    allocation_df = pd.DataFrame({
        'Ação': selected_tickers,
        'Peso (%)': [f"{weight * 100:.2f}" for weight in best_weights],
        'Valor Alocado ($)': [f"{weight * investment:,.2f}" for weight in best_weights],
        'Quantidade de Ações': [(weight * investment / latest_prices[ticker]).round(2) for weight, ticker in zip(best_weights, selected_tickers)]
    })
    export_results(allocation_df, benchmark_returns)

def display_portfolio_metrics(final_return, final_vol, final_sharpe, investment, best_weights, selected_tickers, data, evaluation_method):
    """
    Exibe métricas e alocação do portfólio final.
    
    Parâmetros:
        final_return (float): Retorno anualizado do portfólio final.
        final_vol (float): Volatilidade anualizada do portfólio final.
        final_sharpe (float): Índice de Sharpe do portfólio final.
        investment (float): Valor do investimento.
        best_weights (np.ndarray): Pesos do melhor portfólio.
        selected_tickers (list): Lista de códigos das ações selecionadas.
        data (pd.DataFrame): DataFrame com os preços ajustados das ações.
        evaluation_method (str): Método de avaliação.
    """
    col1, col2 = st.columns(2)
    monthly_return = (1 + final_return) ** (1/12) - 1
    projected_values = {
        "1 mês": investment * (1 + monthly_return),
        "3 meses": investment * (1 + monthly_return) ** 3,
        "6 meses": investment * (1 + monthly_return) ** 6,
        "1 ano": investment * (1 + final_return),
        "2 anos": investment * (1 + final_return) ** 2,
        "5 anos": investment * (1 + final_return) ** 5
    }
    
    # Coluna 1: Métricas do Portfólio
    with col1:
        st.subheader("💼 Alocação do Investimento")
        latest_prices = data.iloc[-1]
        allocation_df = pd.DataFrame({
            'Ação': selected_tickers,
            'Peso (%)': [f"{weight * 100:.2f}" for weight in best_weights],
            'Valor Alocado ($)': [f"{weight * investment:,.2f}" for weight in best_weights],
            'Quantidade de Ações': [(weight * investment / latest_prices[ticker]).round(2) for weight, ticker in zip(best_weights, selected_tickers)]
        })
        
        # Exibir tabela de alocação
        st.dataframe(allocation_df)
        
    # Coluna 2: Alocação do Investimento
    with col2:
        # Exibir gráfico de pizza para alocação
        st.markdown("### Gráfico de Alocação")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%', startangle=90)
        ax.set_title("Distribuição do Portfólio")
        st.pyplot(fig)
    
    # Após a otimização
    highlight_best_portfolio(best_weights, selected_tickers, investment)
    
    # Exibir resultados finais
    st.header("📈 Resultados Finais")
    display_summary(final_return, final_vol, final_sharpe, investment, projected_values, evaluation_method)

def display_final_charts(pareto_front_history, best_history, evaluation_method):
    """
    Exibe gráficos finais: Pareto Front e evolução do índice.
    
    Parâmetros:
        pareto_front_history (list): Lista com o histórico do Pareto Front por geração.
        best_history (list): Lista com o histórico do melhor índice por geração.
        evaluation_method (str): Método de avaliação.
    """
    col1, col2 = st.columns(2)

    # Gráfico do Pareto Front
    with col1:
        st.subheader("Evolução do Pareto Front")
        fig, ax = plt.subplots(figsize=(6, 4))
        for generation, pareto_front in enumerate(pareto_front_history):
            returns = [score[0] for _, score in pareto_front]
            risks = [score[1] for _, score in pareto_front]
            ax.scatter(risks, returns, label=f'Geração {generation + 1}', alpha=0.6)
        ax.set_xlabel('Risco (Volatilidade)')
        ax.set_ylabel('Retorno')
        ax.set_title('Evolução do Pareto Front')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    # Gráfico de evolução do índice
    with col2:
        st.subheader("Evolução do Melhor Índice")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(best_history, 'b-')
        ax.set_xlabel('Geração')
        ax.set_ylabel(f'Melhor Índice ({evaluation_method})')
        ax.set_title('Progresso da Otimização')
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

def highlight_best_portfolio(best_weights, selected_tickers, investment):
    """
    Destaca o melhor portfólio encontrado.
    
    Parâmetros:
        best_weights (np.ndarray): Pesos do melhor portfólio.
        selected_tickers (list): Lista de códigos das ações selecionadas.
        investment (float): Valor do investimento.
    """
    st.markdown("### 🏆 Melhor Portfólio")
    for ticker, weight in zip(selected_tickers, best_weights):
        st.markdown(f"- **{ticker}**: {weight:.2%} (${weight * investment:,.2f})")

def plot_interactive_allocation(best_weights, selected_tickers):
    """
    Gera um gráfico interativo de alocação do portfólio.
    
    Parâmetros:
        best_weights (np.ndarray): Pesos do melhor portfólio.
        selected_tickers (list): Lista de códigos das ações selecionadas.
    """
    df = pd.DataFrame({
        'Ação': selected_tickers,
        'Peso': [w * 100 for w in best_weights]
    })
    fig = px.pie(df, values='Peso', names='Ação', title='Alocação do Portfólio')
    st.plotly_chart(fig)

def display_summary(final_return, final_vol, final_sharpe, investment, projected_values, evaluation_method):
    """
    Exibe um resumo das métricas e projeções do portfólio.
    
    Parâmetros:
        final_return (float): Retorno anualizado do portfólio final.
        final_vol (float): Volatilidade anualizada do portfólio final.
        final_sharpe (float): Índice de Sharpe do portfólio final.
        investment (float): Valor do investimento.
        projected_values (dict): Dicionário com projeções de valores.
        evaluation_method (str): Método de avaliação.
    """
    st.subheader("📊 Métricas do Portfólio")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Retorno Anualizado:** {final_return:.2%}")
        st.markdown(f"**Volatilidade Anualizada:** {final_vol:.2%}")
        st.markdown(f"**Índice {evaluation_method}:** {final_sharpe:.4f}")
    
    with col2:
        st.markdown("**Projeção de Valor do Investimento:**")
        for period, value in projected_values.items():
            st.markdown(f"- **{period}:** ${value:,.2f}")

def plot_correlation_matrix(train_cov_matrix):
    """
    Exibe a matriz de correlação das ações.
    
    Parâmetros:
        train_cov_matrix (pd.DataFrame): Matriz de covariância dos retornos de treinamento.
    """
    correlation_matrix = train_cov_matrix.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Matriz de Correlação das Ações')
    st.pyplot(fig)

def generate_unique_key(prefix="key"):
    """
    Gera uma chave única para elementos Streamlit que precisam ser distintos.
    
    Parâmetros:
        prefix (str): Prefixo para a chave.
        
    Retorna:
        str: Chave única.
    """
    return f"{prefix}_{uuid.uuid4().hex}"

def export_results(allocation_df, benchmark_returns=None):
    """
    Exibe opções para exportar os resultados da otimização.
    
    Parâmetros:
        allocation_df (pd.DataFrame): DataFrame com a alocação do portfólio.
        benchmark_returns (float, opcional): Retorno do benchmark.
    """
    # Exibir o DataFrame de alocação
    st.subheader("Resumo de Alocação")
    st.dataframe(allocation_df)
    
    # Botão para baixar como CSV
    csv = allocation_df.to_csv(index=False)
    st.download_button(
        label="Baixar como CSV",
        data=csv,
        file_name="portfolio_allocation.csv",
        mime="text/csv",
        key=generate_unique_key()
    )
    
    # Adicionar comparação com benchmark se disponível
    if benchmark_returns is not None:
        st.markdown(f"**Retorno do Benchmark:** {benchmark_returns:.2%}") 