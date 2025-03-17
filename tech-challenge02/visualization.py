import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
from metrics import calculate_metrics, calculate_diversification, calculate_sortino_ratio, calculate_treynor_ratio, calculate_var

def display_stock_prices(data):
    """Exibir preços das ações"""
    st.subheader("Preços Atuais das Ações")
    latest_prices = data.iloc[-1]
    price_df = pd.DataFrame({
        'Ação': latest_prices.index,
        'Preço': latest_prices.values
    })
    st.dataframe(price_df.style.format({'Preço': '${:.2f}'}))

def display_progress_chart(best_history):
    """Exibir gráfico de progresso"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_history, 'b-')
    ax.set_xlabel('Geração')
    ax.set_ylabel('Melhor Índice Sharpe')
    ax.set_title('Progresso da Otimização')
    ax.grid(True)
    st.pyplot(fig)

def display_allocation_chart(best_weights, selected_tickers):
    """Exibir gráfico de alocação"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%')
    ax.set_title('Melhor Alocação de Portfólio Atual')
    st.pyplot(fig)

def plot_pareto_front(pareto_front_history):
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

# Funções auxiliares para simplificação
def display_initial_data(data, train_cov_matrix):
    """Exibir preços e diversificação inicial."""
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
    """Atualizar a exibição de progresso e métricas."""
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
    """Atualizar gráfico de progresso."""
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
    """Atualizar gráfico de alocação."""
    if best_weights is not None:
        with allocation_chart.container():
            st.subheader("Alocação do Portfólio")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%')
            ax.set_title('Melhor Alocação de Portfólio Atual')
            st.pyplot(fig)
            plt.close(fig)


def display_final_results(best_weights, test_data, test_cov_matrix, risk_free_rate, investment, selected_tickers, data, train_cov_matrix, returns, evaluation_method):
    """Exibir os resultados finais do portfólio."""
    # Avaliar no conjunto de teste
    test_return, test_vol, test_sharpe = calculate_metrics(best_weights, test_data, test_cov_matrix, risk_free_rate)
    st.markdown(f"**Desempenho no Conjunto de Teste:**")
    st.markdown(f"- Retorno: {test_return:.2%}")
    st.markdown(f"- Volatilidade: {test_vol:.2%}")
    st.markdown(f"- Índice {evaluation_method}: {test_sharpe:.4f}")

    # Exibir métricas finais
    st.header("🎯 Portfólio Final")
    final_return, final_vol, final_sharpe = calculate_metrics(best_weights, returns, train_cov_matrix, risk_free_rate)
    display_portfolio_metrics(final_return, final_vol, final_sharpe, investment, best_weights, selected_tickers, data)



def display_portfolio_metrics(final_return, final_vol, final_sharpe, investment, best_weights, selected_tickers, data):
    """Exibir métricas e alocação do portfólio final."""
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
    
    # Exibir resultados finais
    st.header("📈 Resultados Finais")
    display_summary(final_return, final_vol, final_sharpe, investment, projected_values)
    
    # Após a otimização
    highlight_best_portfolio(best_weights, selected_tickers, investment)
    
    # plot_interactive_allocation(best_weights, selected_tickers)

    # benchmark_history = [benchmark_returns] * len(best_history)  # Exemplo de benchmark constante
    # plot_with_benchmark(best_history, benchmark_history, evaluation_method)

def plot_pareto_front(pareto_front_history):
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

def plot_best_index_evolution(best_history, evaluation_method):
    st.subheader("Evolução do Melhor Índice")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_history, 'b-')
    ax.set_xlabel('Geração')
    ax.set_ylabel(f'Melhor Índice ({evaluation_method})')
    ax.set_title('Progresso da Otimização')
    ax.grid(True)
    st.pyplot(fig)

def display_final_charts(pareto_front_history, best_history, evaluation_method):
    """Exibir gráficos finais: Pareto Front e evolução do índice."""
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
    
def update_real_time_charts(pareto_front_history, best_history, evaluation_method, pareto_placeholder, index_placeholder):
    """Atualizar gráficos em tempo real."""
    # Atualizar Pareto Front
    with pareto_placeholder.container():
        st.subheader("Evolução do Pareto Front (Tempo Real)")
        fig_pareto, ax_pareto = plt.subplots(figsize=(6, 4))
        if pareto_front_history:  # Verifica se há dados no histórico
            # Apenas plota o Pareto Front da última geração
            pareto_front = pareto_front_history[-1]
            returns = [score[0] for _, score in pareto_front]
            risks = [score[1] for _, score in pareto_front]
            ax_pareto.scatter(risks, returns, label=f'Última Geração', alpha=0.6)
            ax_pareto.set_xlabel('Risco (Volatilidade)')
            ax_pareto.set_ylabel('Retorno')
            ax_pareto.set_title('Evolução do Pareto Front')
            ax_pareto.legend()
            ax_pareto.grid(True)
        st.pyplot(fig_pareto)
        plt.close(fig_pareto)

    # Atualizar Melhor Índice
    with index_placeholder.container():
        st.subheader("Progresso do Melhor Índice (Tempo Real)")
        fig_index, ax_index = plt.subplots(figsize=(6, 4))
        if best_history:  # Verifica se há dados no histórico
            ax_index.plot(best_history, 'b-')
            ax_index.set_xlabel('Geração')
            ax_index.set_ylabel(f'Melhor Índice ({evaluation_method})')
            ax_index.set_title('Progresso da Otimização')
            ax_index.grid(True)
        st.pyplot(fig_index)
        plt.close(fig_index)

def highlight_best_portfolio(best_weights, selected_tickers, investment):
    """Destacar o melhor portfólio encontrado."""
    st.markdown("### 🎯 Melhor Portfólio Encontrado")
    allocation_df = pd.DataFrame({
        'Ação': selected_tickers,
        'Peso (%)': [f"{weight * 100:.2f}" for weight in best_weights],
        'Valor Alocado ($)': [f"{weight * investment:,.2f}" for weight in best_weights]
    })
    st.dataframe(allocation_df.style.highlight_max(axis=0, color='lightgreen'))


def plot_interactive_pareto_front(pareto_front_history):
    generations = []
    returns = []
    risks = []

    for generation, pareto_front in enumerate(pareto_front_history):
        for ret, risk in pareto_front:
            generations.append(generation + 1)
            returns.append(ret)
            risks.append(risk)

    df = pd.DataFrame({'Geração': generations, 'Retorno': returns, 'Risco': risks})
    fig = px.scatter(df, x='Risco', y='Retorno', color='Geração', title="Evolução do Pareto Front")
    st.plotly_chart(fig)

def plot_with_benchmark(best_history, benchmark_history, evaluation_method):
    """Exibir progresso com benchmark."""
    st.subheader("Progresso do Melhor Índice com Benchmark")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(best_history, label='Melhor Índice', color='blue')
    ax.plot(benchmark_history, label='Benchmark', color='orange', linestyle='--')
    ax.set_xlabel('Geração')
    ax.set_ylabel(f'Índice ({evaluation_method})')
    ax.set_title('Progresso da Otimização com Benchmark')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def display_pareto_summary(pareto_front_history, best_history):
    """Exibir resumo final."""
    st.markdown("### 📊 Resumo Final")
    best_portfolios = []
    
    for generation, pareto_front in enumerate(pareto_front_history):
        # Verificar se o índice da geração existe no best_history
        if generation < len(best_history):
            best_metric = best_history[generation]  # Melhor índice da geração
        else:
            best_metric = None  # Caso não exista, deixar como None
        
        # Encontrar o portfólio correspondente ao melhor índice
        best_portfolio = max(pareto_front, key=lambda x: x[1][0])  # Maior retorno
        best_portfolios.append({
            'Geração': generation + 1,
            'Retorno': best_portfolio[1][0],
            'Risco': best_portfolio[1][1],
            'Melhor Índice': best_metric
        })

    # Criar DataFrame para exibição
    summary_df = pd.DataFrame(best_portfolios)
    st.dataframe(summary_df.style.format({
        'Retorno': '{:.2%}',
        'Risco': '{:.2%}',
        'Melhor Índice': '{:.4f}'  # Formatar o índice com 4 casas decimais
    }))

def plot_interactive_allocation(best_weights, selected_tickers):
    """Exibir gráfico interativo de alocação."""
    allocation_df = pd.DataFrame({
        'Ação': selected_tickers,
        'Peso (%)': [weight * 100 for weight in best_weights]
    })
    fig = px.pie(allocation_df, values='Peso (%)', names='Ação', title='Distribuição do Portfólio')
    st.plotly_chart(fig)

def display_summary(final_return, final_vol, final_sharpe, investment, projected_values):
    """Exibir resumo consolidado."""
    st.markdown("### 📊 Resumo Consolidado")
    st.markdown(f"""
    - **Retorno Anual Esperado:** {final_return:.2%}
    - **Volatilidade Anual Esperada:** {final_vol:.2%}
    - **Índice Sharpe:** {final_sharpe:.4f}
    """)
    st.markdown("### 📈 Projeções de Investimento")
    projections_df = pd.DataFrame({
        "Período": list(projected_values.keys()),
        "Valor Projetado ($)": [f"{value:,.2f}" for value in projected_values.values()],
        "Lucro ($)": [f"{(value - investment):,.2f}" for value in projected_values.values()],
        "Lucro (%)": [f"{((value - investment) / investment) * 100:+.1f}%" for value in projected_values.values()]
    })
    st.dataframe(projections_df)

def plot_correlation_matrix(train_cov_matrix):
    """Exibir matriz de correlação."""
    st.subheader("Matriz de Correlação dos Ativos")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(train_cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def plot_cumulative_returns(data):
    """Exibir retornos acumulados."""
    st.subheader("Retornos Acumulados dos Ativos")
    fig, ax = plt.subplots(figsize=(10, 6))
    (data + 1).cumprod().plot(ax=ax)
    ax.set_title("Retornos Acumulados")
    ax.grid(True)
    st.pyplot(fig)

def export_results(allocation_df):
    """Permitir exportação dos resultados."""
    csv = allocation_df.to_csv(index=False)
    st.download_button(
        label="📥 Baixar Resultados",
        data=csv,
        file_name='resultados_portfolio.csv',
        mime='text/csv',
    )