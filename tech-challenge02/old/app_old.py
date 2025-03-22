import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page config
st.set_page_config(
    page_title="Otimizador de Portfólio",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📊 Otimizador de Portfólio com Algoritmo Genético")
st.markdown("""
Este aplicativo usa um algoritmo genético para encontrar a alocação ideal de portfólio baseada no Índice Sharpe.
Você pode selecionar ações, definir seu valor de investimento e acompanhar o processo de otimização em tempo real!
""")

# Sidebar
with st.sidebar:
    st.header("📝 Parâmetros")
    
    # Investment amount
    investment = st.number_input(
        "Valor do Investimento ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )
    
    # Date range
    st.subheader("Período")
    start_date = st.date_input(
        "Data Inicial",
        value=pd.Timestamp("2020-01-01")
    )
    end_date = st.date_input(
        "Data Final",
        value=pd.Timestamp("2023-01-01")
    )
    
    # Stock selection
    st.subheader("Seleção de Ações")
    default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    custom_tickers = st.text_input(
        "Digite códigos de ações adicionais (separados por vírgula)",
        "META, NVDA"
    ).replace(" ", "")
    
    all_tickers = default_tickers + [t.strip() for t in custom_tickers.split(",") if t.strip()]
    selected_tickers = st.multiselect(
        "Selecione ações para seu portfólio",
        all_tickers,
        default=default_tickers[:3]
    )
    
    # Algorithm parameters
    st.subheader("Parâmetros do Algoritmo")
    population_size = st.slider("Tamanho da População", 50, 200, 100)
    num_generations = st.slider("Número de Gerações", 10, 100, 50)
    mutation_rate = st.slider("Taxa de Mutação", 0.0, 0.5, 0.1)
    risk_free_rate = st.slider("Taxa Livre de Risco (%)", 0.0, 5.0, 2.0) / 100
    # Restrições de peso
    min_weight = st.sidebar.slider("Peso Mínimo (%)", 0, 20, 5) / 100
    max_weight = st.sidebar.slider("Peso Máximo (%)", 20, 100, 50) / 100

def download_data(tickers, start_date, end_date):
    """Baixar dados das ações e tratar valores ausentes"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        data = data.dropna(how='all')  # Remover colunas com todos os valores ausentes
        return data
    except Exception as e:
        st.error(f"Erro ao baixar dados: {str(e)}")
        return None

def calculate_metrics(weights, returns, cov_matrix, risk_free_rate):
    """Calcular métricas do portfólio"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe_ratio

def create_individual(size, strategy="random"):
    """Criar pesos para o portfólio com diferentes estratégias"""
    if strategy == "random":
        weights = np.random.random(size)
    elif strategy == "uniform":
        weights = np.ones(size) / size
    else:
        raise ValueError("Estratégia desconhecida para inicialização.")
    return weights / np.sum(weights)

def calculate_diversification(cov_matrix):
    """Calcular a diversificação média do portfólio"""
    correlations = cov_matrix.corr()
    avg_correlation = correlations.mean().mean()
    return avg_correlation

def optimize_portfolio():
    if len(selected_tickers) < 2:
        st.warning("Por favor, selecione pelo menos 2 ações.")
        return
    
    # Download data
    with st.spinner("Baixando dados das ações..."):
        data = download_data(selected_tickers, start_date, end_date)
        if data is None:
            return
    
    # Calculate returns and covariance
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    
    # Display current stock prices
    latest_prices = data.iloc[-1]
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Preços Atuais das Ações")
        price_df = pd.DataFrame({
            'Ação': latest_prices.index,
            'Preço': latest_prices.values
        })
        st.dataframe(price_df.style.format({'Preço': '${:.2f}'}))
        # Exibir diversificação
        avg_correlation = calculate_diversification(cov_matrix)
        st.markdown(f"**Correlação Média do Portfólio:** {avg_correlation:.2f}")
    with col2:
        st.subheader("Retorno das Ações")
        returns_chart = st.empty()
        fig_returns, ax = plt.subplots(figsize=(10, 6))
        (returns + 1).cumprod().plot(ax=ax)
        ax.set_title("Retornos Acumulados")
        ax.grid(True)
        returns_chart.pyplot(fig_returns)
        plt.close(fig_returns)
    
    # Setup progress displays
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_text = st.empty()
    
    # Charts
    col3, col4 = st.columns(2)
    with col3:
        progress_chart = st.empty()
    with col4:
        allocation_chart = st.empty()
    
    # Initialize population
    population = [create_individual(len(selected_tickers), strategy="random") for _ in range(int(population_size * 0.8))]
    population += [create_individual(len(selected_tickers), strategy="uniform") for _ in range(int(population_size * 0.2))]

    best_sharpe = float('-inf')
    best_weights = None
    best_history = []
    
    # Optimization
    fig_progress, ax_progress = plt.subplots(figsize=(10, 6))
    fig_allocation, ax_allocation = plt.subplots(figsize=(10, 6))
    
    try:
        for generation in range(num_generations):
            # Evaluate fitness
            fitness_scores = []
            for weights in population:
                portfolio_return, portfolio_vol, sharpe_ratio = calculate_metrics(
                    weights, returns, cov_matrix, risk_free_rate
                )
                fitness_scores.append(sharpe_ratio)
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_sharpe:
                best_sharpe = fitness_scores[max_idx]
                best_weights = population[max_idx].copy()
            best_history.append(best_sharpe)
            
            # Update progress
            progress = (generation + 1) / num_generations
            progress_bar.progress(progress)
            status_text.text(f"Geração {generation + 1}/{num_generations}")
            
            # Update metrics
            if best_weights is not None:
                ret, vol, _ = calculate_metrics(best_weights, returns, cov_matrix, risk_free_rate)
                metrics_text.markdown(f"""
                **Melhor Portfólio Atual:**
                - Retorno Anual Esperado: {ret:.2%}
                - Volatilidade Anual Esperada: {vol:.2%}
                - Índice Sharpe: {best_sharpe:.4f}
                """)
            
            # Update charts
            if generation % 2 == 0:
                # Progress chart
                ax_progress.clear()
                ax_progress.plot(best_history, 'b-')
                ax_progress.set_xlabel('Geração')
                ax_progress.set_ylabel('Melhor Índice Sharpe')
                ax_progress.set_title('Progresso da Otimização')
                ax_progress.grid(True)
                # Adicionar linha de referência para o melhor índice Sharpe
                ax_progress.axhline(y=best_sharpe, color='r', linestyle='--', label='Melhor Sharpe')
                ax_progress.legend()
                progress_chart.pyplot(fig_progress)
                
                # Allocation chart
                ax_allocation.clear()
                if best_weights is not None:
                    ax_allocation.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%')
                    ax_allocation.set_title('Melhor Alocação de Portfólio Atual')
                allocation_chart.pyplot(fig_allocation)
            
            # Selection
            new_population = []
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 3
                tournament = np.random.choice(population_size, tournament_size)
                parent1 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
                tournament = np.random.choice(population_size, tournament_size)
                parent2 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
                
                # Cruzamento uniforme
                if np.random.random() < 0.8:
                    mask = np.random.randint(0, 2, len(selected_tickers))
                    child = mask * parent1 + (1 - mask) * parent2
                    child = child / np.sum(child)
                else:
                    child = parent1.copy()
                
                # Mutation
                mutation_intensity = 0.1 * (1 - generation / num_generations)  # Decresce ao longo das gerações
                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, mutation_intensity, len(selected_tickers))
                    child = child + mutation
                    child = np.clip(child, min_weight, max_weight)
                    child = child / np.sum(child)
                
                new_population.append(child)
            
            population = new_population
    
    except Exception as e:
        st.error(f"Erro de otimização: {str(e)}")
        return
    finally:
        plt.close(fig_progress)
        plt.close(fig_allocation)
    
    # Final Results
    st.header("🎯 Portfólio Final")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Métricas do Portfólio")
        final_return, final_vol, final_sharpe = calculate_metrics(
            best_weights, returns, cov_matrix, risk_free_rate
        )
        
        # Calculate projected returns for different time horizons
        monthly_return = (1 + final_return) ** (1/12) - 1
        projected_values = {
            "1 mês": investment * (1 + monthly_return),
            "3 meses": investment * (1 + monthly_return) ** 3,
            "6 meses": investment * (1 + monthly_return) ** 6,
            "1 ano": investment * (1 + final_return),
            "2 anos": investment * (1 + final_return) ** 2,
            "5 anos": investment * (1 + final_return) ** 5
        }
        
        st.markdown(f"""
        #### Métricas Atuais
        - Retorno Anual Esperado: **{final_return:.2%}**
        - Volatilidade Anual Esperada: **{final_vol:.2%}**
        - Índice Sharpe: **{final_sharpe:.4f}**
        
        #### Valor Projetado do Investimento
        Assumindo que a taxa de retorno esperada permaneça constante:
        """)
        
        for period, value in projected_values.items():
            profit = value - investment
            profit_percent = (profit / investment) * 100
            st.markdown(f"""
            **{period}:**
            - Valor: **${value:,.2f}**
            - Lucro: **${profit:,.2f}** (*{profit_percent:+.1f}%*)
            """)
            
        st.warning("""
        ⚠️ Nota: Estas projeções são baseadas em dados históricos e retornos esperados. 
        Os retornos reais podem variar devido às condições do mercado e outros fatores.
        O desempenho passado não garante resultados futuros.
        """)
    
    with col6:
        st.subheader("Alocação do Investimento")
        allocation_df = pd.DataFrame({
            'Ação': selected_tickers,
            'Peso': best_weights,
            'Valor': best_weights * investment,
            'Ações': (best_weights * investment / latest_prices).round(2)
        })
        st.dataframe(
            allocation_df.style.format({
                'Peso': '{:.2%}',
                'Valor': '${:,.2f}',
                'Ações': '{:.2f}'
            })
        )

# Run button
if st.button("🚀 Otimizar Portfólio"):
    optimize_portfolio()
