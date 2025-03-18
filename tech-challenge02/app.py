import streamlit as st
import pandas as pd
from genetic_algorithm import optimize_portfolio

# Configurações da página
st.set_page_config(
    page_title="Otimizador de Portfólio",
    page_icon="📈",
    layout="wide"
)

# Título e descrição
st.title("📊 Otimizador de Portfólio com Algoritmo Genético")
st.markdown("""
Este aplicativo usa um algoritmo genético para encontrar a alocação ideal de portfólio baseada no Índice Sharpe.
Você pode selecionar ações, definir seu valor de investimento e acompanhar o processo de otimização em tempo real!
""")

# Sidebar
with st.sidebar:
    st.sidebar.markdown("""
    ### Tutorial
    1. Selecione as ações e o período de análise.
    2. Configure os parâmetros do algoritmo genético.
    3. Clique em "Otimizar Portfólio" para iniciar.
    """)
    st.header("📝 Configurações Gerais")
    with st.expander("📅 Parâmetros Gerais"):
        investment = st.number_input("Valor do Investimento ($)", min_value=1000, max_value=10000000, value=10000, step=1000)
        start_date = st.date_input("Data Inicial", value=pd.Timestamp("2020-01-01"))
        end_date = st.date_input("Data Final", value=pd.Timestamp("2023-01-01"))
        default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        custom_tickers = st.text_input("Digite códigos de ações adicionais (separados por vírgula)", "META, NVDA").replace(" ", "")
        all_tickers = default_tickers + [t.strip() for t in custom_tickers.split(",") if t.strip()]
        selected_tickers = st.multiselect("Selecione ações para seu portfólio", all_tickers, default=default_tickers[:3])

    with st.expander("⚙️ Parâmetros do Algoritmo Genético"):
        population_size = st.slider("Tamanho da População", 50, 200, 100)
        num_generations = st.slider("Número de Gerações", 10, 100, 50)
        mutation_rate = st.slider("Taxa de Mutação", 0.0, 0.5, 0.1)
        risk_free_rate = st.slider("Taxa Livre de Risco (%)", 0.0, 5.0, 2.0) / 100

    with st.expander("🔧 Configurações Avançadas"):
        multiobjective = st.checkbox("Ativar Multiobjetivo (Retorno e Risco)", value=False)
        init_strategy = st.selectbox(
            "Estratégia de Inicialização",
            options=["random", "uniform", "return_based", "volatility_inverse"],
            index=0
        )
        selection_method = st.selectbox(
            "Método de Seleção",
            options=["tournament", "roulette", "elitism"],
            index=0
        )
        evaluation_method = st.selectbox(
            "Método de Avaliação",
            options=["sharpe", "sortino", "treynor", "var"],
            index=0
        )
        crossover_method = st.selectbox(
            "Método de Crossover",
            options=["uniform", "single_point", "arithmetic"],
            index=0
        )
        mutation_distribution = st.selectbox(
            "Distribuição de Mutação",
            options=["normal", "uniform"],
            index=0
        )
        min_weight = st.slider("Peso Mínimo (%)", 0, 20, 5) / 100
        max_weight = st.slider("Peso Máximo (%)", 20, 100, 50) / 100
        elitism_count = st.slider("Número de Indivíduos Elitistas", 1, 10, 1)

    st.markdown("### ℹ️ Ajuda")
    st.markdown("""
    - **População**: Número de indivíduos na população.
    - **Gerações**: Número de iterações do algoritmo.
    - **Taxa de Mutação**: Probabilidade de mutação em cada indivíduo.
    - **Taxa Livre de Risco**: Taxa de retorno sem risco usada no cálculo do Sharpe.
    """)

# Botão para rodar a otimização
if st.button("🚀 Otimizar Portfólio"):
    optimize_portfolio(
        selected_tickers,
        start_date,
        end_date,
        investment,
        population_size,
        num_generations,
        mutation_rate,
        risk_free_rate,
        min_weight,
        max_weight,
        init_strategy=init_strategy,
        evaluation_method=evaluation_method,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_distribution=mutation_distribution,
        elitism_count=elitism_count,
        multiobjective=multiobjective
    )