import streamlit as st
import pandas as pd
import plotly.express as px
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

# Divisão em abas
tab1, tab2 = st.tabs(["📈 Portfólio", "📊 Benchmark"])

# Aba de Portfólio
with tab1:
    st.header("🚀 Otimização de Portfólio")
    if st.button("Otimizar Portfólio"):
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

# Aba de Benchmark
with tab2:
    st.header("📊 Benchmark de Configurações")
    st.markdown("""
    ### Como Usar o Benchmark
    1. Insira diferentes configurações no formato JSON.
    2. Clique em "Executar Benchmark" para comparar os resultados.
    3. Cada aba mostrará os resultados de uma configuração específica.
    """)

    # Input for multiple configurations
    benchmark_configs = st.text_area(
        "Defina as configurações (JSON format)",
        value="""
            [
                {
                    "population_size": 100,
                    "num_generations": 50,
                    "mutation_rate": 0.1,
                    "evaluation_method": "sharpe",
                    "init_strategy": "return_based",
                    "selection_method": "tournament",
                    "crossover_method": "single_point",
                    "mutation_distribution": "normal",
                    "elitism_count": 2,
                    "multiobjective": false
                },
                {
                    "population_size": 150,
                    "num_generations": 75,
                    "mutation_rate": 0.2,
                    "evaluation_method": "sharpe",
                    "init_strategy": "volatility_inverse",
                    "selection_method": "elitism",
                    "crossover_method": "arithmetic",
                    "mutation_distribution": "uniform",
                    "elitism_count": 3,
                    "multiobjective": false
                },
                {
                    "population_size": 120,
                    "num_generations": 60,
                    "mutation_rate": 0.15,
                    "evaluation_method": "sharpe",
                    "init_strategy": "random",
                    "selection_method": "roulette",
                    "crossover_method": "uniform",
                    "mutation_distribution": "normal",
                    "elitism_count": 1,
                    "multiobjective": false
                }
            ]
        """
    )

    # Parse configurations
    try:
        configs = pd.read_json(benchmark_configs)
    except ValueError:
        st.error("Erro ao processar as configurações. Certifique-se de que o JSON está correto.")
        configs = None

    if configs is not None:
        for i, config in configs.iterrows():
            if not (50 <= config["population_size"] <= 200):
                st.error(f"Configuração {i+1}: 'population_size' deve estar entre 50 e 200.")
            if not (10 <= config["num_generations"] <= 100):
                st.error(f"Configuração {i+1}: 'num_generations' deve estar entre 10 e 100.")

    if st.button("Executar Benchmark") and configs is not None:
        progress_bar = st.progress(0)
        results = []
        tabs = st.tabs([f"Configuração {i+1}" for i in range(len(configs))])  # Criar abas para cada configuração
    
        for i, (tab, config) in enumerate(zip(tabs, configs.iterrows())):
            with tab:
                st.subheader(f"Configuração {i+1}")
                st.json(config[1].to_dict())  # Exibir as configurações escolhidas
    
                result = optimize_portfolio(
                    selected_tickers=selected_tickers,
                    start_date=start_date,
                    end_date=end_date,
                    investment=investment,
                    population_size=config[1]["population_size"],
                    num_generations=config[1]["num_generations"],
                    mutation_rate=config[1]["mutation_rate"],
                    risk_free_rate=risk_free_rate,
                    min_weight=min_weight,
                    max_weight=max_weight,
                    init_strategy=config[1].get("init_strategy", "random"),  # Padrão: "random"
                    evaluation_method=config[1]["evaluation_method"],
                    selection_method=config[1].get("selection_method", "tournament"),  # Padrão: "tournament"
                    crossover_method=config[1].get("crossover_method", "uniform"),  # Padrão: "uniform"
                    mutation_distribution=config[1].get("mutation_distribution", "normal"),  # Padrão: "normal"
                    elitism_count=config[1].get("elitism_count", 1),  # Padrão: 1
                    multiobjective=config[1].get("multiobjective", False)
                )
                results.append(result)
                st.write("Resultados:", result)  # Exibir os resultados da configuração atual
                progress_bar.progress((i + 1) / len(configs))
    
        st.success("Benchmark concluído!")
    
        if results:
            st.subheader("Resumo Geral do Benchmark")
            summary_df = pd.DataFrame(results)
            
            # Adicionar uma coluna de identificação para as configurações
            summary_df["Configuração"] = [f"Configuração {i+1}" for i in range(len(summary_df))]
            
            st.dataframe(summary_df)
            
            # Criar o gráfico de comparação
            st.subheader("Gráfico de Comparação de Configurações")
            fig = px.bar(
                summary_df,
                x="Configuração",
                y="best_sharpe",  # Substitua por qualquer métrica que você queira comparar
                title="Comparação de Configurações",
                labels={"best_sharpe": "Índice Sharpe"}
            )
            st.plotly_chart(fig)