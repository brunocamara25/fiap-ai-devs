"""
Aplicativo principal para otimização de portfólio com algoritmos genéticos.

Este módulo implementa uma interface Streamlit para interação com o algoritmo genético
de otimização de portfólio, permitindo configurar parâmetros, visualizar resultados
e exportar as análises.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import io
import base64
import json

# Importações do projeto
from config import (
    DEFAULT_POPULATION_SIZE, DEFAULT_NUM_GENERATIONS, DEFAULT_MUTATION_RATE,
    DEFAULT_RISK_FREE_RATE, DEFAULT_MIN_WEIGHT, DEFAULT_MAX_WEIGHT,
    DEFAULT_EVALUATION_METHODS, DEFAULT_INIT_STRATEGIES,
    DEFAULT_SELECTION_METHODS, DEFAULT_CROSSOVER_METHODS,
    DEFAULT_MUTATION_DISTRIBUTIONS, DEFAULT_ELITISM_COUNT,
    POPULAR_BR_TICKERS, POPULAR_US_TICKERS,
    DEFAULT_BENCHMARK, DEFAULT_START_DATE, DEFAULT_END_DATE,
    STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT
)
from src.data.loader import download_data, get_risk_free_rate
from src.data.processor import prepare_returns as calculate_returns, calculate_cov_matrix as calculate_covariance_matrix
from src.models.genetic_algorithm import GeneticAlgorithm, optimize_portfolio
from src.optimization.portfolio import Portfolio
from src.visualization.plots import (
    plot_correlation_matrix, plot_efficient_frontier, plot_portfolio_allocation,
    plot_cumulative_returns, plot_drawdowns, plot_ga_evolution, plot_pareto_front,
    create_dashboard
)

# Configurações da página Streamlit
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)

# Funções auxiliares para a UI
def get_download_link(df, filename, text):
    """Gera um link para download de dados como CSV"""
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {text}</a>'
    return href

def save_portfolio_config(config_dict):
    """Converte a configuração do portfólio para JSON e fornece um link para download"""
    json_str = json.dumps(config_dict, indent=4)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="portfolio_config.json">📥 Baixar configuração</a>'
    return href

# Título do aplicativo
st.title("Otimização de Portfólio com Algoritmos Genéticos")

# Barra lateral para configurações
with st.sidebar:
    st.header("Configurações")
    
    # Seleção de ativos
    st.subheader("1. Seleção de Ativos")
    
    market_selection = st.radio(
        "Mercado",
        options=["Brasil 🇧🇷", "EUA 🇺🇸", "Personalizado"],
        horizontal=True
    )
    
    if market_selection == "Brasil 🇧🇷":
        tickers_options = POPULAR_BR_TICKERS
        benchmark = "^BVSP"
    elif market_selection == "EUA 🇺🇸":
        tickers_options = POPULAR_US_TICKERS
        benchmark = "^GSPC"
    else:
        tickers_input = st.text_input(
            "Digite os tickers separados por vírgula (ex: AAPL,MSFT,GOOGL)",
            value="AAPL,MSFT,GOOGL,AMZN,META"
        )
        tickers_options = [t.strip() for t in tickers_input.split(",") if t.strip()]
        benchmark = st.text_input("Índice de referência (benchmark)", value="^GSPC")
    
    selected_tickers = st.multiselect(
        "Selecione os ativos para o portfólio",
        options=tickers_options,
        default=tickers_options[:5] if len(tickers_options) >= 5 else tickers_options
    )
    
    # Período de tempo
    st.subheader("2. Período de Análise")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Data inicial",
            value=datetime.strptime(DEFAULT_START_DATE, "%Y-%m-%d").date()
        )
    with col2:
        end_date = st.date_input(
            "Data final",
            value=datetime.strptime(DEFAULT_END_DATE, "%Y-%m-%d").date()
        )
    
    # Parâmetros do algoritmo genético
    st.subheader("3. Algoritmo Genético")
    
    population_size = st.slider(
        "Tamanho da população",
        min_value=20,
        max_value=500,
        value=DEFAULT_POPULATION_SIZE,
        step=10
    )
    
    num_generations = st.slider(
        "Número de gerações",
        min_value=10,
        max_value=200,
        value=DEFAULT_NUM_GENERATIONS,
        step=5
    )
    
    mutation_rate = st.slider(
        "Taxa de mutação",
        min_value=0.01,
        max_value=0.5,
        value=DEFAULT_MUTATION_RATE,
        step=0.01
    )
    
    risk_free_rate = st.slider(
        "Taxa livre de risco (%)",
        min_value=0.0,
        max_value=10.0,
        value=DEFAULT_RISK_FREE_RATE*100,
        step=0.1
    ) / 100.0
    
    # Configurações avançadas em um expander
    with st.expander("Configurações Avançadas"):
        min_weight = st.slider(
            "Peso mínimo por ativo",
            min_value=0.0,
            max_value=0.2,
            value=DEFAULT_MIN_WEIGHT,
            step=0.01
        )
        
        max_weight = st.slider(
            "Peso máximo por ativo",
            min_value=0.1,
            max_value=1.0,
            value=DEFAULT_MAX_WEIGHT,
            step=0.01
        )
        
        elitism_count = st.slider(
            "Elitismo (melhores indivíduos preservados)",
            min_value=1,
            max_value=10,
            value=DEFAULT_ELITISM_COUNT,
            step=1
        )
        
        evaluation_method = st.selectbox(
            "Método de avaliação",
            options=DEFAULT_EVALUATION_METHODS,
            format_func=lambda x: {
                "sharpe": "Índice de Sharpe",
                "sortino": "Índice de Sortino",
                "treynor": "Índice de Treynor",
                "var": "Value at Risk (VaR)",
                "multi": "Multi-objetivo (Retorno e Risco)"
            }.get(x, x)
        )
        
        init_strategy = st.selectbox(
            "Estratégia de inicialização",
            options=DEFAULT_INIT_STRATEGIES,
            format_func=lambda x: {
                "random": "Aleatória",
                "uniform": "Uniforme",
                "diversified": "Diversificada"
            }.get(x, x)
        )
        
        selection_method = st.selectbox(
            "Método de seleção",
            options=DEFAULT_SELECTION_METHODS,
            format_func=lambda x: {
                "tournament": "Torneio",
                "roulette": "Roleta",
                "rank": "Classificação"
            }.get(x, x)
        )
        
        crossover_method = st.selectbox(
            "Método de crossover",
            options=DEFAULT_CROSSOVER_METHODS,
            format_func=lambda x: {
                "uniform": "Uniforme",
                "single_point": "Ponto único",
                "blend": "Mistura"
            }.get(x, x)
        )
        
        mutation_distribution = st.selectbox(
            "Distribuição de mutação",
            options=DEFAULT_MUTATION_DISTRIBUTIONS,
            format_func=lambda x: {
                "normal": "Normal (Gaussiana)",
                "uniform": "Uniforme"
            }.get(x, x)
        )
    
    # Valor de investimento (opcional)
    investment = st.number_input(
        "Valor do investimento (opcional)",
        min_value=0,
        value=10000,
        step=1000
    )
    
    # Botão para iniciar otimização
    optimize_button = st.button("Otimizar Portfólio", type="primary")

# Área principal
if len(selected_tickers) < 2:
    st.warning("Selecione pelo menos 2 ativos para otimizar o portfólio.")
else:
    # Iniciar processo de otimização quando o botão for clicado
    if optimize_button:
        with st.spinner("Baixando dados históricos..."):
            # Converter datas para string no formato esperado
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Baixar dados dos ativos
            prices_df = download_data(
                selected_tickers,
                start_date_str,
                end_date_str
            )
            
            # Baixar dados do benchmark
            benchmark_df = download_data(
                [benchmark],
                start_date_str,
                end_date_str
            )
            
            if prices_df is None or prices_df.empty:
                st.error("Não foi possível baixar os dados dos ativos selecionados.")
                st.stop()
            
            if benchmark_df is None or benchmark_df.empty:
                st.info("Não foi possível baixar dados do benchmark. A análise continuará sem benchmark.")
                benchmark_returns = None
            else:
                # Processar dados dos ativos primeiro
                returns_df = calculate_returns(prices_df)
                
                # Processar dados do benchmark e alinhar com os retornos dos ativos
                benchmark_returns_full = calculate_returns(benchmark_df)
                # Garantir que use apenas as datas que existem em returns_df
                common_dates = returns_df.index.intersection(benchmark_returns_full.index)
                if len(common_dates) > 0:
                    benchmark_returns = benchmark_returns_full.loc[common_dates, benchmark]
                    returns_df = returns_df.loc[common_dates]
                else:
                    st.warning("Não há datas comuns entre os dados dos ativos e o benchmark. A análise continuará sem benchmark.")
                    benchmark_returns = None
        
        with st.spinner("Processando dados e otimizando portfólio..."):
            # Processar dados dos ativos (caso não tenha sido feito acima)
            if 'returns_df' not in locals():
                returns_df = calculate_returns(prices_df)
            
            cov_matrix = calculate_covariance_matrix(returns_df)
            
            # Configurar multi-objetivo baseado no método de avaliação
            multiobjective = evaluation_method == "multi"
            if multiobjective:
                evaluation_method = "sharpe"  # Usamos Sharpe como base para multi-objetivo
            
            # Executar otimização
            result = optimize_portfolio(
                selected_tickers=selected_tickers,
                start_date=start_date_str,
                end_date=end_date_str,
                investment=investment,
                population_size=population_size,
                num_generations=num_generations,
                mutation_rate=mutation_rate,
                risk_free_rate=risk_free_rate,
                min_weight=min_weight,
                max_weight=max_weight,
                evaluation_method=evaluation_method,
                multiobjective=multiobjective,
                init_strategy=init_strategy,
                selection_method=selection_method,
                crossover_method=crossover_method,
                mutation_distribution=mutation_distribution,
                elitism_count=elitism_count
            )
            
            if result is None:
                st.error("Falha na otimização do portfólio.")
                st.stop()
            
            # Desempacotar os resultados da otimização
            best_weights, best_fitness, pareto_front, returns_data, cov_matrix_data = result
        
        # Exibir resultados
        st.success("✅ Otimização concluída com sucesso!")
        
        # Criar tabs para diferentes visualizações
        tab1, tab2, tab3, tab4 = st.tabs([
            "Resumo do Portfólio", 
            "Análise de Risco", 
            "Evolução do Algoritmo", 
            "Dados e Exportação"
        ])
        
        with tab1:
            st.header("Portfólio Otimizado")
            
            # Criar um portfólio com os pesos otimizados
            portfolio = Portfolio(
                best_weights,
                selected_tickers,
                returns_df
            )
            
            # Layout em colunas para métricas
            col1, col2, col3 = st.columns(3)
            perf_metrics = portfolio.get_performance_metrics(benchmark_returns, risk_free_rate)
            
            with col1:
                st.metric(
                    "Retorno Anualizado", 
                    f"{perf_metrics['return']:.2%}"
                )
            
            with col2:
                volatility = portfolio.get_volatility()
                st.metric(
                    "Volatilidade Anualizada", 
                    f"{volatility:.2%}"
                )
            
            with col3:
                st.metric(
                    "Índice de Sharpe", 
                    f"{perf_metrics['sharpe']:.2f}"
                )
            
            # Gráfico de alocação do portfólio
            st.subheader("Alocação do Portfólio")
            fig_allocation = plot_portfolio_allocation(best_weights, selected_tickers)
            st.plotly_chart(fig_allocation, use_container_width=True)
            
            # Retornos acumulados
            st.subheader("Retornos Acumulados")
            fig_returns = plot_cumulative_returns(
                returns_df, 
                best_weights, 
                benchmark_returns
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # Matriz de correlação
            st.subheader("Matriz de Correlação")
            fig_corr = plot_correlation_matrix(returns_df)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with tab2:
            st.header("Análise de Risco")
            
            risk_metrics = portfolio.get_risk_metrics()
            
            # Métricas de risco
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Value at Risk (95%)", f"{risk_metrics['var_95']:.2%}")
                st.metric("Conditional VaR (95%)", f"{risk_metrics['cvar_95']:.2%}")
            
            with col2:
                st.metric("Máximo Drawdown", f"{risk_metrics['max_drawdown']:.2%}")
                st.metric("Volatilidade Anualizada", f"{portfolio.get_volatility():.2%}")
            
            # Drawdowns
            st.subheader("Análise de Drawdowns")
            fig_drawdown = plot_drawdowns(returns_df, best_weights)
            st.plotly_chart(fig_drawdown, use_container_width=True)
            
            # Fronteira eficiente
            if multiobjective and pareto_front:
                st.subheader("Fronteira Eficiente (Pareto)")
                fig_pareto = plot_pareto_front(pareto_front, risk_free_rate)
                st.plotly_chart(fig_pareto, use_container_width=True)
            else:
                st.subheader("Fronteira Eficiente")
                fig_frontier = plot_efficient_frontier(returns_df, cov_matrix, risk_free_rate)
                st.plotly_chart(fig_frontier, use_container_width=True)
                
        with tab3:
            st.header("Evolução do Algoritmo Genético")
            
            # Progresso da otimização
            # Infelizmente não temos o histórico de fitness disponível neste ponto,
            # então exibimos uma mensagem informativa
            st.info("""
                Historico de evolução não disponível para visualização.
                A otimização foi realizada com sucesso, mas os dados de evolução do algoritmo 
                não foram retornados pelo otimizador.
            """)
            
            # Parâmetros utilizados
            st.subheader("Parâmetros do Algoritmo")
            
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                st.write("**Configuração básica:**")
                st.write(f"- População: {population_size}")
                st.write(f"- Gerações: {num_generations}")
                st.write(f"- Taxa de mutação: {mutation_rate:.2f}")
                st.write(f"- Elitismo: {elitism_count}")
            
            with param_col2:
                st.write("**Estratégias utilizadas:**")
                st.write(f"- Avaliação: {evaluation_method}")
                st.write(f"- Inicialização: {init_strategy}")
                st.write(f"- Seleção: {selection_method}")
                st.write(f"- Crossover: {crossover_method}")
                st.write(f"- Mutação: {mutation_distribution}")
            
        with tab4:
            st.header("Dados e Exportação")
            
            # Exibir pesos otimizados
            weights_df = pd.DataFrame({
                'Ativo': selected_tickers,
                'Peso (%)': [round(w * 100, 2) for w in best_weights],
                'Alocação': [round(w * investment, 2) if investment > 0 else '-' for w in best_weights]
            })
            
            st.subheader("Alocação do Portfólio")
            st.dataframe(weights_df)
            
            # Link para download dos pesos
            st.markdown(
                get_download_link(
                    weights_df,
                    "portfolio_weights.csv",
                    "Baixar alocação do portfólio (CSV)"
                ),
                unsafe_allow_html=True
            )
            
            # Informações de retorno e risco
            st.subheader("Retornos e Correlações")
            st.write("**Retornos Diários**")
            st.dataframe(returns_df)
            st.markdown(
                get_download_link(
                    returns_df,
                    "returns_data.csv",
                    "Baixar dados de retorno (CSV)"
                ),
                unsafe_allow_html=True
            )
            
            # Configuração utilizada
            st.subheader("Configuração Utilizada")
            config_dict = {
                "assets": selected_tickers,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "algorithm": {
                    "population_size": population_size,
                    "generations": num_generations,
                    "mutation_rate": mutation_rate,
                    "evaluation_method": evaluation_method,
                    "risk_free_rate": risk_free_rate,
                    "min_weight": min_weight,
                    "max_weight": max_weight,
                    "elitism": elitism_count,
                    "multiobjective": multiobjective,
                    "init_strategy": init_strategy,
                    "selection_method": selection_method,
                    "crossover_method": crossover_method,
                    "mutation_distribution": mutation_distribution
                },
                "results": {
                    "return": perf_metrics['return'],
                    "volatility": portfolio.get_volatility(),
                    "sharpe": perf_metrics['sharpe'],
                    "var": risk_metrics['var_95'],
                    "max_drawdown": risk_metrics['max_drawdown']
                },
                "weights": {ticker: float(weight) for ticker, weight in zip(selected_tickers, best_weights)}
            }
            
            st.json(config_dict)
            st.markdown(
                save_portfolio_config(config_dict),
                unsafe_allow_html=True
            )
    else:
        # Mensagem inicial antes da otimização
        st.info("👈 Selecione os ativos e parâmetros desejados na barra lateral e clique em 'Otimizar Portfólio' para começar.")
        
        # Explicação do projeto
        st.header("Sobre o Projeto")
        st.write("""
        Esta aplicação utiliza **Algoritmos Genéticos** para otimizar portfólios de investimento, 
        buscando a melhor alocação de capital entre diferentes ativos para maximizar o retorno 
        ajustado ao risco.
        
        **Como funciona:**
        1. Selecione os ativos de interesse na barra lateral
        2. Defina o período de análise histórica
        3. Ajuste os parâmetros do algoritmo genético
        4. Clique em "Otimizar Portfólio"
        5. Analise os resultados nas diferentes abas
        
        **Métricas disponíveis:**
        - **Índice de Sharpe**: Retorno ajustado ao risco considerando a taxa livre de risco
        - **Índice de Sortino**: Similar ao Sharpe, mas considera apenas volatilidade negativa
        - **Índice de Treynor**: Retorno ajustado ao risco de mercado (beta)
        - **VaR (Value at Risk)**: Estima a perda máxima esperada
        - **Multi-objetivo**: Otimiza simultaneamente retorno e risco
        """)

# Rodapé com informações
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
        Desenvolvido para o Tech Challenge 02 - FIAP Pós em AI & ML<br>
        Fonte de dados: Yahoo Finance | Código: GitHub
    </div>
    """, 
    unsafe_allow_html=True
)
