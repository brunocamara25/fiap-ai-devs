import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.spatial import distance
from metrics import calculate_metrics, calculate_diversification, calculate_sortino_ratio, calculate_treynor_ratio, calculate_var
from data import download_data
from utils import normalize_fitness_scores
from visualization import *
import plotly.express as px

import numpy as np

def create_individual(size, strategy="random", returns=None):
    """
    Cria um indivíduo (pesos do portfólio) com base na estratégia especificada.

    Parâmetros:
        size (int): Número de ativos no portfólio.
        strategy (str): Estratégia de inicialização ("random", "uniform", "return_based", "volatility_inverse").
        returns (pd.DataFrame, opcional): Retornos históricos dos ativos (necessário para algumas estratégias).

    Retorna:
        np.ndarray: Pesos normalizados do portfólio.
    """
    if strategy == "random":
        weights = np.random.random(size)
    elif strategy == "uniform":
        weights = np.ones(size) / size
    elif strategy == "return_based":
        if returns is None:
            raise ValueError("Para a estratégia 'return_based', 'returns' deve ser fornecido.")
        weights = returns.mean().values
    elif strategy == "volatility_inverse":
        if returns is None:
            raise ValueError("Para a estratégia 'volatility_inverse', 'returns' deve ser fornecido.")
        weights = 1 / returns.std().values
    else:
        raise ValueError("Estratégia desconhecida para inicialização.")
    return weights / np.sum(weights)

def evaluate_population(population, returns, cov_matrix, risk_free_rate, metric=None, market_returns=None, multiobjective=False):
    """
    Avalia a população de portfólios com base em métricas de desempenho.

    Parâmetros:
        population (list): Lista de indivíduos (pesos do portfólio).
        returns (pd.DataFrame): Retornos históricos dos ativos.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        risk_free_rate (float): Taxa livre de risco.
        metric (str, opcional): Métrica de avaliação ("sharpe", "sortino", "treynor", "var").
        market_returns (pd.Series, opcional): Retornos do mercado (necessário para algumas métricas).
        multiobjective (bool): Se True, avalia retorno e risco como objetivos separados.

    Retorna:
        list: Lista de scores de fitness para cada indivíduo.
    """
    fitness_scores = []
    for weights in population:
        if multiobjective:
            ret = np.sum(returns.mean() * weights) * 252
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            fitness_scores.append((ret, vol))
        else:
            if metric == "sharpe":
                _, _, score = calculate_metrics(weights, returns, cov_matrix, risk_free_rate)
            elif metric == "sortino":
                score = calculate_sortino_ratio(weights, returns, risk_free_rate)
            elif metric == "treynor":
                score = calculate_treynor_ratio(weights, returns, cov_matrix, risk_free_rate, market_returns)
            elif metric == "var":
                score = -calculate_var(weights, returns)
            fitness_scores.append(score)
    return fitness_scores

def select_pareto_front(population, fitness_scores):
    """
    Seleciona o Pareto Front (conjunto de soluções não dominadas).

    Parâmetros:
        population (list): Lista de indivíduos (pesos do portfólio).
        fitness_scores (list): Lista de scores de fitness (retorno e risco).

    Retorna:
        list: Lista de indivíduos e seus scores no Pareto Front.
    """
    pareto_front = []
    for i, (ret1, vol1) in enumerate(fitness_scores):
        dominated = False
        for j, (ret2, vol2) in enumerate(fitness_scores):
            if i != j and ret2 >= ret1 and vol2 <= vol1 and (ret2 > ret1 or vol2 < vol1):
                dominated = True
                break
        if not dominated:
            pareto_front.append((population[i], fitness_scores[i]))
    # Ordenar por retorno decrescente
    pareto_front.sort(key=lambda x: x[1][0], reverse=True)
    return pareto_front

def select_parents_from_pareto(pareto_front):
    """
    Seleciona dois pais aleatoriamente do Pareto Front.

    Parâmetros:
        pareto_front (list): Lista de indivíduos no Pareto Front.

    Retorna:
        tuple: Dois indivíduos selecionados como pais.
    """
    indices = np.random.choice(len(pareto_front), size=2, replace=False)
    parent1 = pareto_front[indices[0]][0]
    parent2 = pareto_front[indices[1]][0]
    return parent1, parent2

def select_parents(population, fitness_scores, method="tournament", tournament_size=3):
    """
    Seleciona dois pais da população com base no método especificado.

    Parâmetros:
        population (list): Lista de indivíduos.
        fitness_scores (list): Lista de scores de fitness.
        method (str): Método de seleção ("tournament", "roulette", "elitism").
        tournament_size (int): Tamanho do torneio (apenas para o método "tournament").

    Retorna:
        tuple: Dois indivíduos selecionados como pais.
    """
    if method == "tournament":
        tournament = np.random.choice(len(population), tournament_size)
        parent1 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
        tournament = np.random.choice(len(population), tournament_size)
        parent2 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
    elif method == "roulette":
        probabilities = fitness_scores / np.sum(fitness_scores)
        parent1 = population[np.random.choice(len(population), p=probabilities)]
        parent2 = population[np.random.choice(len(population), p=probabilities)]
    elif method == "elitism":
        sorted_indices = np.argsort(fitness_scores)[-2:]
        parent1, parent2 = population[sorted_indices[0]], population[sorted_indices[1]]
    return parent1, parent2

def crossover(parent1, parent2, method="uniform", crossover_rate=0.8):
    """
    Realiza o crossover entre dois pais para gerar um filho.

    Parâmetros:
        parent1 (np.ndarray): Pesos do primeiro pai.
        parent2 (np.ndarray): Pesos do segundo pai.
        method (str): Método de crossover ("uniform", "single_point", "arithmetic").
        crossover_rate (float): Taxa de crossover.

    Retorna:
        np.ndarray: Pesos do filho gerado.
    """
    if np.random.random() < crossover_rate:
        if method == "uniform":
            mask = np.random.randint(0, 2, len(parent1))
            child = mask * parent1 + (1 - mask) * parent2
        elif method == "single_point":
            point = np.random.randint(1, len(parent1))
            child = np.concatenate((parent1[:point], parent2[point:]))
        elif method == "arithmetic":
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
        return child / np.sum(child)
    return parent1.copy()

def mutate(child, mutation_rate, mutation_intensity, min_weight=0.01, max_weight=1.0, distribution="normal"):
    """
    Aplica mutação a um indivíduo.

    Parâmetros:
        child (np.ndarray): Pesos do indivíduo.
        mutation_rate (float): Taxa de mutação.
        mutation_intensity (float): Intensidade da mutação.
        min_weight (float): Peso mínimo permitido.
        max_weight (float): Peso máximo permitido.
        distribution (str): Distribuição da mutação ("normal", "uniform").

    Retorna:
        np.ndarray: Pesos do indivíduo após a mutação.
    """
    if np.random.random() < mutation_rate:
        if distribution == "normal":
            mutation = np.random.normal(0, mutation_intensity, len(child))
        elif distribution == "uniform":
            mutation = np.random.uniform(-mutation_intensity, mutation_intensity, len(child))
        child = child + mutation
        child = np.clip(child, min_weight, max_weight)
    return child / np.sum(child)

def optimize_portfolio(
    selected_tickers, start_date, end_date, investment, population_size, num_generations,
    mutation_rate, risk_free_rate, min_weight, max_weight,
    init_strategy="random", 
    evaluation_method="treynor",
    selection_method="tournament", 
    crossover_method="uniform", 
    mutation_distribution="normal",
    elitism_count=1,
    multiobjective=True
):
    # Verificar seleção mínima de ações
    if len(selected_tickers) < 2:
        st.warning("Por favor, selecione pelo menos 2 ações.")
        return

    # Baixar dados
    with st.spinner("Baixando dados das ações..."):
        data = download_data(selected_tickers, start_date, end_date)
        if data is None:
            return


    # Preparar dados
    returns = data.pct_change().dropna()
    market_returns = data[selected_tickers].pct_change().dropna().mean(axis=1).reindex(returns.index).fillna(method='ffill').fillna(method='bfill')
    benchmark_returns = market_returns.mean()
    st.markdown(f"**Retorno do Benchmark:** {benchmark_returns:.2%}")

    # Dividir dados em treinamento e teste
    train_data, test_data = returns[:int(0.7 * len(returns))], returns[int(0.7 * len(returns)):]
    train_cov_matrix, test_cov_matrix = train_data.cov(), test_data.cov()

    # Organização em abas para exibição inicial
    st.header("📊 Dados Iniciais")
    tabs = st.tabs(["📈 Retornos e Estatísticas", "📊 Matrizes de Covariância e Correlação", "📉 Visualizações"])
    
    with tabs[0]:
        st.subheader("Retornos e Estatísticas")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Dados de Retorno (Treinamento):")
            st.dataframe(train_data)
        with col2:
            st.write("Estatísticas Descritivas dos Dados de Retorno:")
            st.write(train_data.describe())

    with tabs[1]:
        st.subheader("Matrizes de Covariância e Correlação")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Matriz de Covariância (Treinamento):")
            st.dataframe(train_cov_matrix)
        with col2:
            correlation_matrix = train_cov_matrix.corr()
            st.write("Matriz de Correlação (Treinamento):")
            st.dataframe(correlation_matrix)

    with tabs[2]:
        st.subheader("Visualizações")
        col1, col2 = st.columns(2)
        with col1:
            plot_correlation_matrix(correlation_matrix)
        with col2:
            plot_cumulative_returns(data)

    st.header("📊 Informações iniciais")
    # Exibir informações iniciais
    display_initial_data(data, train_cov_matrix)

    st.header("📊 Otimização de portfolio")
    col1, col2 = st.columns(2)

    # Inicializar população
    population = [create_individual(len(selected_tickers), strategy=init_strategy, returns=train_data) for _ in range(population_size)]
    best_sharpe, best_weights, best_history = float('-inf'), None, []

    pareto_front_history = []  # Para armazenar o histórico do Pareto Front

    # Configurar exibição de progresso
    progress_bar, status_text, metrics_text = st.progress(0), st.empty(), st.empty()

    col1, col2 = st.columns(2)

    # Dividir a interface em duas colunas
    with col1:
        progress_chart = st.empty()
        pareto_placeholder = st.empty()
    with col2:
        allocation_chart = st.empty()
        index_placeholder = st.empty()
    
    # Otimização
    for generation in range(num_generations):
        fitness_scores, pareto_front = evaluate_population_step(
            population, train_data, train_cov_matrix, risk_free_rate, evaluation_method, market_returns, multiobjective
        )

        if multiobjective:
            pareto_front_history.append(pareto_front)

        best_weights, best_sharpe = update_best_solution(
            fitness_scores, pareto_front, best_weights, best_sharpe, multiobjective, population
        )        
        best_history.append(best_sharpe)

        # Atualizar exibição
        update_progress_display(
            generation, num_generations, progress_bar, status_text, metrics_text,
            best_weights, train_data, train_cov_matrix, risk_free_rate, evaluation_method, best_sharpe
        )
        
        if generation % 2 == 0:
            if multiobjective:  # Exibir gráficos relacionados ao multiobjetivo apenas se for True
                update_real_time_charts(pareto_front_history, best_history, evaluation_method, pareto_placeholder, index_placeholder)
            else:
                update_progress_chart(progress_chart, best_history)
                update_allocation_chart(allocation_chart, best_weights, selected_tickers)

    
        # Gerar nova população
        population = generate_new_population(
            population, fitness_scores, pareto_front, multiobjective,
            selection_method, crossover_method, mutation_rate, generation,
            num_generations, min_weight, max_weight, mutation_distribution,
            elitism_count=elitism_count
        )

    # Chamando o método display_final_results no final da função optimize_portfolio
    display_final_results(
        best_weights=best_weights,               # Pesos do melhor portfólio encontrado
        test_data=test_data,                     # Dados de teste (retornos)
        test_cov_matrix=test_cov_matrix,         # Matriz de covariância dos dados de teste
        risk_free_rate=risk_free_rate,           # Taxa livre de risco
        investment=investment,                   # Valor do investimento
        selected_tickers=selected_tickers,       # Lista de tickers selecionados
        data=data,                               # Dados históricos das ações
        train_cov_matrix=train_cov_matrix,       # Matriz de covariância dos dados de treinamento
        returns=returns,                         # Retornos calculados a partir dos dados históricos
        evaluation_method=evaluation_method,     # Método de avaliação (ex.: "treynor")
        pareto_front_history=pareto_front_history,
        best_history=best_history
    )

def evaluate_population_step(population, train_data, train_cov_matrix, risk_free_rate, evaluation_method, market_returns, multiobjective):
    """
    Avaliar a população e retornar os scores e o Pareto Front.

    Parâmetros:
    - population (list): Lista de indivíduos (pesos do portfólio).
    - train_data (pd.DataFrame): Retornos históricos de treinamento.
    - train_cov_matrix (pd.DataFrame): Matriz de covariância dos retornos de treinamento.
    - risk_free_rate (float): Taxa livre de risco.
    - evaluation_method (str): Método de avaliação (ex.: "sharpe", "sortino").
    - market_returns (pd.Series): Retornos do mercado (necessário para algumas métricas).
    - multiobjective (bool): Se True, avalia retorno e risco como objetivos separados.

    Retorna:
    - fitness_scores (list): Lista de scores de fitness para cada indivíduo.
    - pareto_front (list ou None): Pareto Front se multiobjetivo, caso contrário None.
    """
    fitness_scores = evaluate_population(population, train_data, train_cov_matrix, risk_free_rate, evaluation_method, market_returns, multiobjective)
    if multiobjective:
        normalized_scores = normalize_fitness_scores(fitness_scores)
        pareto_front = select_pareto_front(population, normalized_scores)
    else:
        normalized_scores = fitness_scores  # Não normaliza se não for multiobjetivo
        pareto_front = None
    return fitness_scores, pareto_front


def update_best_solution(fitness_scores, pareto_front, best_weights, best_sharpe, multiobjective, population):
    """
    Atualizar a melhor solução encontrada.

    Parâmetros:
    - fitness_scores (list): Lista de scores de fitness.
    - pareto_front (list): Pareto Front (apenas se multiobjetivo).
    - best_weights (np.ndarray): Pesos do melhor portfólio encontrado até agora.
    - best_sharpe (float): Melhor índice Sharpe encontrado até agora.
    - multiobjective (bool): Se True, avalia retorno e risco como objetivos separados.
    - population (list): Lista de indivíduos (pesos do portfólio).

    Retorna:
    - best_weights (np.ndarray): Pesos do melhor portfólio atualizado.
    - best_sharpe (float): Melhor índice Sharpe atualizado.
    """
    if multiobjective:
        best_individual = max(pareto_front, key=lambda x: x[1][0])  # Baseado no maior retorno
        best_weights = best_individual[0]
        best_sharpe = best_individual[1][0] / best_individual[1][1]  # Retorno/Risco
    else:
        max_idx = np.argmax(fitness_scores)
        if fitness_scores[max_idx] > best_sharpe:
            best_sharpe = fitness_scores[max_idx]
            best_weights = population[max_idx].copy()
    return best_weights, best_sharpe


def generate_new_population(population, fitness_scores, pareto_front, multiobjective, selection_method, crossover_method, mutation_rate, generation, num_generations, min_weight, max_weight, mutation_distribution, elitism_count=1):
    """
    Gerar nova população com seleção, crossover, mutação e elitismo.

    Parâmetros:
    - population (list): Lista de indivíduos (pesos do portfólio).
    - fitness_scores (list): Lista de scores de fitness.
    - pareto_front (list): Pareto Front (apenas se multiobjetivo).
    - multiobjective (bool): Se True, avalia retorno e risco como objetivos separados.
    - selection_method (str): Método de seleção (ex.: "tournament", "roulette").
    - crossover_method (str): Método de crossover (ex.: "uniform", "single_point").
    - mutation_rate (float): Taxa de mutação.
    - generation (int): Geração atual.
    - num_generations (int): Número total de gerações.
    - min_weight (float): Peso mínimo permitido.
    - max_weight (float): Peso máximo permitido.
    - mutation_distribution (str): Distribuição da mutação ("normal", "uniform").
    - elitism_count (int): Número de indivíduos elitistas a serem preservados.

    Retorna:
    - new_population (list): Nova população gerada.
    """
    new_population = []

    # **Elitismo**: Preservar os melhores indivíduos
    if multiobjective:
        # Ordenar o Pareto Front pelo maior retorno
        pareto_front_sorted = sorted(pareto_front, key=lambda x: x[1][0], reverse=True)
        elites = [individual[0] for individual in pareto_front_sorted[:elitism_count]]
    else:
        # Ordenar a população pelo maior fitness score
        sorted_indices = np.argsort(fitness_scores)[-elitism_count:]
        elites = [population[i] for i in sorted_indices]

    # Adicionar os indivíduos elitistas à nova população
    new_population.extend(elites)

    # Gerar o restante da população
    while len(new_population) < len(population):
        if multiobjective:
            parent1, parent2 = select_parents_from_pareto(pareto_front)
        else:
            parent1, parent2 = select_parents(population, fitness_scores, method=selection_method)

        # Realizar crossover
        child = crossover(parent1, parent2, method=crossover_method)

        # Realizar mutação
        mutation_intensity = 0.1 * (1 - generation / num_generations)  # Mutação adaptativa
        child = mutate(child, mutation_rate, mutation_intensity, min_weight, max_weight, distribution=mutation_distribution)

        new_population.append(child)

    return new_population


def log_evolution(generation, parents, children):
    """
    Log da evolução do algoritmo.

    Parâmetros:
    - generation (int): Geração atual.
    - parents (list): Lista de pais selecionados.
    - children (list): Lista de filhos gerados.
    """
    st.text(f"Geração {generation}:")
    st.text(f"Pais Selecionados: {parents}")
    st.text(f"Filhos Gerados: {children}")
