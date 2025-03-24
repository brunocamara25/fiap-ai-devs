"""
Módulo de implementação do algoritmo genético para otimização de portfólio.

Este módulo implementa um algoritmo genético para otimização de portfólio de investimentos,
com suporte a diversas métricas de avaliação, estratégias de seleção, crossover e mutação.

"""
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Tuple, Union, Callable

from src.data.loader import download_data
from src.metrics.performance import calculate_metrics, calculate_sortino_ratio, calculate_treynor_ratio
from src.metrics.risk import calculate_var
from src.optimization.portfolio import Portfolio
from src.optimization.constraints import (
    weights_sum_to_one, enforce_weights_sum_to_one,
    weights_within_bounds, enforce_weights_within_bounds
)
from src.optimization.objective import (
    get_objective_function, sharpe_ratio_objective, sortino_ratio_objective,
    treynor_ratio_objective, var_objective, pareto_front_objective
)


class GeneticAlgorithm:
    """
    Classe que implementa um algoritmo genético para otimização de portfólio.
    
    Esta classe encapsula todas as funcionalidades do algoritmo genético,
    permitindo a configuração flexível de estratégias de seleção, crossover,
    mutação e avaliação.
    
    Attributes:
        population_size (int): Tamanho da população.
        num_generations (int): Número máximo de gerações.
        mutation_rate (float): Taxa de mutação.
        min_weight (float): Peso mínimo permitido para cada ativo.
        max_weight (float): Peso máximo permitido para cada ativo.
        evaluation_method (str): Método de avaliação ('sharpe', 'sortino', etc.).
        elitism_count (int): Número de melhores indivíduos a preservar entre gerações.
        multiobjective (bool): Se True, usa otimização multi-objetivo.
        init_strategy (str): Estratégia de inicialização da população.
        selection_method (str): Método de seleção de pais.
        crossover_method (str): Método de cruzamento genético.
        mutation_distribution (str): Distribuição usada para mutação.
    """

    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 50,
        mutation_rate: float = 0.1,
        min_weight: float = 0.01,
        max_weight: float = 0.5,
        evaluation_method: str = "sharpe",
        elitism_count: int = 2,
        multiobjective: bool = False,
        init_strategy: str = "random",
        selection_method: str = "tournament",
        crossover_method: str = "uniform",
        mutation_distribution: str = "normal"
    ):
        """
        Inicializa o algoritmo genético com os parâmetros fornecidos.
        
        Args:
            population_size: Tamanho da população.
            num_generations: Número máximo de gerações.
            mutation_rate: Taxa de mutação.
            min_weight: Peso mínimo permitido para cada ativo.
            max_weight: Peso máximo permitido para cada ativo.
            evaluation_method: Método de avaliação ('sharpe', 'sortino', etc.).
            elitism_count: Número de melhores indivíduos a preservar entre gerações.
            multiobjective: Se True, usa otimização multi-objetivo.
            init_strategy: Estratégia de inicialização ('random', 'uniform', etc.).
            selection_method: Método de seleção ('tournament', 'roulette', etc.).
            crossover_method: Método de cruzamento ('uniform', 'single_point', etc.).
            mutation_distribution: Distribuição para mutação ('normal', 'uniform').
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.evaluation_method = evaluation_method
        self.elitism_count = elitism_count
        self.multiobjective = multiobjective
        self.init_strategy = init_strategy
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_distribution = mutation_distribution

        # Atributos que serão inicializados durante a otimização
        self.population = None
        self.best_individual = None
        self.best_fitness = -np.inf if not self.multiobjective else None
        self.pareto_front = None
        self.fitness_history = []
        self.pareto_front_history = []

    def create_individual(self, size: int) -> np.ndarray:
        """
        Cria um indivíduo (pesos do portfólio) baseado na estratégia de inicialização.
        
        Args:
            size: Número de ativos no portfólio.
            
        Returns:
            np.ndarray: Pesos normalizados do portfólio.
        """
        if self.init_strategy == "uniform":
            # Inicializar com pesos uniformes
            weights = np.ones(size) / size
        elif self.init_strategy == "random":
            # Inicializar com pesos aleatórios
            weights = np.random.random(size)
        elif self.init_strategy == "return_based":
            # Inicialização baseada em retornos (seria implementada com dados reais)
            # Placeholder: inicializa como random por enquanto
            weights = np.random.random(size)
        elif self.init_strategy == "volatility_inverse":
            # Inicialização inversa à volatilidade (seria implementada com dados reais)
            # Placeholder: inicializa como random por enquanto
            weights = np.random.random(size)
        else:
            # Padrão: pesos aleatórios
            weights = np.random.random(size)

        # Aplicar restrições de peso mínimo e máximo
        weights = enforce_weights_within_bounds(weights, self.min_weight, self.max_weight)

        return weights

    def initialize_population(self, size: int, num_assets: int) -> List[np.ndarray]:
        """
        Inicializa a população com indivíduos criados aleatoriamente.
        
        Args:
            size: Tamanho da população.
            num_assets: Número de ativos no portfólio.
            
        Returns:
            List[np.ndarray]: Lista de indivíduos (pesos do portfólio).
        """
        population = []
        for _ in range(size):
            individual = self.create_individual(num_assets)
            population.append(individual)
        return population

    def evaluate_population(
        self,
        population: List[np.ndarray],
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float,
        market_returns: Optional[pd.Series] = None
    ) -> Union[List[float], List[Tuple[float, float]]]:
        """
        Avalia a população de portfólios com base em métricas de desempenho.
        
        Args:
            population: Lista de indivíduos (pesos do portfólio).
            returns: Retornos históricos dos ativos.
            cov_matrix: Matriz de covariância dos retornos.
            risk_free_rate: Taxa livre de risco.
            market_returns: Retornos do mercado (necessário para algumas métricas).
            
        Returns:
            Union[List[float], List[Tuple[float, float]]]: Scores de fitness para cada indivíduo.
        """
        fitness_scores = []

        if self.multiobjective:
            # Abordagem multi-objetivo retorna (retorno, risco)
            for weights in population:
                fitness = pareto_front_objective(weights, returns, cov_matrix)
                fitness_scores.append(fitness)
        else:
            # Abordagem de objetivo único
            if self.evaluation_method == "sharpe":
                for weights in population:
                    fitness = -sharpe_ratio_objective(
                        weights, returns, cov_matrix, risk_free_rate
                    )
                    fitness_scores.append(fitness)
            elif self.evaluation_method == "sortino":
                for weights in population:
                    fitness = -sortino_ratio_objective(
                        weights, returns, risk_free_rate
                    )
                    fitness_scores.append(fitness)
            elif self.evaluation_method == "treynor":
                if market_returns is None:
                    raise ValueError(
                        "Para o método de avaliação 'treynor', 'market_returns' deve ser fornecido."
                    )
                for weights in population:
                    fitness = -treynor_ratio_objective(
                        weights, returns, market_returns, risk_free_rate
                    )
                    fitness_scores.append(fitness)
            elif self.evaluation_method == "var":
                for weights in population:
                    # Negativo porque queremos minimizar o VaR
                    fitness = -var_objective(weights, returns)
                    fitness_scores.append(fitness)
            else:
                raise ValueError(
                    f"Método de avaliação '{self.evaluation_method}' não reconhecido."
                )

        return fitness_scores

    def select_pareto_front(
        self,
        population: List[np.ndarray],
        fitness_scores: List[Tuple[float, float]]
    ) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
        """
        Seleciona o Pareto Front (conjunto de soluções não dominadas).
        
        Args:
            population: Lista de indivíduos (pesos do portfólio).
            fitness_scores: Lista de scores de fitness (retorno e risco).
            
        Returns:
            List[Tuple[np.ndarray, Tuple[float, float]]]:
            Lista de indivíduos e seus scores no Pareto Front.
        """
        pareto_front = []

        for i, score_i in enumerate(fitness_scores):
            ret1, vol1 = score_i
            dominated = False
            for j, score_j in enumerate(fitness_scores):
                ret2, vol2 = score_j
                if (i != j and ret2 >= ret1 and vol2 <= vol1 and
                        (ret2 > ret1 or vol2 < vol1)):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append((population[i], fitness_scores[i]))

        # Ordenar por retorno decrescente
        pareto_front.sort(key=lambda x: x[1][0], reverse=True)
        return pareto_front

    def select_parents(
        self,
        population: List[np.ndarray],
        fitness_scores: List[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Seleciona dois pais da população usando o método de seleção especificado.
        
        Args:
            population: Lista de indivíduos.
            fitness_scores: Lista de scores de fitness.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Dois indivíduos selecionados como pais.
        """
        if self.selection_method == "tournament":
            # Seleção por torneio
            tournament_size = 3
            
            # Primeiro pai
            tournament = np.random.choice(len(population), tournament_size)
            idx_best = np.argmax([fitness_scores[i] for i in tournament])
            parent1 = population[tournament[idx_best]]
            
            # Segundo pai
            tournament = np.random.choice(len(population), tournament_size)
            idx_best = np.argmax([fitness_scores[i] for i in tournament])
            parent2 = population[tournament[idx_best]]
            
        elif self.selection_method == "roulette":
            # Seleção por roleta (seleção proporcional ao fitness)
            # Ajustar scores para serem positivos
            adjusted_scores = np.array(fitness_scores) - min(fitness_scores) + 1e-10
            probabilities = adjusted_scores / sum(adjusted_scores)
            
            # Selecionar dois pais baseados nas probabilidades
            selected_indices = np.random.choice(len(population), 2, p=probabilities, replace=False)
            parent1, parent2 = population[selected_indices[0]], population[selected_indices[1]]
            
        elif self.selection_method == "elitism":
            # Seleção elitista (os dois melhores indivíduos)
            sorted_indices = np.argsort(fitness_scores)[::-1]  # Ordenar em ordem decrescente
            parent1, parent2 = population[sorted_indices[0]], population[sorted_indices[1]]
            
        else:
            # Padrão: seleção por torneio
            tournament_size = 3
            
            # Primeiro pai
            tournament = np.random.choice(len(population), tournament_size)
            idx_best = np.argmax([fitness_scores[i] for i in tournament])
            parent1 = population[tournament[idx_best]]
            
            # Segundo pai
            tournament = np.random.choice(len(population), tournament_size)
            idx_best = np.argmax([fitness_scores[i] for i in tournament])
            parent2 = population[tournament[idx_best]]

        return parent1, parent2

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Realiza o cruzamento entre dois pais para gerar um filho usando o método especificado.
        
        Args:
            parent1: Primeiro pai (vetor de pesos).
            parent2: Segundo pai (vetor de pesos).
            
        Returns:
            np.ndarray: Filho gerado pelo cruzamento.
        """
        if self.crossover_method == "uniform":
            # Crossover uniforme (cada gene tem 50% de chance de vir de cada pai)
            mask = np.random.random(len(parent1)) > 0.5
            child = np.where(mask, parent1, parent2)
            
        elif self.crossover_method == "single_point":
            # Crossover de ponto único
            crossover_point = np.random.randint(1, len(parent1))
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
        elif self.crossover_method == "arithmetic":
            # Crossover aritmético (média ponderada dos pais)
            alpha = np.random.random()  # Peso aleatório
            child = alpha * parent1 + (1 - alpha) * parent2
            
        else:
            # Padrão: crossover uniforme
            mask = np.random.random(len(parent1)) > 0.5
            child = np.where(mask, parent1, parent2)
        
        # Normalizar pesos se necessário
        if not np.isclose(np.sum(child), 1.0):
            child = enforce_weights_sum_to_one(child)
            
        # Aplicar restrições de peso mínimo e máximo
        child = enforce_weights_within_bounds(child, self.min_weight, self.max_weight)
        
        return child

    def mutate(self, child: np.ndarray) -> np.ndarray:
        """
        Aplica mutação a um indivíduo usando a distribuição especificada.
        
        Args:
            child: Indivíduo (vetor de pesos) a ser mutado.
            
        Returns:
            np.ndarray: Indivíduo mutado.
        """
        mutated_child = child.copy()
        
        # Determinar quais genes (pesos) sofrerão mutação
        mutation_mask = np.random.random(len(child)) < self.mutation_rate
        
        if np.any(mutation_mask):
            if self.mutation_distribution == "normal":
                # Mutação normal (adiciona ruído gaussiano)
                mutation = np.random.normal(0, 0.1, len(child))
                mutated_child[mutation_mask] += mutation[mutation_mask]
                
            elif self.mutation_distribution == "uniform":
                # Mutação uniforme (substitui por valor aleatório)
                mutation = np.random.random(len(child))
                mutated_child[mutation_mask] = mutation[mutation_mask]
                
            else:
                # Padrão: mutação normal
                mutation = np.random.normal(0, 0.1, len(child))
                mutated_child[mutation_mask] += mutation[mutation_mask]
            
            # Garantir que os pesos somem 1
            mutated_child = enforce_weights_sum_to_one(mutated_child)
            
            # Aplicar restrições de peso mínimo e máximo
            mutated_child = enforce_weights_within_bounds(mutated_child, self.min_weight, self.max_weight)
        
        return mutated_child

    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0,
        market_returns: Optional[pd.Series] = None,
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, float, Optional[List[Tuple[np.ndarray, Tuple[float, float]]]]]:
        """
        Executa o algoritmo genético para otimizar o portfólio.
        
        Args:
            returns: Retornos históricos dos ativos.
            cov_matrix: Matriz de covariância dos retornos.
            risk_free_rate: Taxa livre de risco.
            market_returns: Retornos do mercado (necessário para algumas métricas).
            callback: Função de callback chamada a cada geração.
            
        Returns:
            Tuple: (melhor_individuo, melhor_fitness, historico_pareto_front)
        """
        # Inicializar população
        num_assets = len(returns.columns)
        self.population = self.initialize_population(self.population_size, num_assets)

        # Histórico do melhor fitness
        self.fitness_history = []

        # Melhor indivíduo encontrado
        self.best_individual = None
        self.best_fitness = -np.inf if not self.multiobjective else None

        # Histórico do Pareto Front (para abordagem multi-objetivo)
        self.pareto_front_history = []

        # Evolução do algoritmo genético
        for generation in range(self.num_generations):
            # Avaliar população
            fitness_scores = self.evaluate_population(
                self.population, returns, cov_matrix, risk_free_rate, market_returns
            )

            # Para abordagem multi-objetivo
            if self.multiobjective:
                self.pareto_front = self.select_pareto_front(self.population, fitness_scores)
                self.pareto_front_history.append(self.pareto_front)

                # Escolher o indivíduo com maior retorno no Pareto Front como melhor
                if self.best_individual is None or self.pareto_front[0][1][0] > self.best_fitness:
                    self.best_individual = self.pareto_front[0][0]
                    self.best_fitness = self.pareto_front[0][1][0]

                self.fitness_history.append(self.best_fitness)
            else:
                # Para abordagem de objetivo único
                best_idx = np.argmax(fitness_scores)
                if self.best_individual is None or fitness_scores[best_idx] > self.best_fitness:
                    self.best_individual = self.population[best_idx]
                    self.best_fitness = fitness_scores[best_idx]

                self.fitness_history.append(self.best_fitness)

            # Chamar callback se fornecido
            if callback is not None:
                callback(
                    generation,
                    self.population,
                    fitness_scores,
                    self.best_individual,
                    self.best_fitness
                )

            # Criar nova população
            new_population = []

            # Elitismo - preservar os melhores indivíduos
            if self.elitism_count > 0:
                if self.multiobjective:
                    for i, (elite, _) in enumerate(self.pareto_front):
                        if i < self.elitism_count:
                            new_population.append(elite)
                else:
                    elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
                    for idx in elite_indices:
                        new_population.append(self.population[idx].copy())

            # Criar o resto da população
            while len(new_population) < self.population_size:
                # Seleção de pais
                if self.multiobjective:
                    # Selecionar aleatoriamente do Pareto Front
                    indices = np.random.choice(len(self.pareto_front), size=2, replace=False)
                    parent1 = self.pareto_front[indices[0]][0]
                    parent2 = self.pareto_front[indices[1]][0]
                else:
                    parent1, parent2 = self.select_parents(self.population, fitness_scores)

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutação
                child = self.mutate(child)

                new_population.append(child)

            # Atualizar população
            self.population = new_population

        return (
            self.best_individual,
            self.best_fitness,
            self.pareto_front_history if self.multiobjective else None
        )


def optimize_portfolio(
    selected_tickers, start_date, end_date, investment, population_size=100, num_generations=50,
    mutation_rate=0.1, risk_free_rate=0.01, min_weight=0.01, max_weight=0.4,
    evaluation_method="sharpe", multiobjective=True,
    init_strategy="random", selection_method="tournament", crossover_method="uniform",
    mutation_distribution="normal", elitism_count=2
):
    """
    Função principal para otimização de portfólio usando o algoritmo genético.
    
    Esta função encapsula todo o processo de otimização, incluindo o download de dados,
    pré-processamento, otimização e visualização dos resultados usando Streamlit.
    
    Args:
        selected_tickers: Lista de tickers selecionados.
        start_date: Data de início dos dados históricos.
        end_date: Data de fim dos dados históricos.
        investment: Valor total do investimento.
        population_size: Tamanho da população do algoritmo genético.
        num_generations: Número de gerações para o algoritmo genético.
        mutation_rate: Taxa de mutação.
        risk_free_rate: Taxa livre de risco anualizada.
        min_weight: Peso mínimo permitido para cada ativo.
        max_weight: Peso máximo permitido para cada ativo.
        evaluation_method: Método de avaliação dos portfólios.
        multiobjective: Se True, usa otimização multi-objetivo.
        init_strategy: Estratégia de inicialização ("random", "uniform", etc.)
        selection_method: Método de seleção ("tournament", "roulette", etc.)
        crossover_method: Método de crossover ("uniform", "single_point", etc.)
        mutation_distribution: Distribuição para mutação ("normal", "uniform", etc.)
        elitism_count: Número de indivíduos elitistas a preservar entre gerações.
        
    Returns:
        Tuple: (melhores_pesos, melhor_fitness, historico_pareto_front, retornos, matriz_cov)
    """
    # Verificar seleção mínima de ações
    if len(selected_tickers) < 2:
        st.warning("Por favor, selecione pelo menos 2 ações.")
        return None, None, None, None, None

    # Baixar dados
    with st.spinner("Baixando dados das ações..."):
        data = download_data(selected_tickers, start_date, end_date)
        if data is None:
            return None, None, None, None, None

    # Preparar dados
    returns = data.pct_change().dropna()
    market_returns = data[selected_tickers].pct_change().dropna().mean(axis=1)\
        .reindex(returns.index)

    # Dividir dados em treinamento e teste
    train_size = int(0.7 * len(returns))
    train_data, test_data = returns[:train_size], returns[train_size:]
    train_cov_matrix, test_cov_matrix = train_data.cov(), test_data.cov()

    # Configurar visualização do processo
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_text = st.empty()

    # Containers para gráficos em tempo real
    progress_chart = st.empty()
    allocation_chart = st.empty()

    # Função de callback para atualizar a interface durante a otimização
    def update_callback(generation, population, fitness_scores, best_weights, best_fitness):
        # Atualizar barra de progresso
        progress = (generation + 1) / num_generations
        progress_bar.progress(progress)

        # Atualizar texto de status
        status_text.text(f"Geração {generation + 1}/{num_generations}")

        # Atualizar métricas
        if multiobjective:
            portfolio_return, portfolio_volatility, sharpe = calculate_metrics(
                best_weights, train_data, train_cov_matrix, risk_free_rate
            )
            metrics_text.markdown(f"""
                **Métricas do Melhor Portfólio:**
                - Retorno Anualizado: {portfolio_return:.2%}
                - Volatilidade Anualizada: {portfolio_volatility:.2%}
                - Índice de Sharpe: {sharpe:.2f}
            """)
        else:
            # Cálculo básico para qualquer método de avaliação
            portfolio_return, portfolio_volatility, sharpe = calculate_metrics(
                best_weights, train_data, train_cov_matrix, risk_free_rate
            )
            metrics_text.markdown(f"""
                **Métricas do Melhor Portfólio:**
                - Retorno Anualizado: {portfolio_return:.2%}
                - Volatilidade Anualizada: {portfolio_volatility:.2%}
                - Índice de Sharpe: {sharpe:.2f}
            """)

        # Atualizar gráficos a cada 5 gerações ou na última geração
        if generation % 5 == 0 or generation == num_generations - 1:
            # Gráfico de alocação de ativos
            allocation_data = pd.DataFrame({
                'Ativo': selected_tickers,
                'Peso': best_weights
            })
            allocation_chart.bar_chart(allocation_data.set_index('Ativo'))

    # Criar e configurar o algoritmo genético
    ga = GeneticAlgorithm(
        population_size=population_size,
        num_generations=num_generations,
        mutation_rate=mutation_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        evaluation_method=evaluation_method,
        multiobjective=multiobjective,
        elitism_count=elitism_count,
        init_strategy=init_strategy,
        selection_method=selection_method,
        crossover_method=crossover_method,
        mutation_distribution=mutation_distribution
    )

    # Executar otimização
    best_weights, best_fitness, pareto_front_history = ga.optimize(
        train_data, train_cov_matrix, risk_free_rate, market_returns, update_callback
    )

    # Exibir resultados finais
    st.header("🏆 Resultados Finais")

    # Avaliar o portfólio otimizado
    portfolio = Portfolio(
        weights=best_weights,
        assets=selected_tickers,
        returns=test_data,
        cov_matrix=test_cov_matrix,
        risk_free_rate=risk_free_rate,
        market_returns=market_returns
    )

    # Exibir métricas de desempenho
    st.subheader("📊 Métricas de Desempenho")
    performance_metrics = portfolio.get_performance_metrics()
    risk_metrics = portfolio.get_risk_metrics()

    col1, col2 = st.columns(2)
    with col1:
        st.write("Métricas de Desempenho:")
        st.dataframe(pd.Series(performance_metrics, name="Valor"))

    with col2:
        st.write("Métricas de Risco:")
        st.dataframe(pd.Series(risk_metrics, name="Valor"))

    # Visualização da alocação de ativos
    st.subheader("💰 Alocação de Ativos")
    weights_dict = portfolio.get_weights_dict()
    weights_df = pd.DataFrame(list(weights_dict.items()), columns=["Ativo", "Peso"])
    weights_df["Valor (R$)"] = weights_df["Peso"] * investment

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(weights_df)

    with col2:
        st.bar_chart(weights_df.set_index("Ativo")["Peso"])

    return (
        ga.best_individual,
        ga.best_fitness,
        ga.pareto_front_history if multiobjective else None,
        train_data,  # Mesmos dados usados para otimização
        train_cov_matrix  # Matriz de covariância usada na otimização
    )
