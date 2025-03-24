"""
Testes para o módulo de algoritmos genéticos.

Este módulo contém testes unitários para as funcionalidades de otimização
por algoritmos genéticos.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.models.genetic_algorithm import GeneticAlgorithm
from src.optimization.objective import sharpe_ratio_objective, sortino_ratio_objective


class TestGeneticAlgorithm:
    """Testes para a classe GeneticAlgorithm."""
    
    @pytest.fixture
    def ga_instance(self):
        """Fixture que cria uma instância do algoritmo genético para testes."""
        return GeneticAlgorithm(
            population_size=20,
            num_generations=5,
            mutation_rate=0.1,
            min_weight=0.01,
            max_weight=0.5,
            evaluation_method="sharpe",
            elitism_count=2
        )
    
    @pytest.fixture
    def sample_data(self):
        """Fixture que cria dados de retorno e covariância para testes."""
        # Criar dados de retorno sintéticos
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)  # Para reprodutibilidade
        
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, size=(100, 5)),
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        )
        
        # Calcular matriz de covariância
        cov_matrix = returns_data.cov()
        
        return returns_data, cov_matrix
    
    def test_initialization(self, ga_instance):
        """Teste para verificar a inicialização correta do GA."""
        assert ga_instance.population_size == 20
        assert ga_instance.num_generations == 5
        assert ga_instance.mutation_rate == 0.1
        assert ga_instance.min_weight == 0.01
        assert ga_instance.max_weight == 0.5
        assert ga_instance.evaluation_method == "sharpe"
        assert ga_instance.elitism_count == 2
        assert ga_instance.population is None
        assert ga_instance.best_individual is None
    
    def test_create_individual(self, ga_instance):
        """Teste para verificar a criação de indivíduos."""
        # Testar criação de indivíduos com diferentes estratégias
        individual = ga_instance.create_individual(5)
        
        assert isinstance(individual, np.ndarray)
        assert len(individual) == 5
        assert np.isclose(np.sum(individual), 1.0)  # Deve somar 1
        assert np.all(individual >= ga_instance.min_weight)
        assert np.all(individual <= ga_instance.max_weight)
    
    def test_initialize_population(self, ga_instance):
        """Teste para verificar a inicialização da população."""
        population = ga_instance.initialize_population(10, 5)
        
        assert isinstance(population, list)
        assert len(population) == 10
        assert all(isinstance(ind, np.ndarray) for ind in population)
        assert all(len(ind) == 5 for ind in population)
        assert all(np.isclose(np.sum(ind), 1.0) for ind in population)
    
    def test_evaluate_population(self, ga_instance, sample_data):
        """Teste para verificar a avaliação da população."""
        returns, cov_matrix = sample_data
        population = ga_instance.initialize_population(10, 5)
        
        # Avaliar população com objetivo único
        fitness_scores = ga_instance.evaluate_population(
            population, returns, cov_matrix, 0.01
        )
        
        assert isinstance(fitness_scores, list)
        assert len(fitness_scores) == 10
        assert all(isinstance(score, float) for score in fitness_scores)
        
        # Mudar para multi-objetivo e testar
        ga_instance.multiobjective = True
        fitness_scores_multi = ga_instance.evaluate_population(
            population, returns, cov_matrix, 0.01
        )
        
        assert isinstance(fitness_scores_multi, list)
        assert len(fitness_scores_multi) == 10
        assert all(isinstance(score, tuple) and len(score) == 2 for score in fitness_scores_multi)
    
    def test_crossover(self, ga_instance):
        """Teste para verificar o crossover genético."""
        parent1 = np.array([0.2, 0.3, 0.1, 0.2, 0.2])
        parent2 = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        
        # Testar diferentes métodos de crossover
        ga_instance.crossover_method = "uniform"
        child1 = ga_instance.crossover(parent1, parent2)
        
        assert isinstance(child1, np.ndarray)
        assert len(child1) == 5
        assert np.isclose(np.sum(child1), 1.0)
        
        ga_instance.crossover_method = "single_point"
        child2 = ga_instance.crossover(parent1, parent2)
        
        assert isinstance(child2, np.ndarray)
        assert len(child2) == 5
        assert np.isclose(np.sum(child2), 1.0)
    
    def test_mutate(self, ga_instance):
        """Teste para verificar a mutação genética."""
        individual = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Forçar uma alta taxa de mutação para garantir que ocorra
        ga_instance.mutation_rate = 1.0
        
        # Testar diferentes distribuições de mutação
        ga_instance.mutation_distribution = "normal"
        mutated1 = ga_instance.mutate(individual.copy())
        
        assert isinstance(mutated1, np.ndarray)
        assert len(mutated1) == 5
        assert np.isclose(np.sum(mutated1), 1.0)
        assert np.all(mutated1 >= ga_instance.min_weight)
        assert np.all(mutated1 <= ga_instance.max_weight)
        
        ga_instance.mutation_distribution = "uniform"
        mutated2 = ga_instance.mutate(individual.copy())
        
        assert isinstance(mutated2, np.ndarray)
        assert len(mutated2) == 5
        assert np.isclose(np.sum(mutated2), 1.0)
        assert np.all(mutated2 >= ga_instance.min_weight)
        assert np.all(mutated2 <= ga_instance.max_weight)
    
    def test_optimize(self, ga_instance, sample_data):
        """Teste para verificar o processo completo de otimização."""
        returns, cov_matrix = sample_data
        
        # Testar otimização com objetivo único (Sharpe)
        best_weights, best_fitness, _ = ga_instance.optimize(
            returns, cov_matrix, 0.01
        )
        
        assert isinstance(best_weights, np.ndarray)
        assert len(best_weights) == 5
        assert np.isclose(np.sum(best_weights), 1.0)
        assert isinstance(best_fitness, float)
        assert best_fitness > 0  # Sharpe deve ser positivo para retornos aleatórios
        
        # Verificar se histórico de fitness foi registrado
        assert len(ga_instance.fitness_history) == ga_instance.num_generations
        
        # Testar otimização multi-objetivo
        ga_instance.multiobjective = True
        best_weights, _, pareto_front = ga_instance.optimize(
            returns, cov_matrix, 0.01
        )
        
        assert isinstance(best_weights, np.ndarray)
        assert len(best_weights) == 5
        assert np.isclose(np.sum(best_weights), 1.0)
        assert isinstance(pareto_front, list)
        assert len(pareto_front) > 0


class TestObjectiveFunctions:
    """Testes para as funções objetivo usadas no algoritmo genético."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture que cria dados de retorno e covariância para testes."""
        # Criar dados de retorno sintéticos positivos para ter Sharpe positivo
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)  # Para reprodutibilidade
        
        returns_data = pd.DataFrame(
            np.random.normal(0.002, 0.01, size=(100, 3)),
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"]
        )
        
        # Calcular matriz de covariância
        cov_matrix = returns_data.cov()
        
        return returns_data, cov_matrix
    
    def test_sharpe_ratio_objective(self, sample_data):
        """Teste para a função objetivo de índice de Sharpe."""
        returns, cov_matrix = sample_data
        
        # Criar pesos uniformes
        weights = np.array([1/3, 1/3, 1/3])
        
        # Calcular Sharpe
        sharpe = sharpe_ratio_objective(
            weights, returns, cov_matrix, 0.01
        )
        
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Para os dados de teste, deve ser positivo
    
    def test_sortino_ratio_objective(self, sample_data):
        """Teste para a função objetivo de índice de Sortino."""
        returns, cov_matrix = sample_data
        
        # Criar pesos uniformes
        weights = np.array([1/3, 1/3, 1/3])
        
        # Calcular Sortino
        sortino = sortino_ratio_objective(
            weights, returns, cov_matrix, 0.01
        )
        
        assert isinstance(sortino, float)
        assert sortino > 0  # Para os dados de teste, deve ser positivo
