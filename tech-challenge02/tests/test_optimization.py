"""
Testes para os módulos de otimização de portfólio.

Este módulo contém testes unitários para as funcionalidades de otimização
de portfólio, incluindo restrições, funções objetivo e a classe Portfolio.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.optimization.portfolio import Portfolio
from src.optimization.constraints import (
    weights_sum_to_one, enforce_weights_sum_to_one,
    weights_within_bounds, enforce_weights_within_bounds
)
from src.optimization.objective import (
    sharpe_ratio_objective, sortino_ratio_objective,
    treynor_ratio_objective, var_objective, 
    return_objective, volatility_objective,
    get_objective_function
)


class TestPortfolioClass:
    """Testes para a classe Portfolio."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture que cria dados de retorno e covariância para testes."""
        # Criar dados de retorno sintéticos
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)  # Para reprodutibilidade
        
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, size=(100, 3)),
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"]
        )
        
        # Calcular matriz de covariância
        cov_matrix = returns_data.cov()
        
        return returns_data, cov_matrix
    
    def test_portfolio_initialization(self, sample_data):
        """Teste para verificar a inicialização correta do Portfolio."""
        returns, cov_matrix = sample_data
        weights = np.array([0.4, 0.3, 0.3])
        assets = ["AAPL", "MSFT", "GOOGL"]
        
        # Criar portfólio
        portfolio = Portfolio(weights, assets, returns, cov_matrix, 0.01)
        
        assert np.allclose(portfolio.weights, weights)
        assert portfolio.assets == assets
        assert portfolio.risk_free_rate == 0.01
        assert portfolio.returns.shape == (100, 3)
        assert portfolio.cov_matrix.shape == (3, 3)
    
    def test_portfolio_validation(self, sample_data):
        """Teste para verificar a validação de parâmetros do Portfolio."""
        returns, cov_matrix = sample_data
        
        # Teste com número incorreto de pesos
        with pytest.raises(ValueError):
            Portfolio(
                np.array([0.5, 0.5]),  # Apenas 2 pesos para 3 ativos
                ["AAPL", "MSFT", "GOOGL"],
                returns,
                cov_matrix
            )
        
        # Teste com ativos não presentes nos retornos
        with pytest.raises(ValueError):
            Portfolio(
                np.array([0.3, 0.3, 0.4]),
                ["AAPL", "MSFT", "TSLA"],  # TSLA não está em returns
                returns,
                cov_matrix
            )
    
    def test_get_weights_dict(self, sample_data):
        """Teste para verificar o método get_weights_dict."""
        returns, cov_matrix = sample_data
        weights = np.array([0.4, 0.3, 0.3])
        assets = ["AAPL", "MSFT", "GOOGL"]
        
        portfolio = Portfolio(weights, assets, returns, cov_matrix)
        weights_dict = portfolio.get_weights_dict()
        
        assert isinstance(weights_dict, dict)
        assert len(weights_dict) == 3
        assert "AAPL" in weights_dict
        assert np.isclose(weights_dict["AAPL"], 0.4)
        assert np.isclose(weights_dict["MSFT"], 0.3)
        assert np.isclose(weights_dict["GOOGL"], 0.3)
    
    def test_get_performance_metrics(self, sample_data):
        """Teste para verificar o método get_performance_metrics."""
        returns, cov_matrix = sample_data
        weights = np.array([0.4, 0.3, 0.3])
        assets = ["AAPL", "MSFT", "GOOGL"]
        
        portfolio = Portfolio(weights, assets, returns, cov_matrix, 0.01)
        metrics = portfolio.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "return" in metrics
        assert "volatility" in metrics
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert isinstance(metrics["return"], float)
        assert isinstance(metrics["volatility"], float)
        assert metrics["volatility"] > 0
    
    def test_get_risk_metrics(self, sample_data):
        """Teste para verificar o método get_risk_metrics."""
        returns, cov_matrix = sample_data
        weights = np.array([0.4, 0.3, 0.3])
        assets = ["AAPL", "MSFT", "GOOGL"]
        
        portfolio = Portfolio(weights, assets, returns, cov_matrix)
        metrics = portfolio.get_risk_metrics()
        
        assert isinstance(metrics, dict)
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "max_drawdown" in metrics
        assert "diversification" in metrics
        assert isinstance(metrics["var_95"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert metrics["var_95"] > 0
        assert metrics["max_drawdown"] > 0
    
    def test_get_portfolio_returns(self, sample_data):
        """Teste para verificar o método get_portfolio_returns."""
        returns, cov_matrix = sample_data
        weights = np.array([0.4, 0.3, 0.3])
        assets = ["AAPL", "MSFT", "GOOGL"]
        
        portfolio = Portfolio(weights, assets, returns, cov_matrix)
        portfolio_returns = portfolio.get_portfolio_returns()
        
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(returns)
        
        # Verificar cálculo manual para o primeiro retorno
        expected_first_return = np.dot(weights, returns.iloc[0])
        assert np.isclose(portfolio_returns.iloc[0], expected_first_return)


class TestConstraints:
    """Testes para as funções de restrição."""
    
    def test_weights_sum_to_one(self):
        """Teste para a função weights_sum_to_one."""
        # Caso válido
        weights1 = np.array([0.3, 0.4, 0.3])
        assert weights_sum_to_one(weights1)
        
        # Caso inválido
        weights2 = np.array([0.3, 0.3, 0.3])  # Soma 0.9
        assert not weights_sum_to_one(weights2)
    
    def test_enforce_weights_sum_to_one(self):
        """Teste para a função enforce_weights_sum_to_one."""
        # Caso que já soma 1
        weights1 = np.array([0.3, 0.4, 0.3])
        result1 = enforce_weights_sum_to_one(weights1)
        assert np.allclose(result1, weights1)
        
        # Caso que não soma 1
        weights2 = np.array([0.3, 0.3, 0.3])  # Soma 0.9
        result2 = enforce_weights_sum_to_one(weights2)
        assert np.isclose(np.sum(result2), 1.0)
        
        # Pesos negativos
        weights3 = np.array([0.5, -0.2, 0.3])
        result3 = enforce_weights_sum_to_one(weights3)
        assert np.isclose(np.sum(result3), 1.0)
    
    def test_weights_within_bounds(self):
        """Teste para a função weights_within_bounds."""
        min_weight = 0.1
        max_weight = 0.5
        
        # Caso válido
        weights1 = np.array([0.3, 0.4, 0.3])
        assert weights_within_bounds(weights1, min_weight, max_weight)
        
        # Caso inválido - abaixo do mínimo
        weights2 = np.array([0.05, 0.45, 0.5])
        assert not weights_within_bounds(weights2, min_weight, max_weight)
        
        # Caso inválido - acima do máximo
        weights3 = np.array([0.6, 0.2, 0.2])
        assert not weights_within_bounds(weights3, min_weight, max_weight)
    
    def test_enforce_weights_within_bounds(self):
        """Teste para a função enforce_weights_within_bounds."""
        min_weight = 0.1
        max_weight = 0.5
        
        # Caso que já está dentro dos limites
        weights1 = np.array([0.3, 0.4, 0.3])
        result1 = enforce_weights_within_bounds(weights1, min_weight, max_weight)
        assert np.allclose(result1, weights1)
        
        # Caso com peso abaixo do mínimo
        weights2 = np.array([0.05, 0.45, 0.5])
        result2 = enforce_weights_within_bounds(weights2, min_weight, max_weight)
        assert np.all(result2 >= min_weight)
        assert np.all(result2 <= max_weight)
        assert np.isclose(np.sum(result2), 1.0)
        
        # Caso com peso acima do máximo
        weights3 = np.array([0.6, 0.2, 0.2])
        result3 = enforce_weights_within_bounds(weights3, min_weight, max_weight)
        assert np.all(result3 >= min_weight)
        assert np.all(result3 <= max_weight)
        assert np.isclose(np.sum(result3), 1.0)


class TestObjectiveFunctions:
    """Testes para as funções objetivo."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture que cria dados de retorno e covariância para testes."""
        # Criar dados de retorno sintéticos positivos para ter Sharpe positivo
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)  # Para reprodutibilidade
        
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.01, size=(100, 3)),
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL"]
        )
        
        # Criar dados de mercado para cálculo do beta
        market_returns = pd.Series(
            np.random.normal(0.0005, 0.015, size=100),
            index=dates
        )
        
        # Calcular matriz de covariância
        cov_matrix = returns_data.cov()
        
        return returns_data, cov_matrix, market_returns
    
    def test_objective_functions(self, sample_data):
        """Teste para verificar todas as funções objetivo."""
        returns, cov_matrix, market_returns = sample_data
        weights = np.array([1/3, 1/3, 1/3])
        risk_free_rate = 0.01
        
        # Testar cada função objetivo
        sharpe = sharpe_ratio_objective(weights, returns, cov_matrix, risk_free_rate)
        sortino = sortino_ratio_objective(weights, returns, risk_free_rate)
        treynor = treynor_ratio_objective(weights, returns, market_returns, risk_free_rate)
        var = var_objective(weights, returns)
        ret = return_objective(weights, returns)
        vol = volatility_objective(weights, cov_matrix)
        
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(treynor, float)
        assert isinstance(var, float)
        assert isinstance(ret, float)
        assert isinstance(vol, float)
    
    def test_get_objective_function(self, sample_data):
        """Teste para verificar a função get_objective_function."""
        returns, cov_matrix, market_returns = sample_data
        
        # Testar diferentes métodos de avaliação
        sharpe_func = get_objective_function("sharpe")
        sortino_func = get_objective_function("sortino")
        treynor_func = get_objective_function("treynor")
        var_func = get_objective_function("var")
        
        weights = np.array([1/3, 1/3, 1/3])
        
        # Verificar se as funções funcionam
        sharpe_value = sharpe_func(weights, returns, cov_matrix, 0.01)
        sortino_value = sortino_func(weights, returns, 0.01)
        treynor_value = treynor_func(weights, returns, market_returns, 0.01)
        var_value = var_func(weights, returns)
        
        assert isinstance(sharpe_value, float)
        assert isinstance(sortino_value, float)
        assert isinstance(treynor_value, float)
        assert isinstance(var_value, float)
        
        # Testar método inválido
        with pytest.raises(ValueError):
            get_objective_function("invalid_method")
