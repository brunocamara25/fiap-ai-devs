"""
Testes para os módulos de processamento e carregamento de dados.

Este módulo contém testes unitários para as funcionalidades de carregamento
e processamento de dados financeiros.
"""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.data.loader import download_data, get_cache_path, get_risk_free_rate
from src.data.processor import (
    prepare_returns as calculate_returns,
    calculate_cov_matrix as calculate_covariance_matrix,
    remove_return_outliers as clean_data
)


class TestDataLoader:
    """Testes para o módulo de carregamento de dados."""

    def test_get_cache_path(self):
        """Teste para a função get_cache_path."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        
        cache_path = get_cache_path(tickers, start_date, end_date)
        
        assert isinstance(cache_path, Path)
        assert str(cache_path).endswith(".parquet")
        
        # Verificar se o mesmo input produz o mesmo path
        cache_path2 = get_cache_path(tickers, start_date, end_date)
        assert cache_path == cache_path2
        
        # Verificar se a ordem dos tickers não altera o path
        cache_path3 = get_cache_path(["MSFT", "GOOGL", "AAPL"], start_date, end_date)
        assert cache_path == cache_path3
    
    def test_download_data_invalid_inputs(self):
        """Teste para a função download_data com inputs inválidos."""
        # Lista vazia de tickers
        result = download_data([], "2020-01-01", "2020-12-31")
        assert result is None
        
        # Formato inválido de data
        result = download_data(["AAPL"], "2020/01/01", "2020-12-31")
        assert result is None
    
    @pytest.mark.skipif(os.environ.get("SKIP_ONLINE_TESTS"), reason="Pula testes online")
    def test_download_data_integration(self):
        """Teste de integração para download_data com API real."""
        tickers = ["AAPL"]
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        result = download_data(tickers, start_date, end_date, use_cache=False)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "AAPL" in result.columns


class TestDataProcessor:
    """Testes para o módulo de processamento de dados."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Fixture que cria dados de preço de exemplo para os testes."""
        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100, 102, 104, 103, 105, 107, 108, 107, 109, 110],
            "MSFT": [200, 201, 203, 202, 205, 204, 207, 208, 210, 211]
        }, index=dates)
        return prices
    
    def test_calculate_returns(self, sample_price_data):
        """Teste para a função calculate_returns."""
        returns = calculate_returns(sample_price_data, method='simple')  # Usando method='simple' para corresponder ao cálculo esperado
        
        assert isinstance(returns, pd.DataFrame)
        assert returns.shape[0] == sample_price_data.shape[0] - 1  # -1 pois há perda da primeira linha
        assert returns.shape[1] == sample_price_data.shape[1]
        
        # Verificar cálculo manual para o primeiro retorno
        expected_return_aapl = (102 / 100) - 1
        expected_return_msft = (201 / 200) - 1
        assert returns.iloc[0, 0] == pytest.approx(expected_return_aapl)
        assert returns.iloc[0, 1] == pytest.approx(expected_return_msft)
    
    def test_calculate_covariance_matrix(self, sample_price_data):
        """Teste para a função calculate_covariance_matrix."""
        returns = calculate_returns(sample_price_data, method='simple')
        cov_matrix = calculate_covariance_matrix(returns)
        
        assert isinstance(cov_matrix, pd.DataFrame)
        assert cov_matrix.shape == (2, 2)
        assert cov_matrix.index.tolist() == returns.columns.tolist()
        assert cov_matrix.columns.tolist() == returns.columns.tolist()
        
        # A diagonal da matriz deve conter a variância (sempre positiva)
        assert cov_matrix.iloc[0, 0] > 0
        assert cov_matrix.iloc[1, 1] > 0
        
        # A matriz deve ser simétrica
        assert cov_matrix.iloc[0, 1] == cov_matrix.iloc[1, 0]
    
    def test_clean_data(self, sample_price_data):
        """Teste para a função clean_data."""
        # Adicionar alguns dados faltantes
        dirty_data = sample_price_data.copy()
        dirty_data.iloc[3, 0] = np.nan
        dirty_data.iloc[5, 1] = np.nan
        
        # Preparar retornos para então limpar outliers
        returns = calculate_returns(dirty_data.fillna(method='ffill'), method='simple')
        cleaned_data = clean_data(returns)
        
        assert isinstance(cleaned_data, pd.DataFrame)
        assert cleaned_data.shape == returns.shape
        assert not cleaned_data.isna().any().any()  # Não deve ter valores NaN
