"""
Módulo com implementação da classe Portfolio para representação e análise de portfólios.

Este módulo define a classe principal que representa um portfólio de investimentos
e implementa métodos para avaliação de performance, exposição de ativos, 
risco e outras métricas relevantes.

"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.metrics.performance import (
    calculate_metrics, calculate_information_ratio, calculate_sortino_ratio,
    calculate_calmar_ratio, calculate_treynor_ratio
)
from src.metrics.risk import (
    calculate_volatility, calculate_var, calculate_cvar, calculate_drawdown,
    calculate_diversification_ratio
)


class Portfolio:
    """
    Classe que representa um portfólio de investimentos com métodos para análise e avaliação.
    
    A classe Portfolio encapsula os dados e operações relacionadas a um portfólio de investimentos,
    fornecendo métodos para avaliação de performance, exposição de ativos, risco e retorno.
    
    Attributes:
        weights (np.ndarray): Pesos dos ativos no portfólio.
        assets (List[str]): Lista de identificadores dos ativos.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        risk_free_rate (float): Taxa livre de risco anualizada.
        market_returns (Optional[pd.Series]): Retornos do mercado de referência.
    """

    def __init__(
        self,
        weights: np.ndarray,
        assets: List[str],
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        market_returns: Optional[pd.Series] = None
    ):
        """
        Inicializa um objeto Portfolio com os parâmetros fornecidos.
        
        Parameters
        ----------
        weights : np.ndarray
            Pesos dos ativos no portfólio.
        assets : List[str]
            Lista de identificadores dos ativos.
        returns : pd.DataFrame
            Retornos históricos dos ativos.
        cov_matrix : pd.DataFrame, optional
            Matriz de covariância dos retornos. Se None, será calculada.
        risk_free_rate : float, optional
            Taxa livre de risco anualizada.
        market_returns : pd.Series, optional
            Retornos do mercado de referência.
        
        Raises
        ------
        ValueError
            Se os tamanhos de weights e assets não forem compatíveis
            ou se os assets não estiverem presentes em returns.
        """
        if len(weights) != len(assets):
            raise ValueError("O número de pesos deve ser igual ao número de ativos.")

        if not all(asset in returns.columns for asset in assets):
            raise ValueError("Todos os ativos devem estar presentes nos dados de retorno.")

        # Normaliza os pesos para somar 1
        self.weights = weights / np.sum(weights)
        self.assets = assets
        self.returns = returns[assets]

        if cov_matrix is None:
            self.cov_matrix = self.returns.cov()
        else:
            if not all(asset in cov_matrix.columns for asset in assets):
                raise ValueError("Todos os ativos devem estar presentes na matriz de covariância.")
            self.cov_matrix = cov_matrix.loc[assets, assets]

        self.risk_free_rate = risk_free_rate
        self.market_returns = market_returns

    def __repr__(self) -> str:
        """Retorna uma representação em string do objeto Portfolio."""
        performance = self.get_performance_metrics()
        return (
            f"Portfolio(assets={len(self.assets)}, "
            f"return={performance['return']:.2%}, "
            f"risk={performance['volatility']:.2%}, "
            f"sharpe={performance['sharpe']:.2f})"
        )

    def get_weights_dict(self) -> Dict[str, float]:
        """
        Retorna um dicionário com os pesos de cada ativo.
        
        Returns
        -------
        Dict[str, float]
            Dicionário com pares {ativo: peso}.
        """
        return dict(zip(self.assets, self.weights))

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calcula e retorna as principais métricas de desempenho do portfólio.
        
        Returns
        -------
        Dict[str, float]
            Dicionário com as métricas calculadas.
        """
        portfolio_return, portfolio_vol, sharpe_ratio = calculate_metrics(
            self.weights, self.returns, self.cov_matrix, self.risk_free_rate
        )

        metrics = {
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe': sharpe_ratio
        }

        # Calcular métricas adicionais se possível
        if self.market_returns is not None:
            metrics['treynor'] = calculate_treynor_ratio(
                self.weights, self.returns, self.market_returns, 
                self.risk_free_rate
            )
            metrics['information_ratio'] = calculate_information_ratio(
                self.weights, self.returns, self.market_returns
            )

        metrics['sortino'] = calculate_sortino_ratio(
            self.weights, self.returns, self.risk_free_rate
        )

        metrics['calmar'] = calculate_calmar_ratio(
            self.weights, self.returns, self.risk_free_rate
        )

        return metrics

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Calcula e retorna as principais métricas de risco do portfólio.
        
        Returns
        -------
        Dict[str, float]
            Dicionário com as métricas de risco calculadas.
        """
        # Criar série de retornos do portfólio
        portfolio_returns = (self.returns * self.weights).sum(axis=1)

        metrics = {
            'volatility': calculate_volatility(self.weights, self.cov_matrix),
            'var_95': calculate_var(self.weights, self.returns, confidence_level=0.95),
            'cvar_95': calculate_cvar(self.weights, self.returns, confidence_level=0.95),
            'max_drawdown': calculate_drawdown(self.weights, self.returns)['max_drawdown'],
            'diversification_ratio': calculate_diversification_ratio(self.weights, self.cov_matrix)
        }

        # Adicionar métricas de cauda
        returns_array = portfolio_returns.values
        metrics['skewness'] = pd.Series(returns_array).skew()
        metrics['kurtosis'] = pd.Series(returns_array).kurtosis()

        return metrics

    def get_portfolio_returns(self) -> pd.Series:
        """
        Calcula a série temporal de retornos do portfólio.
        
        Returns
        -------
        pd.Series
            Série de retornos diários do portfólio.
        """
        return (self.returns * self.weights).sum(axis=1)

    def get_portfolio_cumulative_returns(self) -> pd.Series:
        """
        Calcula a série temporal de retornos cumulativos do portfólio.
        
        Returns
        -------
        pd.Series
            Série de retornos cumulativos do portfólio.
        """
        return (1 + self.get_portfolio_returns()).cumprod() - 1

    def get_drawdown_series(self) -> pd.DataFrame:
        """
        Calcula a série de drawdowns do portfólio ao longo do tempo.
        
        Returns
        -------
        pd.DataFrame
            DataFrame com série de drawdown.
        """
        return calculate_drawdown(self.weights, self.returns)
