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

    
    def __init__(
        self, 
        weights: np.ndarray, 
        assets: List[str], 
        returns: pd.DataFrame, 
        cov_matrix: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        market_returns: Optional[pd.Series] = None
    ):

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

        return dict(zip(self.assets, self.weights))
    
    def get_performance_metrics(self) -> Dict[str, float]:

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

        return (self.returns * self.weights).sum(axis=1)
    
    def get_portfolio_cumulative_returns(self) -> pd.Series:

        return (1 + self.get_portfolio_returns()).cumprod() - 1
    
    def get_drawdown_series(self) -> pd.DataFrame:

        return calculate_drawdown(self.weights, self.returns)