"""
Módulo para processamento e transformação de dados financeiros.
Contém funções para cálculo de retornos, tratamento de outliers,
e análise estatística básica.
"""
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
from scipy import stats

# Configuração de logging
logger = logging.getLogger(__name__)


def prepare_returns(
    data: pd.DataFrame,
    method: str = 'log',
    periods: int = 1,
    remove_outliers: bool = True
) -> pd.DataFrame:
    """
    Calcula os retornos a partir dos preços ajustados.
    
    Parameters
    ----------
    data : pd.DataFrame
        DataFrame com os preços ajustados das ações.
    method : str, default='log'
        Método de cálculo dos retornos. Opções: 'simple', 'log'.
    periods : int, default=1
        Número de períodos para cálculo dos retornos.
    remove_outliers : bool, default=True
        Se True, remove outliers dos retornos.
        
    Returns
    -------
    pd.DataFrame
        DataFrame com os retornos calculados.
    """
    if data is None or data.empty:
        logger.warning("Dados vazios fornecidos para cálculo de retornos")
        return pd.DataFrame()

    # Calcular retornos
    if method.lower() == 'log':
        # Retornos logarítmicos: r_t = ln(P_t / P_{t-1})
        returns = np.log(data).diff(periods=periods)
    else:
        # Retornos simples: r_t = (P_t / P_{t-1}) - 1
        returns = data.pct_change(periods=periods)

    # Remover primeiras linhas com NaN
    returns = returns.dropna()

    # Remover outliers se solicitado
    if remove_outliers:
        returns = remove_return_outliers(returns)

    return returns


def remove_return_outliers(
    returns: pd.DataFrame,
    std_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers nos retornos baseado em desvios padrão.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos.
    std_threshold : float, default=3.0
        Limiar em desvios padrão para considerar um valor como outlier.
        
    Returns
    -------
    pd.DataFrame
        DataFrame com os retornos sem outliers.
    """
    cleaned_returns = returns.copy()

    for col in cleaned_returns.columns:
        series = cleaned_returns[col]
        mean, std = series.mean(), series.std()
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std

        # Identificar outliers
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.info("Encontrados %d outliers em %s", outlier_count, col)

            # Limitar valores extremos (winsorização)
            cleaned_returns.loc[series < lower_bound, col] = lower_bound
            cleaned_returns.loc[series > upper_bound, col] = upper_bound

    return cleaned_returns


def calculate_cov_matrix(
    returns: pd.DataFrame,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calcula a matriz de covariância dos retornos.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários.
    annualize : bool, default=True
        Se True, anualiza a matriz de covariância.
    trading_days : int, default=252
        Número de dias de negociação no ano.
        
    Returns
    -------
    pd.DataFrame
        Matriz de covariância.
    """
    if returns is None or returns.empty:
        logger.warning("Dados vazios fornecidos para cálculo de covariância")
        return pd.DataFrame()

    # Calcular matriz de covariância
    cov_matrix = returns.cov()

    # Anualizar a matriz de covariância, se solicitado
    if annualize:
        cov_matrix = cov_matrix * trading_days

    return cov_matrix


def calculate_corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a matriz de correlação dos retornos.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários.
        
    Returns
    -------
    pd.DataFrame
        Matriz de correlação.
    """
    if returns is None or returns.empty:
        logger.warning("Dados vazios fornecidos para cálculo de correlação")
        return pd.DataFrame()

    return returns.corr()


def split_train_test(
    returns: pd.DataFrame,
    train_size: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide os dados em conjuntos de treinamento e teste.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários.
    train_size : float, default=0.7
        Proporção dos dados para treinamento (0 a 1).
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_data, test_data) - DataFrames com dados de treinamento e teste.
    """
    if returns is None or returns.empty:
        logger.warning("Dados vazios fornecidos para divisão treino/teste")
        return pd.DataFrame(), pd.DataFrame()

    # Divisão sequencial (preserva ordem temporal)
    train_size_idx = int(train_size * len(returns))
    train_data = returns.iloc[:train_size_idx]
    test_data = returns.iloc[train_size_idx:]

    logger.info("Divisão treino/teste: %d treino, %d teste", len(train_data), len(test_data))

    return train_data, test_data


def calculate_financial_ratios(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores financeiros estatísticos para cada ativo.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame com os retornos diários.
        
    Returns
    -------
    pd.DataFrame
        DataFrame com indicadores financeiros por ativo.
    """
    if returns is None or returns.empty:
        logger.warning("Dados vazios fornecidos para cálculo de indicadores")
        return pd.DataFrame()

    # Calcular estatísticas por ativo
    stats_data = []
    for col in returns.columns:
        series = returns[col].dropna()

        # Estatísticas básicas
        mean_return = series.mean() * 252  # Anualizado
        volatility = series.std() * np.sqrt(252)  # Anualizada
        sharpe = mean_return / volatility if volatility > 0 else 0

        # Estatísticas de formato da distribuição
        skewness = series.skew()
        kurtosis = series.kurtosis()

        # Estatísticas de risco
        var_95 = np.percentile(series, 5)  # VaR 95%

        # Métricas de drawdown
        cumulative = (1 + series).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        stats_data.append({
            'ativo': col,
            'retorno_anual': mean_return,
            'volatilidade_anual': volatility,
            'sharpe_ratio': sharpe,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'VaR_95': var_95,
            'max_drawdown': max_drawdown
        })

    return pd.DataFrame(stats_data).set_index('ativo')
