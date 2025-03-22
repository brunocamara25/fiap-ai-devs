"""
Módulo para carregamento e download de dados financeiros.
Contém funções para obter dados de APIs financeiras com gerenciamento de cache
e tratamento de erros.
"""
from typing import List, Optional
import os
import logging
from pathlib import Path
import datetime as dt
import hashlib

import yfinance as yf
import pandas as pd
import numpy as np

# Configuração de logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Diretório de cache para dados financeiros
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(tickers: List[str], start_date: str, end_date: str) -> Path:
    """
    Gera um caminho único para o arquivo de cache baseado nos parâmetros.
    
    Parameters
    ----------
    tickers : List[str]
        Lista de códigos de ações.
    start_date : str
        Data inicial no formato YYYY-MM-DD.
    end_date : str
        Data final no formato YYYY-MM-DD.
        
    Returns
    -------
    Path
        Caminho para o arquivo de cache.
    """
    params = f"{'-'.join(sorted(tickers))}-{start_date}-{end_date}"
    params_hash = hashlib.md5(params.encode()).hexdigest()
    return CACHE_DIR / f"data_{params_hash}.parquet"


def download_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    Baixa dados financeiros das ações especificadas com gerenciamento de cache.
    
    Parameters
    ----------
    tickers : List[str]
        Lista de códigos de ações.
    start_date : str
        Data inicial no formato YYYY-MM-DD.
    end_date : str
        Data final no formato YYYY-MM-DD.
    use_cache : bool, default=True
        Se True, tenta carregar dados do cache antes de baixar.
        
    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame com os preços ajustados das ações ou None se ocorrer erro.
    """
    if not tickers:
        logger.warning("Lista de tickers está vazia")
        return None

    # Verificar datas
    try:
        dt.datetime.strptime(start_date, "%Y-%m-%d")
        dt.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error("Formato de data inválido: %s", e)
        return None

    # Tenta carregar do cache
    cache_path = get_cache_path(tickers, start_date, end_date)
    if use_cache and cache_path.exists():
        try:
            logger.info("Carregando dados do cache: %s", cache_path)
            return pd.read_parquet(cache_path)
        except (IOError, OSError, pd.errors.EmptyDataError) as e:
            logger.warning("Erro ao carregar cache: %s", e)

    # Download dos dados
    try:
        logger.info("Baixando dados para %d ações de %s até %s", len(tickers), start_date, end_date)
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False
        )

        # Lidar com diferentes formatos de retorno do yfinance
        if 'Adj Close' in data.columns:
            data = data['Adj Close']

        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
            data.columns = [tickers[0]]

        # Verificar se obtivemos dados
        if data.empty:
            logger.warning("Nenhum dado retornado para os tickers especificados")
            return None

        # Preencher valores ausentes
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Salvar no cache
        if use_cache and not data.empty:
            try:
                data.to_parquet(cache_path)
                logger.info("Dados salvos no cache: %s", cache_path)
            except (IOError, OSError) as e:
                logger.error("Erro ao salvar cache: %s", e)

        return data

    except (IOError, ConnectionError, ValueError) as e:
        logger.error("Erro ao baixar dados: %s", e)
        return None


def get_risk_free_rate() -> float:
    """
    Retorna uma taxa livre de risco padrão (Selic aproximada).
    
    Returns
    -------
    float
        Taxa livre de risco anual.
    """
    # Taxa Selic média aproximada
    return 0.100  # 10% ao ano
