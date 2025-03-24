"""
Configurações do projeto de otimização de portfólio.

Este módulo contém configurações globais, constantes e parâmetros padrão
utilizados em todo o projeto de otimização de portfólio.
"""
import os
from pathlib import Path

# Configurações de diretórios
# Detecta o diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

# Garantir que os diretórios existam
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configurações do algoritmo genético
DEFAULT_POPULATION_SIZE = 100
DEFAULT_NUM_GENERATIONS = 50
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_ELITISM_COUNT = 2
DEFAULT_MIN_WEIGHT = 0.01
DEFAULT_MAX_WEIGHT = 0.4

# Parâmetros de otimização
DEFAULT_RISK_FREE_RATE = 0.01  # 1% anual
DEFAULT_EVALUATION_METHODS = ["sharpe", "sortino", "treynor", "var", "multi"]
DEFAULT_INIT_STRATEGIES = ["random", "uniform", "diversified"]
DEFAULT_SELECTION_METHODS = ["tournament", "roulette", "rank"]
DEFAULT_CROSSOVER_METHODS = ["uniform", "single_point", "blend"]
DEFAULT_MUTATION_DISTRIBUTIONS = ["normal", "uniform"]

# Configurações de visualização
COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]
ACCENT_COLOR = "#2E5090"  # Azul para destaque
RISK_COLOR = "#D62728"    # Vermelho para risco
RETURN_COLOR = "#2CA02C"  # Verde para retorno

# Configurações de mercado
DEFAULT_BENCHMARK = "^BVSP"  # Ibovespa como benchmark padrão
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2023-12-31"

# Tickers populares para seleção rápida
POPULAR_BR_TICKERS = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
    "B3SA3.SA", "WEGE3.SA", "RENT3.SA", "BBAS3.SA", "MGLU3.SA",
    "RADL3.SA", "SUZB3.SA", "JBSS3.SA", "LREN3.SA", "ITSA4.SA"
]

POPULAR_US_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "DIS", "BAC", "NFLX"
]

# Configurações de sessão Streamlit
STREAMLIT_PAGE_TITLE = "Otimização de Portfólio com Algoritmos Genéticos"
STREAMLIT_LAYOUT = "wide"
STREAMLIT_SIDEBAR_WIDTH = 350
