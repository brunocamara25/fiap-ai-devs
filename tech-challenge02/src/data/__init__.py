"""
Pacote data para carregamento e processamento de dados.
"""
from src.data.loader import download_data
from src.data.processor import (
    prepare_returns,
    calculate_cov_matrix,
    calculate_corr_matrix,
    split_train_test
)
