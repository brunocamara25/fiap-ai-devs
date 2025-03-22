"""
Pacote data para carregamento e processamento de dados.
"""
from src.data.loader import download_data, get_risk_free_rate
from src.data.processor import (
    prepare_returns,
    remove_return_outliers,
    calculate_cov_matrix,
    calculate_corr_matrix,
    split_train_test,
    calculate_financial_ratios
)
