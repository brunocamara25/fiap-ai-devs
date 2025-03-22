"""
Módulo para processamento e transformação de dados financeiros.
"""
import pandas as pd
import numpy as np

def prepare_returns(data):
    """
    Calcular os retornos diários a partir dos preços ajustados.
    
    Parâmetros:
        data (pd.DataFrame): DataFrame com os preços ajustados das ações.
        
    Retorna:
        pd.DataFrame: DataFrame com os retornos diários.
    """
    returns = data.pct_change().dropna()
    return returns

def calculate_cov_matrix(returns):
    """
    Calcular a matriz de covariância dos retornos.
    
    Parâmetros:
        returns (pd.DataFrame): DataFrame com os retornos diários.
        
    Retorna:
        pd.DataFrame: Matriz de covariância.
    """
    return returns.cov()

def calculate_corr_matrix(returns):
    """
    Calcular a matriz de correlação dos retornos.
    
    Parâmetros:
        returns (pd.DataFrame): DataFrame com os retornos diários.
        
    Retorna:
        pd.DataFrame: Matriz de correlação.
    """
    return returns.corr()

def split_train_test(returns, train_size=0.7):
    """
    Dividir os dados em conjuntos de treinamento e teste.
    
    Parâmetros:
        returns (pd.DataFrame): DataFrame com os retornos diários.
        train_size (float): Proporção dos dados para treinamento (0 a 1).
        
    Retorna:
        tuple: (train_data, test_data) - DataFrames com dados de treinamento e teste.
    """
    train_size = int(train_size * len(returns))
    train_data = returns.iloc[:train_size]
    test_data = returns.iloc[train_size:]
    return train_data, test_data 