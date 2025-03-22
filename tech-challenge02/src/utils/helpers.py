"""
Módulo com funções utilitárias para o projeto de otimização de portfólio.
"""
import numpy as np

def normalize_fitness_scores(fitness_scores):
    """
    Normaliza os scores de fitness para escalar os valores entre 0 e 1.
    
    Parâmetros:
        fitness_scores (list): Lista de scores de fitness ou tuplas (retorno, risco) para abordagem multiobjetivo.
        
    Retorna:
        list: Scores de fitness normalizados.
    """
    if isinstance(fitness_scores[0], (int, float)):  # Caso não seja multiobjetivo
        return fitness_scores  # Retorna os scores diretamente, sem normalização
        
    # Caso seja multiobjetivo (tuplas de retorno e risco)
    returns = np.array([score[0] for score in fitness_scores])
    risks = np.array([score[1] for score in fitness_scores])
    
    # Evitar divisão por zero
    returns_range = returns.max() - returns.min()
    risks_range = risks.max() - risks.min()
    
    if returns_range < 1e-8:
        norm_returns = np.zeros_like(returns)
    else:
        norm_returns = (returns - returns.min()) / returns_range
        
    if risks_range < 1e-8:
        norm_risks = np.zeros_like(risks)
    else:
        norm_risks = (risks - risks.min()) / risks_range
        
    return list(zip(norm_returns, norm_risks)) 