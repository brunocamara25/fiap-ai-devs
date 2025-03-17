import numpy as np

def normalize_fitness_scores(fitness_scores):
    """Normalizar os scores de fitness"""
    if isinstance(fitness_scores[0], (int, float)):  # Caso não seja multiobjetivo
        return fitness_scores  # Retorna os scores diretamente, sem normalização
    returns = np.array([score[0] for score in fitness_scores])
    risks = np.array([score[1] for score in fitness_scores])
    norm_returns = (returns - returns.min()) / (returns.max() - returns.min())
    norm_risks = (risks - risks.min()) / (risks.max() - risks.min())
    return list(zip(norm_returns, norm_risks))