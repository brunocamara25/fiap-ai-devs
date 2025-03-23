"""
Módulo com definições de restrições para otimização de portfólio.

Este módulo implementa funções para definir e verificar restrições aplicáveis
a portfólios de investimento, como restrições de pesos e limites de concentração.

"""
from typing import Callable, Dict, List, Optional
import numpy as np


def weights_sum_to_one(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Verifica se a soma dos pesos é aproximadamente igual a 1.
    
    .. math::
        |\\sum_i w_i - 1| \\leq \\text{tolerance}
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    tolerance : float, optional
        Tolerância para a diferença entre a soma e 1.
        
    Returns
    -------
    bool
        True se a soma estiver dentro da tolerância, False caso contrário.
    """
    return abs(np.sum(weights) - 1.0) <= tolerance


def enforce_weights_sum_to_one(weights: np.ndarray) -> np.ndarray:
    """
    Normaliza os pesos para que somem 1.
    
    .. math::
        w_i' = \\frac{w_i}{\\sum_j w_j}
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
        
    Returns
    -------
    np.ndarray
        Array normalizado de pesos.
    """
    if np.sum(weights) == 0:
        # Distribuir uniformemente se os pesos somarem 0
        return np.ones_like(weights) / len(weights)
    return weights / np.sum(weights)


def weights_within_bounds(weights: np.ndarray,
                         min_weight: float = 0.0,
                         max_weight: float = 1.0) -> bool:
    """
    Verifica se todos os pesos estão dentro dos limites especificados.
    
    .. math::
        \\min\_weight \\leq w_i \\leq \\max\_weight, \\forall i
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    min_weight : float, optional
        Peso mínimo permitido para cada ativo.
    max_weight : float, optional
        Peso máximo permitido para cada ativo.
        
    Returns
    -------
    bool
        True se todos os pesos estiverem dentro dos limites, False caso contrário.
    """
    return np.all((weights >= min_weight) & (weights <= max_weight))


def enforce_weights_within_bounds(weights: np.ndarray,
                                min_weight: float = 0.0,
                                max_weight: float = 1.0) -> np.ndarray:
    """
    Ajusta os pesos para que estejam dentro dos limites especificados.
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    min_weight : float, optional
        Peso mínimo permitido para cada ativo.
    max_weight : float, optional
        Peso máximo permitido para cada ativo.
        
    Returns
    -------
    np.ndarray
        Array de pesos ajustados.
    """
    # Primeiro, aplica os limites
    clipped_weights = np.clip(weights, min_weight, max_weight)

    # Em seguida, normaliza para soma = 1
    return enforce_weights_sum_to_one(clipped_weights)


def apply_all_constraints(weights: np.ndarray,
                         constraints: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """
    Aplica sequencialmente todas as restrições fornecidas.
    
    Parameters
    ----------
    weights : np.ndarray
        Array com os pesos dos ativos.
    constraints : List[Callable[[np.ndarray], np.ndarray]]
        Lista de funções de restrição a serem aplicadas.
        
    Returns
    -------
    np.ndarray
        Array de pesos ajustados após aplicar todas as restrições.
    """
    adjusted_weights = weights.copy()

    for constraint_fn in constraints:
        adjusted_weights = constraint_fn(adjusted_weights)

    return adjusted_weights
