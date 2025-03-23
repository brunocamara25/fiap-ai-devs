"""
Módulo com definições de restrições para otimização de portfólio.

Este módulo implementa funções para definir e verificar restrições aplicáveis
a portfólios de investimento, como restrições de pesos e limites de concentração.

"""
from typing import Callable, Dict, List, Optional
import numpy as np


def weights_sum_to_one(weights: np.ndarray, tolerance: float = 1e-6) -> bool:

    return abs(np.sum(weights) - 1.0) <= tolerance


def enforce_weights_sum_to_one(weights: np.ndarray) -> np.ndarray:

    if np.sum(weights) == 0:
        # Distribuir uniformemente se os pesos somarem 0
        return np.ones_like(weights) / len(weights)
    return weights / np.sum(weights)


def weights_within_bounds(weights: np.ndarray, 
                         min_weight: float = 0.0, 
                         max_weight: float = 1.0) -> bool:

    return np.all((weights >= min_weight) & (weights <= max_weight))


def enforce_weights_within_bounds(weights: np.ndarray, 
                                min_weight: float = 0.0, 
                                max_weight: float = 1.0) -> np.ndarray:

    # Primeiro, aplica os limites
    clipped_weights = np.clip(weights, min_weight, max_weight)
    
    # Em seguida, normaliza para soma = 1
    return enforce_weights_sum_to_one(clipped_weights)


def apply_all_constraints(weights: np.ndarray, 
                         constraints: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:

    adjusted_weights = weights.copy()
    
    for constraint_fn in constraints:
        adjusted_weights = constraint_fn(adjusted_weights)
    
    return adjusted_weights