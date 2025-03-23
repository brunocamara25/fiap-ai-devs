"""
Pacote models para modelos de otimização de portfólio.

Este pacote inclui implementações de diversos algoritmos de otimização,
com foco principal no algoritmo genético para seleção de portfólios.
"""
from src.models.genetic_algorithm import (
    GeneticAlgorithm,
    optimize_portfolio
)

__all__ = [
    'GeneticAlgorithm',
    'optimize_portfolio'
]
