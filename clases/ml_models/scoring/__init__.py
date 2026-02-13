"""
Scoring Module - Polaris Data Science Master

Este módulo contiene clases para cálculo de scores de negocio:
- Score: Calculador de NOS Score básico
- ScoreDynamic: Calculador dinámico con múltiples dimensiones
"""

from .NosScore import Score, ScoreDynamic

__all__ = ['Score', 'ScoreDynamic']