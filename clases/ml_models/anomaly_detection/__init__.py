"""
Analytics Module - Polaris Data Science Master

Este módulo contiene clases para análisis avanzados:
- AnomalyDetectionOrchestrator: Orquestador principal (ÚNICO RECOMENDADO) 
- AnomalyOptimizer: Optimización y análisis avanzado (usado por el orquestador)

NOTA: 
- Se eliminaron las clases anteriores para simplificar la estructura.
- Quarteralization y Anualizacion se movieron a clases.data_preparation
"""

# Estructura principal - USAR ESTO
from .anomaly_detection_orchestrator import AnomalyDetectionOrchestrator

# Clases de soporte (no importar directamente, el orquestador las usa)
from .anomaly_optimizer import AnomalyOptimizer

# Alias para compatibilidad y migración fácil
AnomalyDetection = AnomalyDetectionOrchestrator  # Para migración sin cambios

__all__ = [
    # Clase principal (USAR ESTA)
    'AnomalyDetectionOrchestrator',
    
    # Alias para migración fácil
    'AnomalyDetection',  
    
    # Clase de soporte (no usar directamente)
    'AnomalyOptimizer'
]