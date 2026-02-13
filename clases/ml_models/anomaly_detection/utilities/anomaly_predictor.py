"""
Anomaly Predictor Utility

Clase utilitaria para realizar predicciones de anomalías.
Encapsula toda la lógica de predicción e IsolationForest.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Tuple


class AnomalyPredictor:
    """
    Utilidad para entrenar modelos IsolationForest y hacer predicciones.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializar el predictor.
        
        Parameters:
        - random_state: Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
    def fit_model(self, X_train_scaled: np.ndarray, 
                  n_estimators: int = 100,
                  contamination: float = 0.05, 
                  max_samples: str = 'auto') -> None:
        """
        Entrenar el modelo IsolationForest.
        
        Parameters:
        - X_train_scaled: Datos de entrenamiento escalados
        - n_estimators: Número de árboles
        - contamination: Proporción estimada de anomalías
        - max_samples: Número/proporción de muestras por árbol
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=self.random_state
        )
        
        self.model.fit(X_train_scaled)
        self.is_fitted = True
        
        print(f"🤖 Modelo IsolationForest entrenado:")
        print(f"   - Estimadores: {n_estimators}")
        print(f"   - Contaminación: {contamination}")
        print(f"   - Muestras de entrenamiento: {len(X_train_scaled)}")
    
    def predict_anomalies(self, X_test_scaled: np.ndarray, 
                         add_scores: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predecir anomalías en datos de prueba.
        
        Parameters:
        - X_test_scaled: Datos de prueba escalados
        - add_scores: Si calcular scores de anomalía
        
        Returns:
        - Tuple: (predictions, scores) - scores es None si add_scores=False
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit_model first.")
        
        # Predicciones (-1 = anomalía, 1 = normal)
        predictions = self.model.predict(X_test_scaled)
        
        # Scores (más alto = más anómalo)
        scores = None
        if add_scores:
            scores = -self.model.decision_function(X_test_scaled)
        
        return predictions, scores
    
    def create_result_dataframe(self, test_df: pd.DataFrame,
                               valid_mask: pd.Series,
                               predictions: np.ndarray,
                               scores: np.ndarray = None) -> pd.DataFrame:
        """
        Crear DataFrame con resultados de predicción.
        
        Parameters:
        - test_df: DataFrame original de prueba
        - valid_mask: Máscara de registros válidos
        - predictions: Array de predicciones
        - scores: Array de scores (opcional)
        
        Returns:
        - DataFrame con columnas de anomalía y scores
        """
        result_df = test_df.copy()
        
        # Inicializar columnas
        result_df['anomaly'] = False
        if scores is not None:
            result_df['anomaly_score'] = np.nan
        
        # Asignar resultados solo a registros válidos
        result_df.loc[valid_mask, 'anomaly'] = (predictions == -1)
        if scores is not None:
            result_df.loc[valid_mask, 'anomaly_score'] = scores
        
        return result_df
    
    def get_model_info(self) -> Dict:
        """
        Obtener información del modelo entrenado.
        
        Returns:
        - Dict con información del modelo
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "n_estimators": self.model.n_estimators,
            "contamination": self.model.contamination,
            "max_samples": self.model.max_samples,
            "random_state": self.random_state
        }