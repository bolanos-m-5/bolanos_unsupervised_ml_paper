"""
Data Scaler Utility

Clase utilitaria para escalado de datos.
Encapsula toda la lógica de normalización/escalado.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import List, Tuple, Optional


class DataScaler:
    """
    Utilidad para escalado de datos usando RobustScaler.
    """
    
    def __init__(self, scaler_type: str = 'robust'):
        """
        Inicializar el escalador.
        
        Parameters:
        - scaler_type: Tipo de escalador ('robust' por ahora)
        """
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Scaler type '{scaler_type}' not supported")
        
        self.is_fitted = False
    
    def fit_transform_train(self, train_df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Ajustar el escalador con datos de entrenamiento y transformarlos.
        
        Parameters:
        - train_df: DataFrame de entrenamiento
        - features: Lista de features a escalar
        
        Returns:
        - Array escalado de entrenamiento
        """
        # Obtener datos válidos para entrenamiento
        train_data = train_df[features].dropna()
        
        if len(train_data) == 0:
            raise ValueError("No valid training records after removing NaN values")
        
        # Fit y transform
        train_scaled = self.scaler.fit_transform(train_data)
        self.is_fitted = True
        
        print(f"✅ Escalador ajustado con {len(train_data)} registros válidos")
        return train_scaled
    
    def transform_test(self, test_df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, pd.Series]:
        """
        Transformar datos de prueba usando escalador ya ajustado.
        
        Parameters:
        - test_df: DataFrame de prueba
        - features: Lista de features a escalar
        
        Returns:
        - Tuple: (test_scaled, valid_mask)
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_transform_train first.")
        
        # Identificar registros válidos
        valid_mask = ~test_df[features].isna().any(axis=1)
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid test records after removing NaN values")
        
        # Transformar solo registros válidos
        test_data_valid = test_df[features][valid_mask]
        test_scaled = self.scaler.transform(test_data_valid)
        
        return test_scaled, valid_mask
    
    def get_feature_scales(self, features: List[str]) -> pd.DataFrame:
        """
        Obtener información sobre las escalas aplicadas.
        
        Parameters:
        - features: Lista de nombres de features
        
        Returns:
        - DataFrame con información de escalado
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted yet")
        
        scale_info = pd.DataFrame({
            'feature': features,
            'center': self.scaler.center_,
            'scale': self.scaler.scale_
        })
        
        return scale_info