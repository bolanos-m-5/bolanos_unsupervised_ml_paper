"""
Data Validator Utility

Clase utilitaria para validación y limpieza de datos.
Separada del core para mantener la responsabilidad única.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


class DataValidator:
    """
    Utilidad para validar y limpiar datos antes del análisis de anomalías.
    """
    
    @staticmethod
    def validate_features(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> None:
        """
        Validar que las features existan en ambos DataFrames.
        
        Parameters:
        - train_df: DataFrame de entrenamiento
        - test_df: DataFrame de prueba  
        - features: Lista de nombres de columnas
        
        Raises:
        - ValueError: Si alguna feature no existe
        """
        missing_train = [f for f in features if f not in train_df.columns]
        missing_test = [f for f in features if f not in test_df.columns]
        
        if missing_train:
            raise ValueError(f"Columns {missing_train} not found in train_df.")
        if missing_test:
            raise ValueError(f"Columns {missing_test} not found in test_df.")
    
    @staticmethod
    def clean_infinite_values(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Limpiar valores infinitos reemplazándolos con NaN.
        
        Parameters:
        - df: DataFrame a limpiar
        - features: Lista de columnas a limpiar
        
        Returns:
        - DataFrame limpio
        """
        df_clean = df.copy()
        df_clean[features] = df_clean[features].replace([np.inf, -np.inf], np.nan)
        return df_clean
    
    @staticmethod
    def validate_minimum_samples(df: pd.DataFrame, min_samples: int, data_type: str = "data") -> None:
        """
        Validar que haya suficientes muestras para el análisis.
        
        Parameters:
        - df: DataFrame a validar
        - min_samples: Número mínimo de muestras requeridas
        - data_type: Tipo de datos para el mensaje de error
        
        Raises:
        - ValueError: Si no hay suficientes muestras
        """
        if len(df) < min_samples:
            raise ValueError(f"Insufficient {data_type}: {len(df)} < {min_samples} required")
    
    @staticmethod
    def get_valid_records(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Obtener registros válidos (sin NaN) y máscara de validez.
        
        Parameters:
        - df: DataFrame original
        - features: Lista de features a verificar
        
        Returns:
        - Tuple: (df_valido, mask_validez)
        """
        valid_mask = ~df[features].isna().any(axis=1)
        df_valid = df[valid_mask].copy()
        return df_valid, valid_mask
    
    @staticmethod  
    def validate_and_clean_pipeline(train_df: pd.DataFrame, 
                                   test_df: pd.DataFrame, 
                                   features: List[str],
                                   min_train_samples: int = 10,
                                   min_test_samples: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pipeline completo de validación y limpieza.
        
        Parameters:
        - train_df: DataFrame de entrenamiento
        - test_df: DataFrame de prueba
        - features: Lista de features
        - min_train_samples: Mínimo de muestras de entrenamiento
        - min_test_samples: Mínimo de muestras de prueba
        
        Returns:
        - Tuple: (train_df_clean, test_df_clean)
        """
        # 1. Validar features
        DataValidator.validate_features(train_df, test_df, features)
        
        # 2. Limpiar valores infinitos
        train_clean = DataValidator.clean_infinite_values(train_df, features)
        test_clean = DataValidator.clean_infinite_values(test_df, features)
        
        # 3. Validar muestras mínimas
        DataValidator.validate_minimum_samples(train_clean, min_train_samples, "training samples")
        DataValidator.validate_minimum_samples(test_clean, min_test_samples, "test samples")
        
        return train_clean, test_clean