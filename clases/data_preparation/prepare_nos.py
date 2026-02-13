"""
Report Reader Class
Clase para leer y preparar datos de reportes Polaris
"""

import pandas as pd
import glob
import warnings


class PrepareNOSData:
    """
    Clase para leer y preparar reportes de Polaris
    """
    
    def __init__(self, reports_path):
        """
        Inicializa la clase con la ruta de los reportes
        
        Args:
            reports_path: Patrón de ruta a los archivos xlsx de reportes Polaris
        """
        self.reports_path = reports_path
        self.df_raw = None
        self.df_prepared = None
    
    def read_reports(self):
        """
        Lee y combina todos los archivos xlsx de la carpeta de reportes
        
        Returns:
            DataFrame con todos los reportes combinados
        """
        archivos = glob.glob(self.reports_path)
        
        if not archivos:
            raise FileNotFoundError(f"No se encontraron archivos en: {self.reports_path}")
        
        dfs = []
        for archivo in archivos:
            #print(f"Leyendo: {archivo}")
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                df = pd.read_excel(archivo)
            dfs.append(df)
        
        self.df_raw = pd.concat(dfs, ignore_index=True)
        return self.df_raw
    
    def filter_columns(self):
        """
        Filtra las columnas necesarias para el análisis
        
        Returns:
            DataFrame con columnas filtradas
        """
        if self.df_raw is None:
            raise ValueError("Debe ejecutar read_reports() primero")
        
        columns_needed = [
            'time_id', 
            'corp_cust_id', 
            'prod_id', 
            'giv_lc',
            'gs_lc', 
            'nit_lc',
            'nos_lc',
            'nsrd_manual_input_lc',
            'nsrd_sap_lc',
            'nsrd_tie_out_lc', 
            'nsrd_total_lc',
            'sd_live_rates_lc',
            'sd_manual_input_lc',
            'sd_total_lc', 
            'sd_tpr_lc',
            'volume_excl_nit_su', 
            'volume_nit_su',
            'volume_su'
        ]
        
        df_filtered = self.df_raw[columns_needed].copy()
        return df_filtered
    
    def adjust_data_types(self, df):
        """
        Ajusta tipos de datos de las columnas
        
        Args:
            df: DataFrame a ajustar
            
        Returns:
            DataFrame con tipos de datos ajustados
        """
        df['corp_cust_id'] = df['corp_cust_id'].astype('str')
        df['time_id'] = df['time_id'].astype('str')
        df['prod_id'] = df['prod_id'].astype('str')
        
        print("Tipos de datos ajustados a string para IDs")
        
        return df
    
    def remove_dummy_customers(self, df):
        """
        Elimina clientes Dummy del DataFrame
        
        Args:
            df: DataFrame a filtrar
            
        Returns:
            DataFrame sin clientes Dummy
        """
        initial_count = len(df)
        df = df[~df['corp_cust_id'].str.contains('Dummy', na=False)]
        removed = initial_count - len(df)
        
        print(f"Clientes Dummy removidos: {removed}")
        print(f"Registros restantes: {len(df)}")
        
        return df
    
    def rename_customer_column(self, df):
        """
        Renombra la columna de customer ID
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con columna renombrada
        """
        df = df.rename(columns={"corp_cust_id": "cust_id"})
        print("Columna 'corp_cust_id' renombrada a 'cust_id'")
        
        return df
    
    def prepare_data(self):
        """
        Ejecuta todo el proceso de preparación de datos
        
        Returns:
            DataFrame preparado y listo para merge
        """
        print("="*50)
        print("Iniciando preparacion data Polaris NOS")
        
        # Leer reportes
        self.read_reports()
        
        # Filtrar columnas
        df = self.filter_columns()
        
        # Ajustar tipos de datos
        df = self.adjust_data_types(df)
        
        # Remover Dummy
        df = self.remove_dummy_customers(df)
        
        # Renombrar columna
        df = self.rename_customer_column(df)
        
        self.df_prepared = df
        
        print("Preparacion data Polaris NOS completada")
        
        return self.df_prepared
    
    def get_prepared_data(self):
        """
        Obtiene los datos preparados
        
        Returns:
            DataFrame preparado
        """
        if self.df_prepared is None:
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        return self.df_prepared


