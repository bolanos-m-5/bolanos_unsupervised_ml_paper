"""
Customer MDM Reader Class
Clase para leer y preparar datos de MDM de clientes
"""

import warnings
import pandas as pd


class PrepareCustomerMDM:
    """
    Clase para leer y preparar el Master Data Management de clientes
    """
    
    def __init__(self, mdm_cust_path):
        """
        Inicializa la clase con la ruta del MDM de clientes
        
        Args:
            mdm_cust_path: Ruta al archivo CSV de MDM de clientes
        """
        self.mdm_cust_path = mdm_cust_path
        self.df_raw = None
        self.df_prepared = None
    
    def read_mdm_file(self):
        """
        Lee el archivo CSV de MDM de clientes
        
        Returns:
            DataFrame con datos crudos de MDM
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
            self.df_raw = pd.read_csv(self.mdm_cust_path, encoding='latin-1')
            return self.df_raw
        except UnicodeDecodeError:
            self.df_raw = pd.read_csv(self.mdm_cust_path, encoding='utf-8')
            return self.df_raw
    
    def filter_columns(self):
        """
        Filtra las columnas necesarias del MDM
        
        Returns:
            DataFrame con columnas filtradas
        """
        if self.df_raw is None:
            raise ValueError("Debe ejecutar read_mdm_file() primero")
        
        columns_needed = [
            'corp_cust_898_lvl_9_id',
            'cust_country_name', 
            'la_cust_channel_name',
            'la_cust_sub_channel_name',
            'la_cust_team_name', 
            'la_cust_sub_team_name', 
            'la_local_cnosgc_reporting_customer_optima_id',
            'la_local_cnosgc_reporting_customer_name'
        ]
        
        df_filtered = self.df_raw[columns_needed].copy()
        
        return df_filtered
    
    def group_by_customer(self, df):
        """
        Agrupa por customer ID y país, tomando el primer valor de cada grupo
        
        Args:
            df: DataFrame a agrupar
            
        Returns:
            DataFrame agrupado
        """
        df_grouped = df.groupby(
            ['corp_cust_898_lvl_9_id', 'cust_country_name'], 
            dropna=False
        ).agg({
            'la_cust_channel_name': 'first', 
            'la_cust_sub_channel_name': 'first',
            'la_cust_team_name': 'first',
            'la_cust_sub_team_name': 'first',
            'la_local_cnosgc_reporting_customer_optima_id': 'first',
            'la_local_cnosgc_reporting_customer_name': 'first'
        }).reset_index()
        
        return df_grouped
    
    def rename_columns(self, df):
        """
        Renombra la columna de customer ID
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con columna renombrada
        """
        df = df.rename(columns={
            "corp_cust_898_lvl_9_id": "cust_id",
            "la_local_cnosgc_reporting_customer_optima_id": "optima_id",
            'la_local_cnosgc_reporting_customer_name': 'optima_name',
            'la_cust_channel_name': 'channel',
            'la_cust_sub_channel_name': 'sub_channel',
            'la_cust_team_name': 'team',
            'la_cust_sub_team_name': 'sub_team'})
        
        return df
    
    def convert_to_uppercase(self, df):
        """
        Convierte todas las columnas de texto a mayúsculas
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con texto en mayúsculas
        """
        df = df.apply(
            lambda col: col.str.upper() if col.dtype == 'object' else col
        )
        
        return df
    
    def filter_by_country(self, df, country='MEXICO'):
        """
        Filtra registros por país (opcional)
        
        Args:
            df: DataFrame a filtrar
            country: País a filtrar (default: 'MEXICO')
            
        Returns:
            DataFrame filtrado por país
        """
        initial_count = len(df)
        df = df[df['cust_country_name'] == country.upper()]
        print(f"Registros filtrados por país {country}: {len(df)} (antes: {initial_count})")
        
        return df
    
    def prepare_data(self, filter_country=None):
        """
        Ejecuta todo el proceso de preparación del MDM de clientes
        
        Args:
            filter_country: País para filtrar (opcional, ej: 'MEXICO')
            
        Returns:
            DataFrame preparado y listo para merge
        """
        print("="*50)
        print("Iniciando preparación de MDM de clientes")
        
        # Leer archivo
        self.read_mdm_file()
        
        # Filtrar columnas
        df = self.filter_columns()
        
        # Agrupar por customer
        df = self.group_by_customer(df)
        
        # Renombrar columnas
        df = self.rename_columns(df)
        
        # Convertir a mayúsculas
        df = self.convert_to_uppercase(df)
        
        # Filtrar por país si se especifica
        if filter_country:
            print(f"Filtrando por país ({filter_country})...")
            df = self.filter_by_country(df, filter_country)
        
        self.df_prepared = df

        print("Preparación de MDM de clientes completada")
        
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
    
    def get_unique_channels(self):
        """
        Obtiene la lista de channels únicos
        
        Returns:
            Lista de channels únicos
        """
        if self.df_prepared is None:
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        return self.df_prepared['channel'].unique()
    
    def get_unique_teams(self):
        """
        Obtiene la lista de teams únicos
        
        Returns:
            Lista de teams únicos
        """
        if self.df_prepared is None:
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        return self.df_prepared['team'].unique()

