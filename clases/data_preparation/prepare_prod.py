"""
Product MDM Reader Class
Clase para leer y preparar datos de MDM de productos
"""

import warnings
import pandas as pd


class PrepareProductMDM:
    """
    Clase para leer y preparar el Master Data Management de productos
    """
    
    def __init__(self, mdm_prod_path):
        """
        Inicializa la clase con la ruta del MDM de productos
        
        Args:
            mdm_prod_path: Ruta al archivo Excel de MDM de productos
        """
        self.mdm_prod_path = mdm_prod_path
        self.df_raw = None
        self.df_prepared = None
    
    def read_mdm_file(self):
        """
        Lee el archivo Excel de MDM de productos
        
        Returns:
            DataFrame con datos crudos de MDM
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.df_raw = pd.read_excel(self.mdm_prod_path)
        
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
            'corporate_fpc',
            'corporate_item_gtin', 
            'corporate_product_name',
            'corporate_brand_name',
            'corporate_category_name', 
            'corporate_sub_sector_name'
        ]
        
        df_filtered = self.df_raw[columns_needed].copy()
        
        return df_filtered
    
    def group_by_product(self, df):
        """
        Agrupa por product ID y GTIN, tomando el primer valor de cada grupo
        
        Args:
            df: DataFrame a agrupar
            
        Returns:
            DataFrame agrupado
        """
        df_grouped = df.groupby(
            ['corporate_fpc', 'corporate_item_gtin']
        ).agg({
            'corporate_product_name': 'first',
            'corporate_brand_name': 'first',
            'corporate_category_name': 'first',
            'corporate_sub_sector_name': 'first'
        }).reset_index()
        
        return df_grouped
    
    def rename_columns(self, df):
        """
        Renombra la columna de product ID
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con columna renombrada
        """
        df = df.rename(columns={"corporate_fpc": "prod_id"})
        
        return df
    
    def convert_to_uppercase(self, df):
        """
        Convierte columnas de texto a mayúsculas (opcional)
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con texto en mayúsculas
        """
        text_columns = [
            'corporate_product_name',
            'corporate_brand_name',
            'corporate_category_name',
            'corporate_sub_sector_name'
        ]
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].str.upper()
        
        return df
    
    def prepare_data(self, uppercase=False):
        """
        Ejecuta todo el proceso de preparación del MDM de productos
        
        Args:
            uppercase: Si se deben convertir textos a mayúsculas (default: False)
            
        Returns:
            DataFrame preparado y listo para merge
        """
        print("="*50)
        print("Iniciando preparación de MDM de productos")
        
        # Leer archivo
        self.read_mdm_file()
        
        # Filtrar columnas
        df = self.filter_columns()
        
        # Agrupar por producto
        df = self.group_by_product(df)
        
        # Renombrar columnas
        df = self.rename_columns(df)
        
        # Convertir a mayúsculas si se especifica
        if uppercase:
            df = self.convert_to_uppercase(df)
        
        self.df_prepared = df
        
        print("Preparación de MDM de productos completada")
        
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
    
    def get_unique_brands(self):
        """
        Obtiene la lista de marcas únicas
        
        Returns:
            Lista de marcas únicas
        """
        if self.df_prepared is None:
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        return self.df_prepared['corporate_brand_name'].unique()
    
    def get_unique_categories(self):
        """
        Obtiene la lista de categorías únicas
        
        Returns:
            Lista de categorías únicas
        """
        if self.df_prepared is None:
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        return self.df_prepared['corporate_category_name'].unique()
    
    def get_unique_sub_sectors(self):
        """
        Obtiene la lista de sub-sectores únicos
        
        Returns:
            Lista de sub-sectores únicos
        """
        if self.df_prepared is None:
            raise ValueError("Debe ejecutar prepare_data() primero")
        
        return self.df_prepared['corporate_sub_sector_name'].unique()



