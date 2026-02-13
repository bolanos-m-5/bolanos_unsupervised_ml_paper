"""
Data Merger Class
Clase para unir datos de reportes con MDM y aplicar transformaciones finales
"""

import pandas as pd


class DataMerger:
    """
    Clase para unir datos y aplicar transformaciones finales
    """
    
    def __init__(self, df_reports=None, df_customers=None, df_products=None):
        """
        #Init
        Args:
            df_reports: DataFrame de reportes
            df_customers: DataFrame de MDM de clientes
            df_products: DataFrame de MDM de productos
        """
    def read_category_smo_mapping(self):
        """
        Lee el archivo de mapeo de categoría a SMO
        
        Returns:
            DataFrame con mapeo
        """
        path ='C:/Users/bolanos.m.5/OneDrive - Procter and Gamble/Documents/Polaris_DS_Master/Datasets/Dictionaries/category_smo_mapping.xlsx'
        df_mapping = pd.read_excel(path)
        df_mapping = df_mapping[['corporate_category_name', 'category_smo']]
        return df_mapping    
    
    def merge_with_customers(self, df_reports, df_customers):
        """
        Hace merge de reportes con MDM de clientes
        
        Returns:
            DataFrame con merge de clientes
        """
        if df_reports is None or df_customers is None:
            raise ValueError("Debe establecer los datos primero con set_data()")
        
        df_merged = df_reports.merge(
            df_customers, 
            on='cust_id', 
            how='left'
        )
        
        # Analizar match

        # Filtrar solo con match
        df_happy_merged = df_merged[df_merged['cust_country_name'].notna()]
        df_not_merged = df_merged[df_merged['cust_country_name'].isna()]

        matched = df_happy_merged['cust_country_name'].notna().sum()
        not_matched = df_not_merged['cust_country_name'].isna().sum()
        
        print(f"\nMerge con clientes:")
        print(f"  - Con match: {matched} ({matched/len(df_merged)*100:.2f}%)")
        print(f"  - Sin match: {not_matched} ({not_matched/len(df_merged)*100:.2f}%)")
        
         
        return df_happy_merged, df_not_merged
    
    def merge_with_products(self, df_happy_merged, df_products):
        """
        Hace merge con MDM de productos
        
        Args:
            df: DataFrame con merge de clientes
            
        Returns:
            DataFrame con merge de productos
        """
        if df_products is None:
            raise ValueError("Debe establecer los datos primero con set_data()")
        
        df_merged = df_happy_merged.merge(
            df_products, 
            on='prod_id', 
            how='left'
        )
        

        df_happy_merged = df_merged[df_merged['corporate_item_gtin'].notna()]
        df_not_merged = df_merged[df_merged['corporate_item_gtin'].isna()]
                                  
        # Analizar match
        matched = df_happy_merged['corporate_item_gtin'].notna().sum()
        not_matched = df_merged['corporate_item_gtin'].isna().sum()
    
        print(f"\nMerge con productos:")
        print(f"  - Con match: {matched} ({matched/len(df_merged)*100:.2f}%)")
        print(f"  - Sin match: {not_matched} ({not_matched/len(df_merged)*100:.2f}%)")

        
        return df_happy_merged, df_not_merged
    

    def analyze_missing_channel_team(self, df_merged):
        """
        Analiza valores faltantes de channel y team antes del fill
        
        Returns:
            Dict con estadísticas de valores faltantes
        """
        if df_merged is None:
            raise ValueError("Debe ejecutar merge_and_transform() primero")
        
        # Analizar antes del fill (usar columnas originales si existen)
        df_to_analyze = df_merged

        # Values by channel and team
        pd.options.display.float_format = '{:,.2f}'.format

        values_by_channel_team = df_to_analyze.groupby(
            ['cust_id', 'optima_id',
            'optima_name',
            'channel', 'team', 'time_id'], dropna=False
        ).agg({
        'giv_lc': 'sum'
        }).reset_index()

        values_by_channel_team = values_by_channel_team[
            values_by_channel_team['channel'].isna() |
            values_by_channel_team['team'].isna()
        ]
        
        missing_channel = values_by_channel_team['channel'].isna().sum()
        missing_team = values_by_channel_team['team'].isna().sum()
        total = len(df_to_analyze)
        
        result = {
            'total_registros': total,
            'channel_faltante': missing_channel,
            'channel_faltante_pct': (missing_channel / total * 100) if total > 0 else 0,
            'team_faltante': missing_team,
            'team_faltante_pct': (missing_team / total * 100) if total > 0 else 0,
        }
        
        print(f"Análisis de valores faltantes de Channel y Team")
        print(f"Total registros: {result['total_registros']:,}")
        print(f"Channel faltante: {result['channel_faltante']:,} ({result['channel_faltante_pct']:.2f}%)")
        print(f"Team faltante: {result['team_faltante']:,} ({result['team_faltante_pct']:.2f}%)")
        
        return values_by_channel_team
    
    def fill_missing_channel_team(self, df):
        """
        Llena valores faltantes de channel y team con 'OTHERS'
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con valores faltantes llenados
        """
        df['channel'] = df['channel'].fillna('OTHERS')
        df['team'] = df['team'].fillna('OTHERS')
        
        return df
    

    def assign_best_channel_team_by_optima(self, df):
        """
        Asigna el channel y team con mayor GIV a cada optima_id
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con channel y team optimizados
        """
        # Calcular total de giv_lc por optima_id, channel y team
        giv_by_optima = df.groupby(
            ['optima_id', 'channel', 'team']
        ).agg({'giv_lc': 'sum'}).reset_index()
        
        # Encontrar el channel y team con mayor giv_lc por optima_id
        idx = giv_by_optima.groupby('optima_id')['giv_lc'].idxmax()
        best_channel_team = giv_by_optima.loc[idx, ['optima_id', 'channel', 'team']]
        best_channel_team = best_channel_team.rename(
            columns={'channel': 'best_channel', 'team': 'best_team'}
        )
        
        # Merge para asignar los mejores valores
        df = df.merge(best_channel_team, on='optima_id', how='left')
        
        # Reemplazar channel y team con los mejores valores
        df['channel'] = df['best_channel']
        df['team'] = df['best_team']
        
        # Eliminar columnas auxiliares
        df = df.drop(['best_channel', 'best_team'], axis=1)
        
        return df
    
    def quality_check_optima_names(self, df):
        """
        Verifica la calidad de datos: detecta optima_ids con nombres inconsistentes
        a nivel de cust_id. Genera reporte Excel en Final_Reports/
        
        Args:
            df: DataFrame con datos merged (debe contener cust_id, optima_id, optima_name)
            
        Returns:
            DataFrame con errores detectados (optima_ids con múltiples nombres)
        """
        # Obtener combinaciones únicas de cust_id, optima_id y optima_name
        unique_combinations = df[['cust_id', 'optima_id', 'optima_name', 'channel', 'team']].drop_duplicates()
        
        # Agrupar por optima_id y contar nombres únicos
        optima_analysis = unique_combinations.groupby('optima_id').agg({
            'optima_name': lambda x: list(x.unique()),
            'cust_id': lambda x: list(x.unique())
        }).reset_index()
        
        # Calcular número de nombres y cust_ids únicos
        optima_analysis['num_nombres_diferentes'] = optima_analysis['optima_name'].apply(len)
        optima_analysis['num_cust_ids'] = optima_analysis['cust_id'].apply(len)
        
        # Filtrar solo los que tienen inconsistencias (más de un nombre)
        errores = optima_analysis[optima_analysis['num_nombres_diferentes'] > 1].copy()
        
        if len(errores) > 0:
            # Crear reporte detallado
            reporte_detallado = []
            for idx, row in errores.iterrows():
                optima = row['optima_id']
                nombres = row['optima_name']
                cust_ids = row['cust_id']
                
                # Obtener detalles de cada cust_id con ese optima_id
                detalles = unique_combinations[unique_combinations['optima_id'] == optima]
                
                # Contar registros totales para ese optima_id
                num_registros = len(df[df['optima_id'] == optima])
                
                # Filtrar valores None de channel y team
                canales_unicos = [str(c) for c in detalles['channel'].unique() if c is not None]
                equipos_unicos = [str(t) for t in detalles['team'].unique() if t is not None]
                
                reporte_detallado.append({
                    'optima_id': optima,
                    'num_nombres_diferentes': len(nombres),
                    'nombres_encontrados': ' | '.join(nombres),
                    'num_cust_ids': len(cust_ids),
                    'cust_ids_afectados': ', '.join(cust_ids),
                    'num_registros_total': num_registros,
                    'canales': ', '.join(canales_unicos) if canales_unicos else 'N/A',
                    'equipos': ', '.join(equipos_unicos) if equipos_unicos else 'N/A'
                })
            
            reporte_df = pd.DataFrame(reporte_detallado)
            
            # Exportar reporte
            output_path = 'Final_Reports/QUALITY_CHECK_OPTIMA_NAMES.xlsx'
            reporte_df.to_excel(output_path, index=False)
            print(f"⚠️ Quality Check: {len(reporte_df)} Optima IDs con errores → {output_path}")
            
        else:
            reporte_df = pd.DataFrame()
            print("✅ Quality Check: No se detectaron errores en nombres")
        
        return reporte_df
    
    def analyze_missing_product_data(self, happy_merged):
        """
        Analiza registros con datos faltantes de productos

        Returns:
            DataFrame con registros faltantes
        """
        if happy_merged is None:
            raise ValueError("Debe ejecutar merge_and_transform() primero")

        missing = happy_merged[
            happy_merged['corporate_item_gtin'].isna() |
            happy_merged['corporate_product_name'].isna() |
            happy_merged['corporate_brand_name'].isna() |
            happy_merged['corporate_category_name'].isna() |
            happy_merged['corporate_sub_sector_name'].isna()
        ]

        print(f"\nRegistros con datos de producto faltantes: {len(missing)}")

        return missing
    
    def create_final_aggregation(self, df):
        """
        Crea la agregación final por todas las dimensiones
        
        Args:
            df: DataFrame a agrupar
            
        Returns:
            DataFrame agrupado final
        """
        df_grouped = df.groupby([
            'optima_id',
            'optima_name',
            'channel',
            'team',
            'time_id',
            'corporate_item_gtin',
            'corporate_product_name',
            'corporate_brand_name',
            'corporate_category_name',
            'corporate_sub_sector_name'
        ], dropna=False).agg({
            'giv_lc': 'sum',
            'gs_lc': 'sum',
            'nit_lc': 'sum',
            'nos_lc': 'sum',
            'nsrd_manual_input_lc': 'sum',
            'nsrd_sap_lc': 'sum',
            'nsrd_tie_out_lc': 'sum',
            'nsrd_total_lc': 'sum',
            'sd_live_rates_lc': 'sum',
            'sd_manual_input_lc': 'sum',
            'sd_total_lc': 'sum',
            'sd_tpr_lc': 'sum',
            'volume_excl_nit_su': 'sum',
            'volume_nit_su': 'sum',
            'volume_su': 'sum'
        }).reset_index()
        
        return df_grouped
    
    def add_calculated_columns(self, df):
        """
        Agrega columnas calculadas
        
        Args:
            df: DataFrame a modificar
            
        Returns:
            DataFrame con columnas calculadas
        """
        # Customer concatenado
        df['customer'] = (
            df['optima_id'].astype(str) + ' - ' + 
            df['optima_name'].astype(str)
        )
        
        # NOS excluyendo NIT
        df['nos_excl_NIT'] = df['nos_lc'] + df['nit_lc']
        
        # NSRD excluyendo NIT
        df['nsrd_excl_nit'] = df['nsrd_total_lc'] - df['nit_lc']

        df['time_id'] = pd.to_datetime(df['time_id'].astype(str) + '01', format='%Y%m%d')

        # Determinar el año fiscal
        df['fiscal_year'] = df['time_id'].dt.year.where(df['time_id'].dt.month >= 7, df['time_id'].dt.year - 1)

        df['quarter'] = df['time_id'].dt.to_period('Q').astype(str)

        df['quarter'] = pd.PeriodIndex(df['quarter'], freq='Q-DEC')  
        df['quarter'] = df['quarter'].dt.start_time  
        
        return df
    
    def add_category_smo(self, df, df_mapping):
        """
        Agrega columna category_smo basada en el mapeo
        
        Args:
            df: DataFrame a modificar
            mapping_path: Ruta al archivo de mapeo
            
        Returns:
            DataFrame con columna category_smo agregada
        """
        
        final_df_filtered = df.merge(df_mapping, how='left', on ='corporate_category_name')

        missing_category_smo = final_df_filtered[final_df_filtered['category_smo'].isnull()]
        missing_count = len(missing_category_smo)
        print(f"Registros con category_smo faltante: {missing_count}")
        
        print("Columna category_smo agregada según mapeo")
        
        return final_df_filtered
    
    def final_format(self, adjusted):
        """
        Prepara los datos para quarteralización.
        Agrupa por quarter y dimensiones clave, sumando métricas.
        
        Args:
            adjusted: DataFrame con datos ajustados
            
        Returns:
            DataFrame agrupado por quarter
        """
        data = adjusted

        # Renombrar columnas
        data = data.rename(columns={
            "gs_lc": "gross_sales",
            "nos_lc": "nos",
            'nsrd_total_lc': 'nsrd',
            'sd_total_lc': 'sd',
            'la_cust_sub_channel_name': 'sub_channel',
            'la_cust_team_name': 'team',
            'la_cust_sub_team_name': 'sub_team',
            'giv_lc': 'giv',
            'corporate_sub_sector_name': 'sub_sector',
        })

        # Agrupar por 'quarter' y las otras columnas y sumar todas las columnas numéricas
        quarterly_data = (
            data.groupby(['fiscal_year', 'category_smo', 'customer', 'channel', 'team', 'quarter'])[
                ['gross_sales', 'giv', 'nos', 'nsrd', 'nit_lc', 'sd', 'nos_excl_NIT', 
                 'nsrd_excl_nit', 'volume_excl_nit_su']
            ]
            .sum()
            .reset_index()
        )

        # Ordenar resultados
        quarterly_data = quarterly_data.sort_values(
            by=['category_smo', 'customer', 'channel', 'team', 'quarter']
        )

        return quarterly_data
    
    def reorder_columns(self, df):
        """
        Reordena columnas: IDs primero, luego alfabéticamente
        
        Args:
            df: DataFrame a reordenar
            
        Returns:
            DataFrame con columnas reordenadas
        """
        id_cols = ['fiscal_year', 'quarter', 'category_smo', 'customer', 'channel', 'team' ]
        other_cols = sorted([col for col in df.columns if col not in id_cols])
        df = df[id_cols + other_cols]
        
        return df
    
    def data_merge(self, df_reports, df_customers, df_products):
        """
        Ejecuta todo el proceso de merge y transformación
        
        Returns:
            DataFrame final transformado
        """
        # 1. Merge con clientes
        df_happy_merged, df_not_merged_customers = self.merge_with_customers(df_reports, df_customers)
        
        # 🔍 QUALITY CHECK: Verificar consistencia de nombres por Optima ID
        quality_report = self.quality_check_optima_names(df_happy_merged)
        
        # 2. Merge con productos
        df_happy_merged, df_not_merged_products = self.merge_with_products(df_happy_merged, df_products)
        
        missing_channel_team_stats = self.analyze_missing_channel_team(df_happy_merged)

        fill_missing_channel_team = self.fill_missing_channel_team(df_happy_merged)


        assign_best_channel_team_by_optima = self.assign_best_channel_team_by_optima(fill_missing_channel_team)

        #Optima Names check
        reporte_df = self.quality_check_optima_names(assign_best_channel_team_by_optima)

        analyze_missing_product_data = self.analyze_missing_product_data(assign_best_channel_team_by_optima)

        create_final_aggregation = self.create_final_aggregation(assign_best_channel_team_by_optima)

        final_df = self.add_calculated_columns(create_final_aggregation)

        df_mapping = self.read_category_smo_mapping()

        final_df = self.add_category_smo(final_df, df_mapping)

        final_df = self.final_format(final_df)

        final_df = self.reorder_columns(final_df)

        print("Proceso de merge y transformación completado")
        print("="*50)
        
        return final_df, missing_channel_team_stats, analyze_missing_product_data
    
