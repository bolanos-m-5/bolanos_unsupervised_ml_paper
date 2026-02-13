import pandas as pd

class Quarteralization:
    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame.
        Debe contener las columnas:
        fiscal_year, gross_sales, nos, nsrd, sd
        """
        self.df = df

    def prepare_data(self, adjusted):
        """
        Prepara los datos para quarteralización.
        Asegura que la columna 'quarter' esté en formato datetime.
        """
        data = adjusted

        # Filter only quarters in scope

        data = data.rename(columns={
            "gs_lc": "gross_sales",
            "nos_lc": "nos",
            'nsrd_total_lc': 'nsrd',
            'sd_total_lc': 'sd',
            'la_cust_sub_channel_name': 'sub_channel',
            'la_cust_team_name': 'team',
            'la_cust_sub_team_name': 'sub_team',
            'giv_lc': 'giv'})

        # Agrupar por 'quarter' y las otras columnas y sumar todas las columnas numéricas
        quarterly_data = (
                    data.groupby([ 'fiscal_year', 'customer', 'channel', 'team', 'quarter'])[['gross_sales', 'giv', 
                                                                                        'nos', 
                                                                                        'nsrd', 
                                                                                        'nit_lc',
                                                                                        'sd', 
                                                                                        'nos_excl_NIT',
                                                                                            'nsrd_excl_nit', 
                                                                                            'volume_excl_nit_su'
                                                                                            ]]
                    .sum()
                    .reset_index()
                )

        

        quarterly_data= quarterly_data.sort_values(by=['customer', 'channel', 'team', 'quarter', 'quarter'])

        #quarterly_data.set_index('quarter', inplace=True)

        return quarterly_data
    
    def prepare_data_prod(self, adjusted):
        """
        Prepara los datos para quarteralización.
        Asegura que la columna 'quarter' esté en formato datetime.
        """
        data = adjusted

        # Filter only quarters in scope

        data = data.rename(columns={
            "gs_lc": "gross_sales",
            "nos_lc": "nos",
            'nsrd_total_lc': 'nsrd',
            'sd_total_lc': 'sd',
            'la_cust_sub_channel_name': 'sub_channel',
            'la_cust_team_name': 'team',
            'la_cust_sub_team_name': 'sub_team',
            'giv_lc': 'giv',
            'corporate_sub_sector_name': 'sub_sector',})

        # Agrupar por 'quarter' y las otras columnas y sumar todas las columnas numéricas
        quarterly_data = (
                    data.groupby([ 'fiscal_year', 'category_smo', 'customer','channel', 'team', 'quarter'])[['gross_sales', 'giv', 
                                                                                        'nos', 
                                                                                        'nsrd', 
                                                                                        'nit_lc',
                                                                                        'sd', 
                                                                                        'nos_excl_NIT',
                                                                                            'nsrd_excl_nit', 
                                                                                            'volume_excl_nit_su'
                                                                                            ]]
                    .sum()
                    .reset_index()
                )

        

        quarterly_data= quarterly_data.sort_values(by=['category_smo','customer', 'channel', 'team',  'quarter'])

        #quarterly_data.set_index('quarter', inplace=True)

        return quarterly_data

    def calcular_resumen_trimestral(self, adjusted_data):
        """
        Agrupa por trimestre y calcula sumas y tasas de variación.
        """
        # Agrupar por trimestre
        quarter_data = (
            adjusted_data.groupby(['fiscal_year', 'quarter'])[['gross_sales', 'giv', 'nos',
                                                  'nsrd', 'sd', 'nos_excl_NIT',
                                                    'nsrd_excl_nit', 'volume_excl_nit_su'
                                                    ]]
            .sum()
            .reset_index()
        )

        quarter_data['nos_su'] = quarter_data['nos_excl_NIT'] / quarter_data['volume_excl_nit_su']

        # Índice de variación (base 100)
        quarter_data['variation_rate_gs'] = (1 + quarter_data['gross_sales'].pct_change()) * 100
        quarter_data['variation_rate_nos'] = (1 + quarter_data['nos_excl_NIT'].pct_change()) * 100
        quarter_data['variation_rate_volume'] = (1 + quarter_data['volume_excl_nit_su'].pct_change()) * 100
        quarter_data['variation_rate_nos_su'] = (1 + quarter_data['nos_su'].pct_change()) * 100
        quarter_data['variation_rate_sd'] = (1 + quarter_data['sd'].pct_change()) * 100
        quarter_data['variation_rate_nsrd'] = (1 + quarter_data['nsrd_excl_nit'].pct_change()) * 100

        # Rates respecto a gross_sales
        quarter_data['rate_nos'] = (quarter_data['nos_excl_NIT'] / quarter_data['gross_sales']) * 100
        quarter_data['rate_nsrd'] = (quarter_data['nsrd_excl_nit'] / quarter_data['gross_sales']) * 100
        quarter_data['rate_sd'] = (quarter_data['sd'] / quarter_data['gross_sales']) * 100

        quarter_data['srm'] = quarter_data['variation_rate_nos'] - quarter_data['variation_rate_volume']


        return quarter_data
    
    def resumen_trimestral_por_grupo(self, adjusted_data, group_by):
        """
        Agrupa por trimestre y una columna de agrupación específica (channel, team, etc.)
        y calcula sumas y tasas de variación.
        
        Args:
            adjusted_data: DataFrame con los datos ajustados
            group_by: String con el nombre de la columna de agrupación ('channel', 'team', etc.)
        """

        # Convertir a lista si es string
        group_cols = [group_by] if isinstance(group_by, str) else group_by
        # Agrupar por trimestre y la columna especificada
        adjusted_data = adjusted_data.sort_values(by=['fiscal_year'] + group_cols + ['quarter'])

        # Sumar métricas por grupo
        quarter_data = (
            adjusted_data.groupby(['fiscal_year', 'quarter'] + group_cols)[['gross_sales', 'giv', 'nos',
                                                  'nsrd', 'sd', 'nos_excl_NIT',
                                                    'nsrd_excl_nit', 'volume_excl_nit_su'
                                                    ]]
            .sum()
            .reset_index()
        )

        # Contar clientes únicos
        if 'customer' in adjusted_data.columns and 'customer' not in group_cols:
            unique_customers_count = (
                adjusted_data.groupby(['fiscal_year', 'quarter'] + group_cols)['customer'] 
                .nunique()
                .reset_index()
                .rename(columns={'customer': 'n_customers'})
            )    
            quarter_data = quarter_data.merge(unique_customers_count, on=['fiscal_year', 'quarter'] + group_cols, how='left')

        # Separar por años
        df_2023 = quarter_data[quarter_data['fiscal_year'] == 2023]
        df_2024 = quarter_data[quarter_data['fiscal_year'] == 2024]
        df_2025 = quarter_data[quarter_data['fiscal_year'] == 2025]

        # Combinar años para cálculo de variaciones
        combined_2324 = pd.concat([df_2023, df_2024], ignore_index=True)
        combined_2425 = pd.concat([df_2024, df_2025], ignore_index=True)

        var_rates_2324 = self.var_rates_by_group(combined_2324, group_by)
        var_rates_2425 = self.var_rates_by_group(combined_2425, group_by)

        var_rates_25 = var_rates_2425[var_rates_2425['fiscal_year'] == 2025]

        quarter_data = pd.concat([var_rates_2324, var_rates_25]).sort_values(by=group_cols + ['quarter'])

        # Calcular tasas respecto a gross_sales
        quarter_data['rate_nos'] = (quarter_data['nos_excl_NIT'] / quarter_data['gross_sales']) * 100
        quarter_data['rate_nsrd'] = (quarter_data['nsrd_excl_nit'] / quarter_data['gross_sales']) * 100
        quarter_data['rate_sd'] = (quarter_data['sd'] / quarter_data['gross_sales']) * 100

        quarter_data['srm'] = quarter_data['variation_rate_nos'] - quarter_data['variation_rate_volume']

        quarter_data = quarter_data.sort_values(by=group_cols + ['fiscal_year', 'quarter'])

        return quarter_data
     
    
    def calcular_var_rates(self, quarter_data, group_cols=None):
        """
        Calcula las tasas de variación agrupadas por las columnas especificadas.
        
        Args:
            quarter_data: DataFrame con los datos trimestrales
            group_cols: Lista de columnas para agrupar. Por defecto: ['customer', 'channel', 'team']
        """
        if group_cols is None:
            group_cols = ['customer', 'channel', 'team']

        # Convertir a lista si es string
        group_cols = [group_cols] if isinstance(group_cols, str) else group_cols
        
        return self.var_rates_by_group(quarter_data, group_cols)
    
    def var_rates_by_group(self, quarter_data, group_cols):
        """
        Calcula las tasas de variación agrupadas por las columnas especificadas.
        
        """

         # Convertir a lista si es string
        group_cols = [group_cols] if isinstance(group_cols, str) else group_cols

        quarter_data['nos_su'] = quarter_data['nos_excl_NIT'] / quarter_data['volume_excl_nit_su']


        quarter_data = quarter_data.sort_values(by=group_cols + ['quarter'])

        # Índice de variación (base 100)
        quarter_data['variation_rate_gs'] = (1 + quarter_data.groupby(group_cols)['gross_sales'].pct_change()) * 100
        quarter_data['variation_rate_nos'] = (1 + quarter_data.groupby(group_cols)['nos_excl_NIT'].pct_change()) * 100
        quarter_data['variation_rate_volume'] = (1 + quarter_data.groupby(group_cols)['volume_excl_nit_su'].pct_change()) * 100
        quarter_data['variation_rate_nos_su'] = (1 + quarter_data.groupby(group_cols)['nos_su'].pct_change()) * 100
        quarter_data['variation_rate_sd'] = (1 + quarter_data.groupby(group_cols)['sd'].pct_change()) * 100
        quarter_data['variation_rate_nsrd'] = (1 + quarter_data.groupby(group_cols)['nsrd_excl_nit'].pct_change()) * 100

        return quarter_data

        