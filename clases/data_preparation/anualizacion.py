import pandas as pd

class Anualizacion:
    def __init__(self, df):
        """
        Inicializa la clase con un DataFrame.
        Debe contener las columnas:
        fiscal_year, gross_sales, nos, nsrd, sd
        """
        self.df = df

    def calcular_resumen_anual(self):
        """
        Agrupa por año fiscal y calcula sumas y tasas de variación.
        """
        # Agrupar por año fiscal
        year_data = (
            self.df.groupby(['fiscal_year'])[['gross_sales', 'nos', 'nsrd', 'sd']]
            .sum()
            .reset_index()
        )

        # Tasa de variación (%)
        year_data['variation_rate_gs'] = year_data['gross_sales'].pct_change() * 100
        year_data['variation_rate_nos'] = year_data['nos'].pct_change() * 100
        year_data['variation_rate_sd'] = year_data['sd'].pct_change() * 100
        year_data['variation_rate_nsrd'] = year_data['nsrd'].pct_change() * 100

        # Rates respecto a gross_sales
        year_data['rate_nos'] = (year_data['nos'] / year_data['gross_sales']) * 100
        year_data['rate_nsrd'] = (year_data['nsrd'] / year_data['gross_sales']) * 100
        year_data['rate_sd'] = (year_data['sd'] / year_data['gross_sales']) * 100

        return year_data
    
    def anualizar_data(self):
        """
        Agrupa por año fiscal, cliente, canal y equipo,
        calcula sumas y tasas de crecimiento.
        """
        annual_data = (
            self.df.groupby(['fiscal_year', 'Customer', 'Channel', 'Team'])[['gross_sales', 'nos', 'nsrd', 'sd']]
            .sum()
            .reset_index()
        )

        # Ordenar
        annual_data.sort_values(
            by=['fiscal_year', 'Customer', 'Channel', 'Team'],
            inplace=True
        )

        # Tasa de crecimiento por grupo
        annual_data['growth_rate_gs'] = annual_data.groupby(['Customer', 'Channel', 'Team'])['gross_sales'].pct_change() * 100
        annual_data['growth_rate_nos'] = annual_data.groupby(['Customer', 'Channel', 'Team'])['nos'].pct_change() * 100
        annual_data['growth_rate_sd'] = annual_data.groupby(['Customer', 'Channel', 'Team'])['sd'].pct_change() * 100
        annual_data['growth_rate_nsrd'] = annual_data.groupby(['Customer', 'Channel', 'Team'])['nsrd'].pct_change() * 100

        # Rates respecto a gross_sales
        annual_data['rate_nos'] = (annual_data['nos'] / annual_data['gross_sales']) * 100
        annual_data['rate_nsrd'] = (annual_data['nsrd'] / annual_data['gross_sales']) * 100
        annual_data['rate_sd'] = (annual_data['sd'] / annual_data['gross_sales']) * 100

        # Año como texto
        annual_data['fiscal_year'] = annual_data['fiscal_year'].astype(str)

        return annual_data