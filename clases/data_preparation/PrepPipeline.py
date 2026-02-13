import sys
import pandas as pd
from importlib import reload 
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

### Leer Clases Customizadas
sys.path.append('./clases')
sys.path.append('./clases/data_preparation')  # ← Agregar esta línea

import prepare_nos
#reload(prepare_nos)
from prepare_nos import PrepareNOSData as nos_reader

import prepare_cust
#reload(prepare_cust)
from prepare_cust import PrepareCustomerMDM as cust_reader

import prepare_prod
#reload(prepare_prod)
from prepare_prod import PrepareProductMDM as prod_reader

import data_merge
#reload(data_merge)
from data_merge import DataMerger as data_merger


class DataPreparationPipeline:
    def __init__(self, nos_path, cust_path, prod_path):
        self.nos_path = nos_path
        self.cust_path = cust_path
        self.prod_path = prod_path

    def run(self):
        # Preparar datos NOS
        nos_data = nos_reader(self.nos_path).prepare_data()

        # Preparar datos Customer MDM
        cust_data = cust_reader(self.cust_path).prepare_data()

        # Preparar datos Product MDM
        prod_data = prod_reader(self.prod_path).prepare_data()

        # Merge datasets
        merger = data_merger()

        final_df, missing_channel_team_stats, analyze_missing_product_data = merger.data_merge(nos_data, cust_data, prod_data)

        # Crear directorio de salida si no existe
        output_dir = "Final_Reports/data_prep"
        os.makedirs(output_dir, exist_ok=True)

        # Escribir reportes en la carpeta de destino
        final_df.to_csv(os.path.join(output_dir, "final_df.csv"), index=False)
        missing_channel_team_stats.to_csv(os.path.join(output_dir, "missing_channel_team_stats.csv"), index=False)
        analyze_missing_product_data.to_csv(os.path.join(output_dir, "analyze_missing_product_data.csv"), index=False)

        return final_df,  missing_channel_team_stats, analyze_missing_product_data 