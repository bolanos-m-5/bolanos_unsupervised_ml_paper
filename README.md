# 🎯 Polaris Analytics - ML System for Retail Business Intelligence

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📋 Descripción

Sistema integrado de **Machine Learning no supervisado** para análisis de negocio en retail que combina scoring dinámico multidimensional, detección de anomalías mediante Isolation Forest y clustering jerárquico optimizado. Diseñado bajo la metodología **CRISP-DM** con arquitectura modular end-to-end para garantizar reproducibilidad y escalabilidad.

### 🎯 Objetivo del Proyecto

Transformar grandes volúmenes de datos financieros operacionales (ventas, gastos, indicadores) en **inteligencia comercial accionable** mediante:

- **Scoring dinámico** que evalúa performance de clientes, canales y equipos
- **Detección automática de anomalías** sin umbrales arbitrarios
- **Segmentación de clientes** en grupos homogéneos para estrategias diferenciadas

---

## ✨ Características Principales

### 🏆 Scoring Dinámico Multidimensional
- Sistema de ponderación automática de 4 métricas clave (Rate NOS, NOS SU, Variaciones)
- Benchmarks calculados dinámicamente por período fiscal
- Penalizaciones automáticas para crecimientos negativos
- Transformación a escala [0-10] mediante ranking percentil

### 🔍 Detección de Anomalías (Isolation Forest)
- **Optimización exhaustiva**: 702 experimentos (26 combinaciones de features × 27 configuraciones de hiperparámetros)
- Grid search automático: `contamination`, `n_estimators`, `max_samples`
- Validación temporal: entrenamiento 2024, evaluación 2025
- Identificación de comportamientos atípicos sin umbrales predefinidos

### 🎯 Clustering Jerárquico Optimizado
- **Optimización anidada**: Grid search de hiperparámetros + feature selection paralela
- Comparación automática **SFS vs RFE** (28 combinaciones evaluadas)
- Métodos: Ward, Complete, Average con métricas Euclidean, Cosine, Correlation
- Best model: Silhouette Score 0.3413 con 6 clusters interpretables

### 🔧 Pipeline End-to-End Automatizado
- Preparación de datos con validación automática
- Análisis exploratorio integrado
- Orquestación de modelos con clases especializadas
- Exportación automatizada de reportes para consumo en Power BI

---

## 🏗️ Arquitectura del Sistema

```
Polaris_DS_Master/
│
├── clases/                                    # Módulos principales del sistema
│   ├── data_preparation/                      # Pipeline de preparación de datos
│   │   ├── PrepPipeline.py                   # Orquestador principal
│   │   ├── data_merge.py                     # Integración de fuentes
│   │   ├── prepare_cust.py                   # Limpieza datos clientes
│   │   ├── prepare_nos.py                    # Cálculo métricas NOS
│   │   ├── prepare_prod.py                   # Procesamiento productos
│   │   ├── quarteralization.py               # Agregación trimestral
│   │   └── anualizacion.py                   # Agregación anual
│   │
│   ├── exploratory_analysis/                  # Análisis exploratorio
│   │   └── Analisis_exploratorio.py          # EDA automatizado
│   │
│   ├── scoring/                               # Sistema de scoring
│   │   └── NosScore.py                       # Scoring dinámico multidimensional
│   │
│   ├── ml_models/                             # Modelos de ML
│   │   ├── anomaly_detection/                # Detección de anomalías
│   │   │   ├── anomaly_detection_orchestrator.py  # Coordinador principal
│   │   │   ├── anomaly_optimizer.py          # HPO con grid search
│   │   │   └── utilities/                    # Utilidades especializadas
│   │   │       ├── anomaly_predictor.py      # Predicción de anomalías
│   │   │       ├── anomaly_explainer.py      # Interpretación de resultados
│   │   │       ├── anomaly_reporter.py       # Generación de reportes
│   │   │       ├── data_scaler.py            # Normalización de features
│   │   │       └── data_validator.py         # Validación de datos
│   │   │
│   │   └── clustering/                        # Clustering jerárquico
│   │       ├── analyzer.py                   # Análisis e interpretación
│   │       ├── estimator.py                  # Algoritmos de clustering
│   │       ├── feature_selector.py           # SFS + RFE comparativo
│   │       └── parameter_grids.py            # Espacios de búsqueda HPO
│   │
│   └── main_pipelines/                        # Notebooks de ejecución
│       ├── Main_Pipeline.ipynb               # Pipeline principal integrado
│       └── exploratory.ipynb                 # Análisis exploratorio iterativo
│
├── Datasets/                                  # Datos de entrada
│   ├── MDM_cust/                             # Master data clientes
│   ├── MDM_prod/                             # Master data productos
│   └── Polaris_reports/                      # Reportes financieros
│
├── Final_Reports/                             # Reportes generados
│   └── pipeline_results/                     # Resultados del pipeline
│       ├── reporte_final_pipeline_*.csv      # Reporte integrado
│       ├── resumen_nacional.csv              # Agregación total mercado
│       ├── resumen_team.csv                  # Agregación por equipo
│       ├── resumen_channel.csv               # Agregación por canal
│       └── resumen_customer.csv              # Agregación por cliente

```


## 📊 Uso del Sistema

### 1️⃣ Pipeline Completo (Ejecución Automática)

```python
# Abrir notebook principal
jupyter notebook clases/main_pipelines/Main_Pipeline.ipynb

# Ejecutar todas las celdas para pipeline completo:
# ✅ Preparación de datos
# ✅ Análisis exploratorio
# ✅ Scoring dinámico
# ✅ Detección de anomalías
# ✅ Clustering jerárquico
# ✅ Reporte consolidado
```

### 2️⃣ Ejecución Modular (Componentes Individuales)

#### Preparación de Datos
```python
from clases.data_preparation.PrepPipeline import DataPreparationPipeline

pipeline = DataPreparationPipeline(nos_path, cust_path, prod_path)
final_df, missing_stats, product_analysis = pipeline.run()
```

#### Scoring Dinámico
```python
from clases.ml_models.scoring.NosScore import ScoreDynamic

score_calculator = ScoreDynamic(
    year_data=benchmark_data,
    df=customer_data,
    dimension_cols=['customer', 'channel', 'team']
)
scored_data = score_calculator.calcular_score()
```

#### Detección de Anomalías
```python
from clases.ml_models.anomaly_detection.anomaly_detection_orchestrator import AnomalyDetectionOrchestrator

orchestrator = AnomalyDetectionOrchestrator(
    train_df=train_data,
    test_df=test_data,
    features=['rate_nsrd', 'rate_sd', 'nos_su', 'variation_rate_nos', 'variation_rate_volume']
)

results = orchestrator.one_call_complete_analysis(
    segment_columns=['channel', 'team'],
    optimize_model=True,
    verbose=False
)
```

#### Clustering Jerárquico
```python
from clases.ml_models.clustering.analyzer import ClusteringAnalyzer

analyzer = ClusteringAnalyzer(
    data=customer_data,
    min_clusters=5,
    max_clusters=6,
    filter_outliers=True
)

results = analyzer.full_analysis(
    selection_method='sfs_rfe_grid',
    required_features=['SCORE'],
    plot_results=True
)
```

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.8+ |
| **ML/Data Science** | scikit-learn, NumPy, pandas |
| **Visualización** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter |
| **Metodología** | CRISP-DM |
| **Algoritmos** | Isolation Forest, Hierarchical Clustering (Ward, Complete, Average) |
| **Optimización** | Grid Search CV, Sequential Feature Selection, Recursive Feature Elimination |

---

## 🤝 Contribuciones

Este es un proyecto académico desarrollado como trabajo de maestría. Las contribuciones, sugerencias y feedback son bienvenidos.


---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 👤 Autor

**Mario Bolaños Gutiérrez**
- 📧 Email: mabg020997@gmail.com

---

<div align="center">

### ⭐ Si este proyecto te fue útil, considera darle una estrella ⭐

Made  for Retail Analytics

</div>
