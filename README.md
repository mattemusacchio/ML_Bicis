# Proyecto Final - Predicción de Bicicletas Públicas GCBA

**Autores:** Matteo Musacchio, Tiziano Demarco

## Descripción

Este proyecto implementa modelos de Machine Learning para predecir la cantidad de arribos de bicicletas en las estaciones del sistema de bicicletas públicas del Gobierno de la Ciudad de Buenos Aires (GCBA).

El objetivo principal es desarrollar un sistema predictivo que permita anticipar la demanda de bicicletas en cada estación, facilitando la planificación y redistribución del servicio.

## Características

- Modelos XGBoost entrenados para predicción de arribos
- Análisis exploratorio de datos de viajes y estaciones
- Interfaz web interactiva con visualizaciones en tiempo real
- Métricas de evaluación: MAE, RMSE, R²

## Estructura del Proyecto

```
proyecto_final/
├── Musacchio_Demarco_Notebbok_PF.ipynb         # Análisis exploratorio y entrenamiento de modelos
├── analisis_exploratorio.ipynb                 # Análisis adicional de datos
├── models/                                     # Modelos entrenados (XGBoost)
├── pagina/                                     # Aplicación web interactiva
├── scripts/                                    # Scripts de procesamiento
└── src/                                        # Utilidades
```

## Instalación

1. Clonar el repositorio
2. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Dependencias

### Dependencias principales
- pandas==2.2.3
- numpy==2.2.6
- scikit-learn==1.7.0
- seaborn==0.13.2
- matplotlib==3.10.3
- joblib==1.5.1

### Machine Learning
- xgboost==1.7.6
- lightgbm==4.6.0
- torch==2.7.1+cu128
- scipy==1.15.3
- statsmodels==0.14.4
- imbalanced-learn==0.13.0

### Desarrollo web
- fastapi==0.115.12
- uvicorn==0.34.3
- flask==3.1.1

## Uso

### Análisis y Entrenamiento
Ejecutar el notebook principal:
```bash
jupyter notebook data_exploring.ipynb
```

### Aplicación Web
Navegar al directorio de la página web:
```bash
cd pagina
npm install
npm run dev
```
Además tener correctamente instalados los modelos y los datasets procesados.

### Scripts de Procesamiento
Ejecutar scripts auxiliares:
```bash
python scripts/process_trips_heatmap.py
python scripts/extract_stations.py
```

## Modelos

El proyecto incluye varios modelos XGBoost optimizados:
- Modelo base de regresión
- Modelo con distribución Poisson
- Modelo con filtrado de intervalos inactivos
- Modelo normalizado

## Resultados

Los modelos generan predicciones de arribos por estación con métricas de evaluación asociadas, permitiendo el monitoreo continuo del rendimiento predictivo.
