# Enfoque Simplificado - Predicción de Arribos a Estación Individual

## 📋 Descripción del Proyecto

Este proyecto implementa un **enfoque simplificado** para la predicción de arribos en estaciones de EcoBici, enfocándose en **una sola estación** en lugar de todas las estaciones simultáneamente.

### 🎯 Objetivo
Predecir la cantidad de arribos a la **Estación 014 - Pacifico** (la estación más concurrida) en ventanas de 30 minutos, utilizando datos de despachos de los 30 minutos anteriores.

### 🔧 Estrategia de Simplificación

**Problema Original:** Predecir arribos para todas las 482 estaciones (problema de alta dimensionalidad con long-tails)

**Problema Simplificado:** Predecir arribos para 1 estación específica

### 📊 Estación Objetivo
- **ID:** 14
- **Nombre:** 014 - Pacifico
- **Ubicación:** Lat -34.577423, Long -58.426388
- **Actividad Total:** 311,181 viajes (154,234 arribos + 156,947 salidas)
- **Razón de Selección:** Es la estación más concurrida del sistema

### 🛠️ Estructura del Proyecto

```
enfoque_simplificado/
├── README.md                          # Este archivo
├── data/                              # Datos procesados
├── scripts/                           # Scripts de procesamiento
│   ├── feature_engineering.py         # Creación de features para una estación
│   ├── train_global_features.py       # Entrenamiento usando despachos globales  
│   ├── train_nearby_features.py       # Entrenamiento usando estaciones cercanas
│   └── utils.py                       # Funciones auxiliares
├── notebooks/                         # Análisis y experimentación
│   └── analisis_estacion_individual.ipynb
└── models/                            # Modelos entrenados (se crea automáticamente)
```

### 🚀 Enfoques a Comparar

#### 1. **Despachos Globales**
- **Features:** Usar despachos de TODAS las estaciones como input
- **Target:** Arribos a la estación 014 - Pacifico
- **Ventaja:** Captura patrones globales del sistema
- **Desventaja:** Puede introducir ruido de estaciones irrelevantes

#### 2. **Estaciones Cercanas**
- **Features:** Usar solo despachos de estaciones geográficamente cercanas
- **Target:** Arribos a la estación 014 - Pacifico  
- **Ventaja:** Reduce ruido, enfoque más localizado
- **Desventaja:** Puede perder información relevante de estaciones lejanas

### 📈 Métricas de Evaluación
- **Primaria:** R² (coeficiente de determinación)
- **Secundarias:** RMSE (Root Mean Square Error), MAE (Mean Absolute Error)

### 🔄 Plan de Escalamiento
1. **Fase 1:** Entrenar para 1 estación (la más concurrida)
2. **Fase 2:** Entrenar para Top 2 estaciones más concurridas
3. **Fase 3:** Entrenar para Top 5 estaciones más concurridas
4. **Fase N:** Ir escalando gradualmente hasta encontrar el punto óptimo

### 📚 Features Incluidas
- **Temporales:** hora, día_semana, es_fin_semana, mes, día_mes, año
- **Despachos:** conteo de salidas por estación y ventana temporal
- **Usuario:** edad promedio, proporción mujeres, duración promedio de viajes
- **Históricas:** Lags de 1 a 6 períodos (30 min cada uno) de todas las features
- **Geográficas:** coordenadas lat/long de las estaciones
- **Target:** arribos_count (cantidad de arribos en la ventana actual)

### 🏃‍♂️ Cómo Ejecutar

#### 1. Feature Engineering
```bash
cd scripts
python feature_engineering.py
```

#### 2. Entrenamiento con Despachos Globales
```bash
python train_global_features.py
```

#### 3. Entrenamiento con Estaciones Cercanas
```bash
python train_nearby_features.py
```

#### 4. Análisis en Jupyter
```bash
cd notebooks
jupyter notebook analisis_estacion_individual.ipynb
```

### 📝 Resultados Esperados
- **Mejor interpretabilidad:** Enfocarse en 1 estación facilita el análisis
- **Menos overfitting:** Reduce la complejidad del problema
- **Mejor baseline:** Establece una línea base sólida antes de escalar
- **Identificación de patterns:** Permite identificar qué tipos de features funcionan mejor 