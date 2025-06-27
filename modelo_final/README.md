# Pipeline de Feature Engineering - Predicción de Arribos de Bicicletas

## Objetivo
Transformar datos de recorridos de bicicletas para predecir arribos en intervalos de 30 minutos sin data leakage.

## Transformación

### Datos de Entrada (20 columnas):
```
id_recorrido, duracion_recorrido, id_estacion_origen, nombre_estacion_origen, 
direccion_estacion_origen, long_estacion_origen, lat_estacion_origen, 
id_estacion_destino, nombre_estacion_destino, direccion_estacion_destino,
long_estacion_destino, lat_estacion_destino, id_usuario, modelo_bicicleta,
genero, fecha_origen_recorrido, fecha_destino_recorrido, edad_usuario, 
fecha_alta, hora_alta
```

### Datos de Salida (77 columnas):
Features engineered para ML + variable objetivo `N_arribos_intervalo`

## Características del Pipeline

### ✅ Sin Data Leakage
- Para predecir arribos en [T, T+30] solo usa información conocida antes de T
- Features `prev_n` contienen datos de intervalos anteriores
- Ventanas temporales redondeadas hacia abajo

### 🗺️ Clustering Geográfico
- K-means con 16 clusters basado en coordenadas lat/long
- Agrupa estaciones por zona geográfica

### ⏰ Ventanas Temporales
- Intervalos de 30 minutos: 14:00-14:30, 14:30-15:00, etc.
- Redondeo hacia abajo: arribo a las 14:25 → intervalo 14:00-14:30

### 📊 Features Históricas
- `prev_1` a `prev_6`: datos de 1 a 6 intervalos anteriores
- Variables: arribos, salidas, features temporales, estaciones destino

## Uso

### Ejecución directa:
```bash
cd modelo_final
python run_pipeline.py
```

### En Python:
```python
from src.feature_engineering import BikeFeatureEngineering

fe = BikeFeatureEngineering(time_window_minutes=30, n_clusters=16)
df_processed = fe.transform(
    '../data/processed/trips_enriched.csv',
    '../data/processed/trips_final_features.csv'
)
```

### Verificación:
```bash
python verify_output.py
```

## Archivos

- `src/feature_engineering.py`: Clase principal del pipeline
- `run_pipeline.py`: Script de ejecución
- `verify_output.py`: Verificación de columnas de salida
- `notebook.ipynb`: Notebook de demostración

## Variable Objetivo

**`N_arribos_intervalo`**: Número de arribos en la ventana de tiempo [T, T+30]

Este es el valor que queremos predecir usando las demás features.

## Dataset Final

- **Filas**: ~12.8M recorridos procesados
- **Columnas**: 77 features + variable objetivo
- **Archivo**: `../data/processed/trips_final_features.csv` 