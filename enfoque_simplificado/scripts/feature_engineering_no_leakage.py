"""
Feature Engineering SIN DATA LEAKAGE para Enfoque Simplificado
Versión corregida que elimina todas las fuentes de filtración de información.
"""

import pandas as pd
import numpy as np
from utils import (
    TARGET_STATION_ID, TARGET_STATION_NAME, TARGET_STATION_LAT, TARGET_STATION_LON,
    load_trips_data, create_time_windows, get_nearby_stations
)
import warnings
warnings.filterwarnings('ignore')

def create_single_station_dataset_no_leakage(trips_df, target_station_id, time_window_minutes=30, use_nearby_only=False):
    """
    Crea un dataset SIN DATA LEAKAGE enfocado en una sola estación de destino.
    
    CAMBIOS PRINCIPALES PARA ELIMINAR LEAKAGE:
    - NO usar target_arribos en features (excepto lags desplazados)
    - Medias móviles SOLO de features de entrada, NO del target
    - Eliminar ratios que usen el target actual
    - Asegurar que solo usemos información histórica
    """
    print(f"\n🎯 CREANDO DATASET SIN LEAKAGE PARA ESTACIÓN {target_station_id}")
    print("="*60)
    
    trips_df = trips_df.copy()
    
    # 1. Crear ventanas temporales
    trips_df = create_time_windows(trips_df, time_window_minutes)
    
    # 2. Obtener rango de timestamps
    fecha_min = trips_df['ventana_despacho'].min()
    fecha_max = trips_df['ventana_arribo'].max()
    timestamps = pd.date_range(start=fecha_min, end=fecha_max, freq=f'{time_window_minutes}min')
    
    print(f"📅 Rango temporal: {fecha_min} a {fecha_max}")
    print(f"⏱️  Total ventanas: {len(timestamps)}")
    
    # 3. Determinar qué estaciones usar como features
    if use_nearby_only:
        feature_stations = get_nearby_stations(
            trips_df, TARGET_STATION_LAT, TARGET_STATION_LON, 
            radius=0.015, max_stations=25
        )
        print(f"🔍 Usando {len(feature_stations)} estaciones cercanas como features")
    else:
        feature_stations = trips_df['id_estacion_origen'].dropna().unique()
        print(f"🌐 Usando todas las {len(feature_stations)} estaciones como features")
    
    # 4. Crear esqueleto temporal base
    base_df = pd.DataFrame({'timestamp': timestamps})
    
    # 5. Calcular despachos por estación y ventana (FEATURES X)
    print("🔄 Calculando features de despachos...")
    
    despachos = trips_df[trips_df['id_estacion_origen'].isin(feature_stations)].groupby(
        ['ventana_despacho', 'id_estacion_origen']
    ).agg(
        despachos_count=('id_estacion_origen', 'count'),
        duracion_promedio=('duracion_recorrido', 'mean'),
        duracion_std=('duracion_recorrido', 'std'),
        edad_promedio=('edad_usuario', 'mean'),
        edad_std=('edad_usuario', 'std'),
        proporcion_mujeres=('genero', lambda x: (x == 'F').sum() / len(x) if len(x) > 0 else 0),
    ).reset_index()
    
    despachos = despachos.rename(columns={
        'ventana_despacho': 'timestamp',
        'id_estacion_origen': 'id_estacion'
    })
    
    # Rellenar NaNs
    despachos['duracion_promedio'] = despachos['duracion_promedio'].fillna(0)
    despachos['duracion_std'] = despachos['duracion_std'].fillna(0)
    despachos['edad_promedio'] = despachos['edad_promedio'].fillna(0)
    despachos['edad_std'] = despachos['edad_std'].fillna(0)
    despachos['proporcion_mujeres'] = despachos['proporcion_mujeres'].fillna(0)
    
    # 6. Calcular arribos para la estación objetivo (TARGET Y)
    print(f"🎯 Calculando arribos para estación {target_station_id}...")
    
    arribos_target = trips_df[trips_df['id_estacion_destino'] == target_station_id].groupby(
        'ventana_arribo'
    ).size().reset_index(name='target_arribos')
    
    arribos_target = arribos_target.rename(columns={'ventana_arribo': 'timestamp'})
    
    # 7. Crear dataset agregado por timestamp
    print("🔗 Agregando features por timestamp...")
    
    # Agregar despachos por timestamp (suma todas las estaciones)
    despachos_agregados = despachos.groupby('timestamp').agg(
        total_despachos=('despachos_count', 'sum'),
        duracion_promedio_global=('duracion_promedio', 'mean'),
        duracion_std_global=('duracion_std', 'mean'),
        edad_promedio_global=('edad_promedio', 'mean'),
        edad_std_global=('edad_std', 'mean'),
        proporcion_mujeres_global=('proporcion_mujeres', 'mean'),
        estaciones_activas=('despachos_count', lambda x: (x > 0).sum())
    ).reset_index()
    
    # 8. Crear dataset final
    dataset = base_df.merge(despachos_agregados, on='timestamp', how='left')
    dataset = dataset.merge(arribos_target, on='timestamp', how='left')
    
    # Rellenar NaNs
    dataset['target_arribos'] = dataset['target_arribos'].fillna(0).astype(int)
    dataset['total_despachos'] = dataset['total_despachos'].fillna(0).astype(int)
    dataset['duracion_promedio_global'] = dataset['duracion_promedio_global'].fillna(0)
    dataset['duracion_std_global'] = dataset['duracion_std_global'].fillna(0)
    dataset['edad_promedio_global'] = dataset['edad_promedio_global'].fillna(0)
    dataset['edad_std_global'] = dataset['edad_std_global'].fillna(0)
    dataset['proporcion_mujeres_global'] = dataset['proporcion_mujeres_global'].fillna(0)
    dataset['estaciones_activas'] = dataset['estaciones_activas'].fillna(0).astype(int)
    
    # 9. Agregar features temporales
    print("📅 Agregando features temporales...")
    dataset['hora'] = dataset['timestamp'].dt.hour
    dataset['dia_semana'] = dataset['timestamp'].dt.dayofweek
    dataset['es_fin_semana'] = (dataset['dia_semana'] >= 5).astype(int)
    dataset['mes'] = dataset['timestamp'].dt.month
    dataset['dia_mes'] = dataset['timestamp'].dt.day
    dataset['año'] = dataset['timestamp'].dt.year
    
    # 10. ORDENAR POR TIMESTAMP ANTES DE CREAR LAGS (CRÍTICO)
    print("⏮️  Creando features de lags SIN LEAKAGE...")
    dataset = dataset.sort_values('timestamp').reset_index(drop=True)
    
    # 11. Crear features de lags SOLO HISTÓRICOS
    lag_features = [
        'total_despachos', 'duracion_promedio_global', 'duracion_std_global',
        'edad_promedio_global', 'edad_std_global', 'proporcion_mujeres_global',
        'estaciones_activas', 'target_arribos'
    ]
    
    for lag in range(1, 7):  # Lags de 1 a 6 períodos ANTERIORES
        for feature in lag_features:
            dataset[f'{feature}_lag_{lag}'] = dataset[feature].shift(lag)
    
    # Rellenar NaNs de lags con 0
    lag_columns = [col for col in dataset.columns if '_lag_' in col]
    dataset[lag_columns] = dataset[lag_columns].fillna(0)
    
    # 12. Crear features de tendencias SIN LEAKAGE
    print("📈 Creando features de tendencias SIN LEAKAGE...")
    
    # CORRECCIÓN: Medias móviles SOLO de features de INPUT, NO del target
    input_features = ['total_despachos', 'estaciones_activas', 'duracion_promedio_global']
    
    for feature in input_features:
        # Media móvil de los 3 períodos ANTERIORES
        dataset[f'{feature}_ma3'] = dataset[feature].shift(1).rolling(window=3, min_periods=1).mean()
        dataset[f'{feature}_ma6'] = dataset[feature].shift(1).rolling(window=6, min_periods=1).mean()
    
    # Features de tendencia de INPUT
    dataset['despachos_diff'] = dataset['total_despachos'].diff().fillna(0)
    dataset['estaciones_activas_diff'] = dataset['estaciones_activas'].diff().fillna(0)
    
    # 13. Features de ratios SIN USAR EL TARGET
    print("📊 Creando ratios SIN LEAKAGE...")
    
    # Ratio de despachos vs estaciones activas
    dataset['despachos_por_estacion'] = np.where(
        dataset['estaciones_activas'] > 0,
        dataset['total_despachos'] / dataset['estaciones_activas'],
        0
    )
    
    # 14. Features cíclicas
    print("🔄 Creando features cíclicas...")
    dataset['hora_sin'] = np.sin(2 * np.pi * dataset['hora'] / 24)
    dataset['hora_cos'] = np.cos(2 * np.pi * dataset['hora'] / 24)
    dataset['dia_semana_sin'] = np.sin(2 * np.pi * dataset['dia_semana'] / 7)
    dataset['dia_semana_cos'] = np.cos(2 * np.pi * dataset['dia_semana'] / 7)
    
    print(f"\n✅ Dataset SIN LEAKAGE creado con forma: {dataset.shape}")
    print(f"📊 Features disponibles: {dataset.shape[1] - 2}")
    print(f"🎯 Target: {dataset['target_arribos'].sum():,} arribos totales")
    print(f"📈 Target promedio por ventana: {dataset['target_arribos'].mean():.2f}")
    print(f"📉 Target std: {dataset['target_arribos'].std():.2f}")
    
    return dataset

def main():
    """
    Función principal que ejecuta todo el pipeline de feature engineering SIN LEAKAGE.
    """
    print("🚀 INICIANDO FEATURE ENGINEERING SIN LEAKAGE - ENFOQUE SIMPLIFICADO")
    print("="*80)
    
    # 1. Cargar datos
    trips = load_trips_data()
    
    # 2. Crear dataset global SIN LEAKAGE
    print(f"\n🌐 CREANDO DATASET GLOBAL SIN LEAKAGE...")
    dataset_global = create_single_station_dataset_no_leakage(
        trips, TARGET_STATION_ID, use_nearby_only=False
    )
    
    # Guardar dataset global
    dataset_global.to_csv('../data/dataset_global_no_leakage.csv', index=False)
    print(f"💾 Dataset global sin leakage guardado")
    
    # 3. Crear dataset con estaciones cercanas SIN LEAKAGE
    print(f"\n🔍 CREANDO DATASET CERCANAS SIN LEAKAGE...")
    dataset_nearby = create_single_station_dataset_no_leakage(
        trips, TARGET_STATION_ID, use_nearby_only=True
    )
    
    # Guardar dataset cercanas
    dataset_nearby.to_csv('../data/dataset_nearby_no_leakage.csv', index=False)
    print(f"💾 Dataset cercanas sin leakage guardado")
    
    print("\n✅ FEATURE ENGINEERING SIN LEAKAGE COMPLETADO")
    print("="*60)
    print("🎯 Ahora ejecuta los scripts de entrenamiento para ver resultados realistas")

if __name__ == "__main__":
    main() 