"""
Feature Engineering para Enfoque Simplificado - Una Estación
Adaptado del notebook original para predecir arribos a una sola estación.
"""

import pandas as pd
import numpy as np
from utils import (
    TARGET_STATION_ID, TARGET_STATION_NAME, TARGET_STATION_LAT, TARGET_STATION_LON,
    load_trips_data, create_time_windows, get_nearby_stations
)
import warnings
warnings.filterwarnings('ignore')

def create_single_station_dataset(trips_df, target_station_id, time_window_minutes=30, use_nearby_only=False):
    """
    Crea un dataset de series temporales enfocado en una sola estación de destino.
    
    Args:
        trips_df: DataFrame con datos de trips
        target_station_id: ID de la estación objetivo (para predecir arribos)
        time_window_minutes: Tamaño de ventana temporal en minutos
        use_nearby_only: Si True, usa solo estaciones cercanas como features
    
    Returns:
        pd.DataFrame: Dataset listo para entrenamiento
    """
    print(f"\n🎯 CREANDO DATASET PARA ESTACIÓN {target_station_id}")
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
    dataset['es_feriado'] = 0  # Placeholder - se puede mejorar
    
    # 10. Crear features de lags (1-6 períodos históricos)
    print("⏮️  Creando features de lags...")
    
    # Ordenar por timestamp para crear lags correctamente
    dataset = dataset.sort_values('timestamp').reset_index(drop=True)
    
    lag_features = [
        'total_despachos', 'duracion_promedio_global', 'duracion_std_global',
        'edad_promedio_global', 'edad_std_global', 'proporcion_mujeres_global',
        'estaciones_activas', 'target_arribos'
    ]
    
    for lag in range(1, 7):  # Lags de 1 a 6 períodos
        for feature in lag_features:
            dataset[f'{feature}_lag_{lag}'] = dataset[feature].shift(lag)
    
    # Rellenar NaNs de lags con 0
    lag_columns = [col for col in dataset.columns if '_lag_' in col]
    dataset[lag_columns] = dataset[lag_columns].fillna(0)
    
    # 11. Crear features de tendencias y estadísticas móviles
    print("📈 Creando features de tendencias...")
    
    # Promedios móviles de 3 períodos
    for feature in ['total_despachos', 'target_arribos']:
        dataset[f'{feature}_ma3'] = dataset[feature].rolling(window=3, min_periods=1).mean()
        dataset[f'{feature}_ma6'] = dataset[feature].rolling(window=6, min_periods=1).mean()
    
    # Features de tendencia (diferencias)
    dataset['despachos_diff'] = dataset['total_despachos'].diff().fillna(0)
    dataset['arribos_diff'] = dataset['target_arribos'].diff().fillna(0)
    
    # 12. Features de ratios y relaciones
    dataset['ratio_arribos_despachos'] = np.where(
        dataset['total_despachos'] > 0,
        dataset['target_arribos'] / dataset['total_despachos'],
        0
    )
    
    # Feature de actividad relativa de la estación
    dataset['actividad_relativa'] = np.where(
        dataset['estaciones_activas'] > 0,
        dataset['target_arribos'] / dataset['estaciones_activas'],
        0
    )
    
    print(f"\n✅ Dataset creado con forma: {dataset.shape}")
    print(f"📊 Features disponibles: {dataset.shape[1] - 2}")  # -2 por timestamp y target
    print(f"🎯 Target: {dataset['target_arribos'].sum():,} arribos totales")
    print(f"📈 Target promedio por ventana: {dataset['target_arribos'].mean():.2f}")
    print(f"📉 Target std: {dataset['target_arribos'].std():.2f}")
    
    return dataset

def prepare_train_val_split(dataset, train_ratio=0.8):
    """
    Divide el dataset en entrenamiento y validación respetando el orden temporal.
    
    Args:
        dataset: DataFrame con el dataset completo
        train_ratio: Proporción para entrenamiento
    
    Returns:
        tuple: (train_df, val_df)
    """
    print(f"\n📊 DIVISIÓN TRAIN/VAL ({train_ratio:.1%} / {1-train_ratio:.1%})")
    print("="*50)
    
    # Ordenar por timestamp
    dataset = dataset.sort_values('timestamp').reset_index(drop=True)
    
    # Dividir temporalmente
    split_idx = int(len(dataset) * train_ratio)
    
    train_df = dataset.iloc[:split_idx].copy()
    val_df = dataset.iloc[split_idx:].copy()
    
    print(f"🚂 ENTRENAMIENTO:")
    print(f"   Período: {train_df['timestamp'].min()} a {train_df['timestamp'].max()}")
    print(f"   Registros: {len(train_df):,}")
    print(f"   Arribos promedio: {train_df['target_arribos'].mean():.2f}")
    
    print(f"\n🔍 VALIDACIÓN:")
    print(f"   Período: {val_df['timestamp'].min()} a {val_df['timestamp'].max()}")
    print(f"   Registros: {len(val_df):,}")
    print(f"   Arribos promedio: {val_df['target_arribos'].mean():.2f}")
    
    return train_df, val_df

def prepare_features_target(train_df, val_df):
    """
    Separa features (X) y target (y) de los datasets de entrenamiento y validación.
    
    Args:
        train_df: DataFrame de entrenamiento
        val_df: DataFrame de validación
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, feature_names)
    """
    print("\n🎯 PREPARANDO FEATURES Y TARGET")
    print("="*40)
    
    # Identificar columnas a excluir
    exclude_cols = ['timestamp', 'target_arribos']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Separar X e y
    X_train = train_df[feature_cols]
    y_train = train_df['target_arribos']
    
    X_val = val_df[feature_cols]
    y_val = val_df['target_arribos']
    
    print(f"✅ Features seleccionadas: {len(feature_cols)}")
    print(f"📊 X_train: {X_train.shape}")
    print(f"📊 X_val: {X_val.shape}")
    print(f"🎯 y_train: {y_train.shape} (promedio: {y_train.mean():.2f})")
    print(f"🎯 y_val: {y_val.shape} (promedio: {y_val.mean():.2f})")
    
    return X_train, y_train, X_val, y_val, feature_cols

def main():
    """
    Función principal que ejecuta todo el pipeline de feature engineering.
    """
    print("🚀 INICIANDO FEATURE ENGINEERING - ENFOQUE SIMPLIFICADO")
    print("="*70)
    
    # 1. Cargar datos
    trips = load_trips_data()
    
    # 2. Crear dataset global (todas las estaciones como features)
    print(f"\n🌐 CREANDO DATASET CON FEATURES GLOBALES...")
    dataset_global = create_single_station_dataset(
        trips, TARGET_STATION_ID, use_nearby_only=False
    )
    
    # Guardar dataset global
    dataset_global.to_csv('../data/dataset_global.csv', index=False)
    print(f"💾 Dataset global guardado: ../data/dataset_global.csv")
    
    # 3. Crear dataset con estaciones cercanas
    print(f"\n🔍 CREANDO DATASET CON ESTACIONES CERCANAS...")
    dataset_nearby = create_single_station_dataset(
        trips, TARGET_STATION_ID, use_nearby_only=True
    )
    
    # Guardar dataset cercanas
    dataset_nearby.to_csv('../data/dataset_nearby.csv', index=False)
    print(f"💾 Dataset cercanas guardado: ../data/dataset_nearby.csv")
    
    # 4. Crear splits de entrenamiento/validación para ambos
    print(f"\n📊 CREANDO SPLITS TRAIN/VAL...")
    
    # Global
    train_global, val_global = prepare_train_val_split(dataset_global)
    train_global.to_csv('../data/train_global.csv', index=False)
    val_global.to_csv('../data/val_global.csv', index=False)
    
    # Cercanas
    train_nearby, val_nearby = prepare_train_val_split(dataset_nearby)
    train_nearby.to_csv('../data/train_nearby.csv', index=False)
    val_nearby.to_csv('../data/val_nearby.csv', index=False)
    
    print("\n✅ FEATURE ENGINEERING COMPLETADO")
    print("="*50)
    print("📁 Archivos generados:")
    print("   📄 ../data/dataset_global.csv")
    print("   📄 ../data/dataset_nearby.csv")
    print("   📄 ../data/train_global.csv")
    print("   📄 ../data/val_global.csv")
    print("   📄 ../data/train_nearby.csv")
    print("   📄 ../data/val_nearby.csv")
    
    print(f"\n🎯 Próximos pasos:")
    print("   1. Ejecutar: python train_global_features.py")
    print("   2. Ejecutar: python train_nearby_features.py")
    print("   3. Comparar resultados en el notebook")

if __name__ == "__main__":
    main() 