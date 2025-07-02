"""
Utilidades comunes para el enfoque simplificado de predicción de arribos.
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Configuración de la estación objetivo
TARGET_STATION_ID = 14.0
TARGET_STATION_NAME = "014 - Pacifico"
TARGET_STATION_LAT = -34.577423
TARGET_STATION_LON = -58.426388

def load_trips_data(data_path="../../data/processed/trips_enriched.csv"):
    """
    Carga y prepara los datos de trips_enriched.csv
    
    Returns:
        pd.DataFrame: Dataset de trips con fechas parseadas
    """
    print("🔄 Cargando datos de trips_enriched...")
    trips = pd.read_csv(data_path)
    
    # Convertir fechas
    trips['fecha_origen_recorrido'] = pd.to_datetime(trips['fecha_origen_recorrido'])
    trips['fecha_destino_recorrido'] = pd.to_datetime(trips['fecha_destino_recorrido'])
    
    print(f"✅ Datos cargados: {trips.shape}")
    print(f"📅 Rango temporal: {trips['fecha_origen_recorrido'].min()} a {trips['fecha_origen_recorrido'].max()}")
    
    return trips

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia euclidiana entre dos puntos geográficos.
    
    Args:
        lat1, lon1: Coordenadas del primer punto
        lat2, lon2: Coordenadas del segundo punto
    
    Returns:
        float: Distancia euclidiana
    """
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def get_nearby_stations(trips_df, target_lat, target_lon, radius=0.01, max_stations=20):
    """
    Encuentra estaciones cercanas a una ubicación objetivo.
    
    Args:
        trips_df: DataFrame con datos de trips
        target_lat, target_lon: Coordenadas de la estación objetivo
        radius: Radio de búsqueda (en grados, aprox 0.01 = ~1km)
        max_stations: Máximo número de estaciones a retornar
    
    Returns:
        list: Lista de IDs de estaciones cercanas
    """
    print(f"🔍 Buscando estaciones cercanas a ({target_lat}, {target_lon})...")
    
    # Obtener estaciones únicas con coordenadas
    estaciones = trips_df[['id_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']].drop_duplicates()
    estaciones.columns = ['id_estacion', 'lat', 'lon']
    estaciones = estaciones.dropna()
    
    # Calcular distancias
    estaciones['distancia'] = calculate_distance(
        estaciones['lat'], estaciones['lon'], target_lat, target_lon
    )
    
    # Filtrar por radio y ordenar por distancia
    estaciones_cercanas = estaciones[estaciones['distancia'] <= radius].sort_values('distancia')
    
    if len(estaciones_cercanas) < 5:
        print(f"⚠️  Solo {len(estaciones_cercanas)} estaciones en radio {radius}, expandiendo...")
        # Si hay muy pocas, tomar las más cercanas
        estaciones_cercanas = estaciones.nsmallest(max_stations, 'distancia')
    
    estaciones_cercanas = estaciones_cercanas.head(max_stations)
    
    print(f"✅ Encontradas {len(estaciones_cercanas)} estaciones cercanas")
    print(f"📍 Distancias: {estaciones_cercanas['distancia'].min():.4f} - {estaciones_cercanas['distancia'].max():.4f}")
    
    return estaciones_cercanas['id_estacion'].tolist()

def create_time_windows(trips_df, time_window_minutes=30):
    """
    Crea ventanas temporales para despachos y arribos.
    
    Args:
        trips_df: DataFrame con datos de trips
        time_window_minutes: Tamaño de ventana en minutos
    
    Returns:
        pd.DataFrame: Dataset con ventanas temporales asignadas
    """
    print(f"⏰ Creando ventanas temporales de {time_window_minutes} minutos...")
    
    trips_df = trips_df.copy()
    
    # Ventana de despacho: el recorrido sirve como input para la ventana posterior
    trips_df['ventana_despacho'] = (
        trips_df['fecha_origen_recorrido'].dt.floor(f'{time_window_minutes}min') + 
        pd.Timedelta(minutes=time_window_minutes)
    )
    
    # Ventana de arribo: cae en la ventana actual
    trips_df['ventana_arribo'] = trips_df['fecha_destino_recorrido'].dt.floor(f'{time_window_minutes}min')
    
    print(f"✅ Ventanas temporales creadas")
    
    return trips_df

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name="Modelo"):
    """
    Evalúa un modelo y retorna métricas.
    
    Args:
        model: Modelo entrenado
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        model_name: Nombre del modelo para logging
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    print(f"\n📊 Evaluando {model_name}...")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Métricas
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'val_r2': r2_score(y_val, y_val_pred),
    }
    
    # Mostrar resultados
    print(f"\n📈 MÉTRICAS DE {model_name.upper()}")
    print("="*50)
    print(f"🚂 ENTRENAMIENTO:")
    print(f"   RMSE: {metrics['train_rmse']:.4f}")
    print(f"   MAE:  {metrics['train_mae']:.4f}")
    print(f"   R²:   {metrics['train_r2']:.4f}")
    
    print(f"\n🔍 VALIDACIÓN:")
    print(f"   RMSE: {metrics['val_rmse']:.4f}")
    print(f"   MAE:  {metrics['val_mae']:.4f}")
    print(f"   R²:   {metrics['val_r2']:.4f}")
    
    # Indicador de overfitting
    r2_diff = metrics['train_r2'] - metrics['val_r2']
    if r2_diff > 0.1:
        print(f"\n⚠️  Posible overfitting (diff R²: {r2_diff:.3f})")
    elif r2_diff < 0:
        print(f"\n✅ Buen ajuste (val R² > train R²)")
    else:
        print(f"\n✅ Buen balance train/val")
    
    return metrics, y_val_pred

def save_model(model, features, metrics, model_name, target_info=None):
    """
    Guarda un modelo entrenado con su metadata.
    
    Args:
        model: Modelo entrenado
        features: Lista de features utilizadas
        metrics: Diccionario con métricas del modelo
        model_name: Nombre del archivo (sin extensión)
        target_info: Información adicional sobre el target
    """
    # Crear carpeta de modelos si no existe
    os.makedirs("../models", exist_ok=True)
    
    model_info = {
        'model': model,
        'features': features,
        'metrics': metrics,
        'train_timestamp': datetime.now().isoformat(),
        'target_station_id': TARGET_STATION_ID,
        'target_station_name': TARGET_STATION_NAME,
        'target_info': target_info
    }
    
    filepath = f"../models/{model_name}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"\n💾 Modelo guardado en: {filepath}")
    print(f"📊 R² validación: {metrics['val_r2']:.4f}")

def load_model(model_path):
    """
    Carga un modelo previamente entrenado.
    
    Args:
        model_path: Ruta al archivo del modelo
    
    Returns:
        dict: Información completa del modelo
    """
    with open(model_path, 'rb') as f:
        model_info = pickle.load(f)
    
    print(f"📂 Modelo cargado: {model_path}")
    print(f"🎯 Estación objetivo: {model_info['target_station_name']}")
    print(f"📊 R² validación: {model_info['metrics']['val_r2']:.4f}")
    print(f"📅 Entrenado: {model_info['train_timestamp']}")
    
    return model_info

def print_feature_importance(model, features, top_n=15):
    """
    Muestra la importancia de features de un modelo XGBoost.
    
    Args:
        model: Modelo XGBoost entrenado
        features: Lista de nombres de features
        top_n: Número de features top a mostrar
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 TOP {top_n} FEATURES MÁS IMPORTANTES:")
        print("="*50)
        for i, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<35} {row['importance']:.4f}")
        
        return importance_df
    else:
        print("\n⚠️  El modelo no tiene feature_importances_")
        return None 