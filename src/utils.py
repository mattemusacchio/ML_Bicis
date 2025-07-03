import pandas as pd
import numpy as np

dtypes = {
    'id_recorrido': 'string',
    'duracion_recorrido': 'int32',
    'id_estacion_origen': 'string',
    'nombre_estacion_origen': 'string',
    'direccion_estacion_origen': 'string',
    'long_estacion_origen': 'float32',
    'lat_estacion_origen': 'float32',
    'id_estacion_destino': 'string',  # puede tener NaN → usar float
    'nombre_estacion_destino': 'string',
    'direccion_estacion_destino': 'string',
    'long_estacion_destino': 'float32',
    'lat_estacion_destino': 'float32',
    'id_usuario': 'string',  # puede tener decimales o NaN
    'modelo_bicicleta': 'string',
    'genero': 'string'
}

final_columns = list(dtypes.keys()) + ['fecha_origen_recorrido', 'fecha_destino_recorrido']

def clean_and_load(path, drop_cols=None, rename_cols=None):
    df = pd.read_csv(path, parse_dates=['fecha_origen_recorrido', 'fecha_destino_recorrido'], low_memory=False)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    if rename_cols:
        df = df.rename(columns=rename_cols)

    df.columns = [col.lower().replace('é', 'e').replace('É', 'E') for col in df.columns]

    if 'duracion_recorrido' in df.columns:
        df['duracion_recorrido'] = (
            df['duracion_recorrido']
            .astype(str)
            .str.replace(',', '', regex=False)
        )

    for pair in [('lat_estacion_origen', 'long_estacion_origen'),
                 ('lat_estacion_destino', 'long_estacion_destino')]:
        
        col_lat, col_long = pair
        if col_lat in df.columns and df[col_lat].astype(str).str.contains(',', regex=False).any():
            split_coords = df[col_lat].astype(str).str.split(',', expand=True)
            df[col_lat] = split_coords[0]
            df[col_long] = split_coords[1] if col_long in df.columns else np.nan

    for col in ['lat_estacion_origen', 'long_estacion_origen', 'lat_estacion_destino', 'long_estacion_destino']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')

    df = df.astype({k: v for k, v in dtypes.items() if k in df.columns})

    df = df[[col for col in final_columns if col in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]

    if 'género' in df.columns and 'genero' in df.columns:
        df['genero'] = df['genero'].fillna(df['género'])
        df = df.drop(columns=['género'])

    df['id_estacion_origen'] = df['id_estacion_origen'].str.replace('BAEcobici', '', regex=False)
    df['id_estacion_destino'] = df['id_estacion_destino'].str.replace('BAEcobici', '', regex=False)
    df['id_recorrido'] = df['id_recorrido'].str.replace('BAEcobici', '', regex=False)
    df['id_usuario'] = df['id_usuario'].str.replace('BAEcobici', '', regex=False)

    if 'duracion_recorrido' in df.columns:
        df['duracion_recorrido'] = pd.to_numeric(df['duracion_recorrido'], errors='coerce')
        df = df[df['duracion_recorrido'] >= 60]

    # delete all rows where id_estacion_origen is NaN
    df = df[df['id_estacion_origen'].notna()]

    return df



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random
def normalize_features(train_df, val_df, exclude_cols=None):
    """
    Normaliza las features numéricas usando StandardScaler.
    """
    
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'hora', 'dia_semana', 'es_fin_semana', 'mes', 'dia_mes', 'año']
    
    print("\nIniciando normalización...")
    
    # Identificar columnas numéricas para normalizar
    numeric_cols = []
    categorical_cols = []
    
    for col in train_df.columns:
        if col in exclude_cols:
            continue
        elif train_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"Columnas numéricas a normalizar: {len(numeric_cols)}")
    print(f"Columnas categóricas (no normalizadas): {len(categorical_cols)}")
    print(f"Columnas excluidas: {len(exclude_cols)}")
    
    # Crear copias para no modificar los originales
    train_norm = train_df.copy()
    val_norm = val_df.copy()
    
    # Inicializar y ajustar el scaler SOLO con datos de train
    scaler = StandardScaler()
    
    if len(numeric_cols) > 0:
        # Fit del scaler solo en train
        scaler.fit(train_norm[numeric_cols])
        
        # Transform en ambos conjuntos
        train_norm[numeric_cols] = scaler.transform(train_norm[numeric_cols])
        val_norm[numeric_cols] = scaler.transform(val_norm[numeric_cols])
        
        print(f"✅ Normalización completada")
        print(f"   Media de features train (debe ser ~0): {train_norm[numeric_cols].mean().mean():.6f}")
        print(f"   Std de features train (debe ser ~1): {train_norm[numeric_cols].std().mean():.6f}")
    else:
        print("⚠️  No se encontraron columnas numéricas para normalizar")
    
    return train_norm, val_norm, scaler, numeric_cols


def split_train_val_by_week(df, random_seed=42):
    """
    Separa los datos en train y val usando 1 día random por semana para validación.
    """
    
    # Establecer semilla para reproducibilidad
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    print("Iniciando separación train/val...")
    print(f"Dataset original: {df.shape}")
    print(f"Rango temporal: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    # Crear columnas auxiliares para la separación
    df_work = df.copy()
    df_work['fecha'] = df_work['timestamp'].dt.date
    df_work['año'] = df_work['timestamp'].dt.year
    df_work['mes'] = df_work['timestamp'].dt.month
    df_work['semana_año'] = df_work['timestamp'].dt.isocalendar().week
    df_work['dia_semana'] = df_work['timestamp'].dt.dayofweek  # 0=Lunes, 6=Domingo
    
    # Obtener todas las combinaciones únicas de año-mes-semana
    semanas_unicas = df_work[['año', 'mes', 'semana_año']].drop_duplicates()
    
    print(f"Total de semanas únicas: {len(semanas_unicas)}")
    
    # Para cada semana, seleccionar un día aleatorio para validación
    dias_val = []
    
    for _, semana in semanas_unicas.iterrows():
        año, mes, num_semana = semana['año'], semana['mes'], semana['semana_año']
        
        # Obtener todos los días de esta semana en este mes
        dias_semana = df_work[
            (df_work['año'] == año) & 
            (df_work['mes'] == mes) & 
            (df_work['semana_año'] == num_semana)
        ]['fecha'].unique()
        
        if len(dias_semana) > 0:
            # Seleccionar un día aleatorio de esta semana
            dia_seleccionado = random.choice(dias_semana)
            dias_val.append(dia_seleccionado)
            
    print(f"Días seleccionados para validación: {len(dias_val)}")
    
    # Crear máscaras para train y val
    mask_val = df_work['fecha'].isin(dias_val)
    mask_train = ~mask_val
    
    # Separar los datasets
    train_df = df_work[mask_train].drop(columns=['fecha', 'semana_año']).reset_index(drop=True)
    val_df = df_work[mask_val].drop(columns=['fecha', 'semana_año']).reset_index(drop=True)
    
    print(f"Dataset train: {train_df.shape} ({mask_train.sum()/len(df)*100:.1f}%)")
    print(f"Dataset val: {val_df.shape} ({mask_val.sum()/len(df)*100:.1f}%)")
    
    # Verificar que no hay solapamiento temporal
    train_dates = set(train_df['timestamp'].dt.date)
    val_dates = set(val_df['timestamp'].dt.date)
    overlap = train_dates.intersection(val_dates)
    
    if len(overlap) > 0:
        print(f"⚠️  ADVERTENCIA: Hay {len(overlap)} días que aparecen en ambos conjuntos")
    else:
        print("✅ Separación correcta: No hay solapamiento de días entre train y val")
        
    return train_df, val_df


def create_time_series_dataset_fast(trips_df, time_window_minutes=30):
    """
    Versión optimizada de transformación de dataset de viajes a series temporales.
    """
    trips_df = trips_df.copy()

    trips_df['fecha_origen_recorrido'] = pd.to_datetime(trips_df['fecha_origen_recorrido'])
    trips_df['fecha_destino_recorrido'] = pd.to_datetime(trips_df['fecha_destino_recorrido'])
    
    # 1. Crear columnas de ventana para despachos (origen) y arribos (destino)
    # trips_df['timestamp_origen_window'] = trips_df['fecha_origen_recorrido'].dt.floor(f'{time_window_minutes}T')
    trips_df['timestamp_origen_window'] = (trips_df['fecha_origen_recorrido'].dt.floor(f'{time_window_minutes}T') + pd.Timedelta(minutes=time_window_minutes))

    trips_df['timestamp_destino_window'] = trips_df['fecha_destino_recorrido'].dt.floor(f'{time_window_minutes}T')
    trips_df['timestamp_destino_prev'] = (
    trips_df['fecha_destino_recorrido'].dt.floor(f'{time_window_minutes}T') + pd.Timedelta(minutes=time_window_minutes))
    arribos_prev = trips_df.groupby(['timestamp_destino_prev','id_estacion_destino']).size().reset_index(name='arribos_prev_count').rename(columns={'timestamp_destino_prev':'timestamp','id_estacion_destino':'id_estacion'})

    # 2. Obtener rango de timestamps
    fecha_min = trips_df['timestamp_origen_window'].min()
    fecha_max = trips_df['timestamp_destino_window'].max()
    timestamps = pd.date_range(start=fecha_min, end=fecha_max, freq=f'{time_window_minutes}T')
    
    # 3. Obtener lista única de estaciones (origen + destino)
    estaciones_origen = trips_df[['id_estacion_origen', 'nombre_estacion_origen',
                                  'direccion_estacion_origen', 'lat_estacion_origen',
                                  'long_estacion_origen']].drop_duplicates()
    estaciones_destino = trips_df[['id_estacion_destino', 'nombre_estacion_destino',
                                   'direccion_estacion_destino', 'lat_estacion_destino',
                                   'long_estacion_destino']].drop_duplicates()

    estaciones_origen.columns = ['id_estacion', 'nombre_estacion', 'direccion_estacion', 'lat_estacion', 'long_estacion']
    estaciones_destino.columns = ['id_estacion', 'nombre_estacion', 'direccion_estacion', 'lat_estacion', 'long_estacion']
    
    estaciones = pd.concat([estaciones_origen, estaciones_destino]).drop_duplicates(subset=['id_estacion']).dropna(subset=['id_estacion'])

    print(f"Total de timestamps: {len(timestamps)}")
    print(f"Total de estaciones: {len(estaciones)}")
    
    # 4. Crear esqueleto base con producto cartesiano entre timestamps y estaciones
    timestamps_df = pd.DataFrame({'timestamp': timestamps})
    ts_grid = timestamps_df.assign(key=1).merge(estaciones.assign(key=1), on='key').drop(columns='key')

    # 5. Precalcular estadísticas históricas (ventana de despachos)
    despachos = trips_df.groupby(['timestamp_origen_window', 'id_estacion_origen']).agg(
        despachos_count=('id_estacion_origen', 'count'),
        duracion_recorrido_mean=('duracion_recorrido', 'mean'),
        duracion_recorrido_std=('duracion_recorrido', 'std'),
        duracion_recorrido_count=('duracion_recorrido', 'count'),
        edad_usuario_mean=('edad_usuario', 'mean'),
        edad_usuario_std=('edad_usuario', 'std'),
        proporcion_mujeres=('genero', lambda x: (x == 'FEMALE').sum() / len(x) if len(x) > 0 else 0),
        modelo_mas_comun=('modelo_bicicleta', lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN')
    ).reset_index()

    despachos = despachos.rename(columns={
        'timestamp_origen_window': 'timestamp',
        'id_estacion_origen': 'id_estacion'
    })

    # 6. Precalcular arribos (ventana futura)
    arribos = trips_df.groupby(['timestamp_destino_window', 'id_estacion_destino']).size().reset_index(name='arribos_count')
    arribos = arribos.rename(columns={
        'timestamp_destino_window': 'timestamp',
        'id_estacion_destino': 'id_estacion'
    })

    # 7. Merge de todas las features al grid base
    ts_df = ts_grid.merge(despachos, on=['timestamp', 'id_estacion'], how='left')
    ts_df = ts_df.merge(arribos, on=['timestamp', 'id_estacion'], how='left')
    ts_df = ts_df.merge(arribos_prev, on=['timestamp','id_estacion'], how='left')
    ts_df['arribos_prev_count'] = ts_df['arribos_prev_count'].fillna(0).astype(int)

    # 8. Rellenar NaNs con valores por defecto
    ts_df['despachos_count'] = ts_df['despachos_count'].fillna(0).astype(int)
    ts_df['arribos_count'] = ts_df['arribos_count'].fillna(0).astype(int)
    ts_df['duracion_recorrido_mean'] = ts_df['duracion_recorrido_mean'].fillna(0)
    ts_df['duracion_recorrido_std'] = ts_df['duracion_recorrido_std'].fillna(0)
    ts_df['duracion_recorrido_count'] = ts_df['duracion_recorrido_count'].fillna(0).astype(int)
    ts_df['edad_usuario_mean'] = ts_df['edad_usuario_mean'].fillna(0)
    ts_df['edad_usuario_std'] = ts_df['edad_usuario_std'].fillna(0)
    ts_df['proporcion_mujeres'] = ts_df['proporcion_mujeres'].fillna(0)
    ts_df['modelo_mas_comun'] = ts_df['modelo_mas_comun'].fillna('UNKNOWN')

    # 9. Agregar variables temporales
    ts_df['hora'] = ts_df['timestamp'].dt.hour
    ts_df['dia_semana'] = ts_df['timestamp'].dt.dayofweek
    ts_df['es_fin_semana'] = (ts_df['dia_semana'] >= 5).astype(int)
    ts_df['mes'] = ts_df['timestamp'].dt.month
    ts_df['dia_mes'] = ts_df['timestamp'].dt.day
    ts_df['año'] = ts_df['timestamp'].dt.year

    print(f"\nDataset final:")
    print(f"Forma: {ts_df.shape}")
    print(f"Rango temporal: {ts_df['timestamp'].min()} a {ts_df['timestamp'].max()}")
    print(f"Estaciones únicas: {ts_df['id_estacion'].nunique()}")
    
            # 10. Crear columnas "prev_1" hasta "prev_6" para todas las features históricas
    features_to_shift = [
        'despachos_count', 'duracion_recorrido_mean', 'duracion_recorrido_std',
        'duracion_recorrido_count', 'edad_usuario_mean', 'edad_usuario_std',
        'proporcion_mujeres', 'arribos_count'
    ]

    ts_df = ts_df.sort_values(['id_estacion', 'timestamp'])

    # Crear columnas prev_1 a prev_6
    for lag in range(1, 7):  # De 1 a 6
        shifted = ts_df.groupby('id_estacion')[features_to_shift].shift(lag)
        shifted.columns = [f'{col}_prev_{lag}' for col in shifted.columns]
        ts_df = pd.concat([ts_df, shifted], axis=1)

    # Rellenar NaNs con valores neutros
    for col in ts_df.columns:
        if col.startswith(tuple(f"{f}_" for f in features_to_shift)) and col.endswith(tuple(f"_prev_{i}" for i in range(1, 7))):
            if ts_df[col].dtype == 'float':
                ts_df[col] = ts_df[col].fillna(0.0)
            else:
                ts_df[col] = ts_df[col].fillna(0)



    return ts_df