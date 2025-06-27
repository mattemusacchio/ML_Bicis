import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class BikeFeatureEngineering:
    """
    Pipeline de feature engineering para predicción de arribos de bicicletas
    Sin data leakage: para predecir arribos en [T, T+30] solo usa información < T
    """
    
    def __init__(self, time_window_minutes=30, n_clusters=16):
        self.time_window_minutes = time_window_minutes
        self.n_clusters = n_clusters
        self.station_clusters = {}
        self.kmeans_model = None
        
    def load_data(self, filepath):
        """Cargar dataset de recorridos"""
        print("Cargando dataset...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Dataset cargado: {df.shape}")
        return df
    
    def create_time_windows(self, df):
        """Crear ventanas temporales redondeando hacia abajo"""
        print("Creando ventanas temporales...")
        
        df = df.copy()
        df['fecha_origen_dt'] = pd.to_datetime(df['fecha_origen_recorrido'])
        df['fecha_destino_dt'] = pd.to_datetime(df['fecha_destino_recorrido'])
        df['fecha_alta_dt'] = pd.to_datetime(df['fecha_alta'], errors='coerce')
        
        # Ventana de despacho: redondear hacia abajo
        df['ventana_despacho'] = df['fecha_origen_dt'].dt.floor(f'{self.time_window_minutes}min')
        
        # Ventana de arribo: redondear hacia abajo
        df['ventana_arribo'] = df['fecha_destino_dt'].dt.floor(f'{self.time_window_minutes}min')
        
        return df
    
    def create_station_clusters(self, df):
        """Crear clusters geográficos de estaciones usando K-means"""
        print("Creando clusters geográficos...")
        
        # Obtener coordenadas únicas de todas las estaciones
        estaciones_origen = df[['id_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']].drop_duplicates()
        estaciones_destino = df[['id_estacion_destino', 'lat_estacion_destino', 'long_estacion_destino']].drop_duplicates()
        
        estaciones_origen.columns = ['id_estacion', 'lat', 'long']
        estaciones_destino.columns = ['id_estacion', 'lat', 'long']
        
        estaciones = pd.concat([estaciones_origen, estaciones_destino]).drop_duplicates().dropna()
        
        # Aplicar K-means
        coords = estaciones[['lat', 'long']].values
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(coords)
        
        # Mapear estaciones a clusters
        self.station_clusters = dict(zip(estaciones['id_estacion'], clusters))
        
        # Asignar clusters al dataset
        df['zona_origen_cluster'] = df['id_estacion_origen'].map(self.station_clusters).fillna(0).astype(int)
        df['zona_destino_cluster'] = df['id_estacion_destino'].map(self.station_clusters).fillna(0).astype(int)
        
        return df
    
    def create_basic_features(self, df):
        """Crear features básicas temporales y de usuario"""
        print("Creando features básicas...")
        
        # Features temporales (basadas en origen - información conocida)
        df['año_origen'] = df['fecha_origen_dt'].dt.year
        df['mes_origen'] = df['fecha_origen_dt'].dt.month
        df['dia_origen'] = df['fecha_origen_dt'].dt.day
        df['hora_origen'] = df['fecha_origen_dt'].dt.hour
        df['minuto_origen'] = df['fecha_origen_dt'].dt.minute
        df['segundo_origen'] = df['fecha_origen_dt'].dt.second
        df['dia_semana'] = df['fecha_origen_dt'].dt.dayofweek
        
        # Features temporales de destino (para crear features prev más tarde)
        df['año_destino'] = df['fecha_destino_dt'].dt.year
        df['mes_destino'] = df['fecha_destino_dt'].dt.month
        df['dia_destino'] = df['fecha_destino_dt'].dt.day
        df['hora_destino'] = df['fecha_destino_dt'].dt.hour
        df['minuto_destino'] = df['fecha_destino_dt'].dt.minute
        df['segundo_destino'] = df['fecha_destino_dt'].dt.second
        
        # Features de ventana de tiempo (objetivo)
        df['año_intervalo'] = df['ventana_arribo'].dt.year
        df['mes_intervalo'] = df['ventana_arribo'].dt.month
        df['dia_intervalo'] = df['ventana_arribo'].dt.day
        df['hora_intervalo'] = df['ventana_arribo'].dt.hour
        df['minuto_intervalo'] = df['ventana_arribo'].dt.minute
        df['fecha_intervalo'] = df['ventana_arribo']
        
        # Features contextuales
        df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
        df['estacion_del_año'] = ((df['mes_origen'] % 12) // 3 + 1)
        
        # Features de usuario
        df['genero_FEMALE'] = (df['genero'] == 'FEMALE').astype(int)
        df['genero_MALE'] = (df['genero'] == 'MALE').astype(int)
        df['genero_OTHER'] = (~df['genero'].isin(['FEMALE', 'MALE'])).astype(int)
        df['usuario_registrado'] = (~df['fecha_alta'].isna()).astype(int)
        
        # Edad (manejar valores inválidos)
        df['edad_usuario'] = df['edad_usuario'].fillna(-1)
        df['edad_usuario'] = df['edad_usuario'].replace([np.inf, -np.inf], -1)
        
        # Modelo de bicicleta
        df['modelo_bicicleta'] = df['modelo_bicicleta'].map({'ICONIC': 1, 'FIT': 0}).fillna(2)
        
        # Features de fecha alta
        df['año_alta'] = df['fecha_alta_dt'].dt.year.fillna(0)
        df['mes_alta'] = df['fecha_alta_dt'].dt.month.fillna(0)
        
        # Estación de referencia
        df['estacion_referencia'] = df['id_estacion_destino']
        
        return df
    
    def calculate_interval_counts(self, df):
        """Calcular arribos y salidas por intervalo de tiempo"""
        print("Calculando conteos por intervalo...")
        
        # Arribos por ventana e id_estacion_destino
        arribos = df.groupby(['ventana_arribo', 'id_estacion_destino']).size().reset_index(name='N_arribos_intervalo')
        
        # Salidas por ventana e id_estacion_origen  
        salidas = df.groupby(['ventana_despacho', 'id_estacion_origen']).size().reset_index(name='N_salidas_intervalo')
        
        # Merge con el dataset principal
        df = df.merge(
            arribos,
            left_on=['ventana_arribo', 'id_estacion_destino'],
            right_on=['ventana_arribo', 'id_estacion_destino'],
            how='left'
        )
        
        df = df.merge(
            salidas,
            left_on=['ventana_arribo', 'id_estacion_destino'],
            right_on=['ventana_despacho', 'id_estacion_origen'],
            how='left'
        )
        
        # Rellenar valores faltantes
        df['N_arribos_intervalo'] = df['N_arribos_intervalo'].fillna(0)
        df['N_salidas_intervalo'] = df['N_salidas_intervalo'].fillna(0)
        
        return df
    
    def create_historical_features(self, df):
        """Crear features históricas (prev_n) sin data leakage"""
        print("Creando features históricas...")
        
        # Ordenar por fecha para lags temporales correctos
        df = df.sort_values(['id_estacion_destino', 'ventana_arribo']).reset_index(drop=True)
        
        # Features de destino históricos (prev_1 a prev_3)
        for lag in [1, 2, 3]:
            df[f'id_estacion_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['id_estacion_destino'].shift(lag)
            df[f'barrio_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['zona_destino_cluster'].shift(lag)
            df[f'cantidad_estaciones_cercanas_destino_prev_{lag}'] = 0  # Placeholder por ahora
            
            df[f'mes_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['mes_destino'].shift(lag)
            df[f'dia_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['dia_destino'].shift(lag)
            df[f'hora_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['hora_destino'].shift(lag)
            df[f'minuto_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['minuto_destino'].shift(lag)
            df[f'segundo_destino_prev_{lag}'] = df.groupby('id_estacion_destino')['segundo_destino'].shift(lag)
        
        # Arribos y salidas históricos (prev_1 a prev_6)
        for lag in range(1, 7):
            df[f'N_ARRIBOS_prev_{lag}'] = df.groupby('id_estacion_destino')['N_arribos_intervalo'].shift(lag)
            df[f'N_SALIDAS_prev_{lag}'] = df.groupby('id_estacion_destino')['N_salidas_intervalo'].shift(lag)
        
        # Rellenar NaN con 0
        lag_columns = [col for col in df.columns if '_prev_' in col]
        for col in lag_columns:
            df[col] = df[col].fillna(0)
            
        return df
    
    def select_final_features(self, df):
        """Seleccionar features finales según especificación"""
        print("Seleccionando features finales...")
        
        final_columns = [
            'id_recorrido', 'duracion_recorrido', 'id_estacion_origen', 'id_estacion_destino',
            'id_usuario', 'modelo_bicicleta', 'dia_semana', 'es_finde', 'estacion_del_año',
            'edad_usuario', 'año_alta', 'mes_alta', 'genero_FEMALE', 'genero_MALE', 'genero_OTHER',
            'usuario_registrado', 'zona_destino_cluster', 'zona_origen_cluster',
            'cantidad_estaciones_cercanas_destino', 'cantidad_estaciones_cercanas_origen',
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'año_destino', 'mes_destino', 'dia_destino', 'hora_destino', 'minuto_destino', 'segundo_destino',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            'fecha_intervalo',
            'id_estacion_destino_prev_1', 'id_estacion_destino_prev_2', 'id_estacion_destino_prev_3',
            'barrio_destino_prev_1', 'barrio_destino_prev_2', 'barrio_destino_prev_3',
            'cantidad_estaciones_cercanas_destino_prev_1', 'cantidad_estaciones_cercanas_destino_prev_2',
            'cantidad_estaciones_cercanas_destino_prev_3',
            'mes_destino_prev_1', 'mes_destino_prev_2', 'mes_destino_prev_3',
            'dia_destino_prev_1', 'dia_destino_prev_2', 'dia_destino_prev_3',
            'hora_destino_prev_1', 'hora_destino_prev_2', 'hora_destino_prev_3',
            'minuto_destino_prev_1', 'minuto_destino_prev_2', 'minuto_destino_prev_3',
            'segundo_destino_prev_1', 'segundo_destino_prev_2', 'segundo_destino_prev_3',
            'estacion_referencia', 'N_arribos_intervalo', 'N_salidas_intervalo',
            'N_ARRIBOS_prev_1', 'N_SALIDAS_prev_1', 'N_ARRIBOS_prev_2', 'N_SALIDAS_prev_2',
            'N_ARRIBOS_prev_3', 'N_SALIDAS_prev_3', 'N_ARRIBOS_prev_4', 'N_SALIDAS_prev_4',
            'N_ARRIBOS_prev_5', 'N_SALIDAS_prev_5', 'N_ARRIBOS_prev_6', 'N_SALIDAS_prev_6'
        ]
        
        # Asegurar que existen todas las columnas
        missing_cols = set(final_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
            
        return df[final_columns].copy()
    
    def transform(self, input_path, output_path):
        """Pipeline completo de transformación"""
        print("=== INICIANDO FEATURE ENGINEERING ===")
        
        # Cargar datos
        df = self.load_data(input_path)
        
        # Aplicar transformaciones en orden
        df = self.create_time_windows(df)
        df = self.create_station_clusters(df)
        df = self.create_basic_features(df)
        df = self.calculate_interval_counts(df)
        df = self.create_historical_features(df)
        df = self.select_final_features(df)
        
        # Guardar resultado
        print(f"Guardando dataset transformado: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("=== FEATURE ENGINEERING COMPLETADO ===")
        print(f"Dataset final: {df.shape}")
        print(f"Columnas: {len(df.columns)}")
        
        return df

if __name__ == "__main__":
    # Ejecutar pipeline
    fe = BikeFeatureEngineering()
    df_processed = fe.transform(
        input_path='../../data/processed/trips_enriched.csv',
        output_path='../../data/processed/trips_final_features.csv'
    ) 