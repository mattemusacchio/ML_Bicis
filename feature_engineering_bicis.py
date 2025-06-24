import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BikeDataFeatureEngineer:
    """
    Clase para realizar feature engineering completo en datos de recorridos de bicicletas
    """
    
    def __init__(self):
        self.estaciones_barrios = {}
        self.estaciones_clusters = {}
        self.estaciones_cercanas = {}
        
    def load_data(self, filepath):
        """Cargar dataset de recorridos"""
        print("Cargando dataset...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Dataset cargado: {df.shape}")
        return df
    
    def prepare_datetime_features(self, df):
        """Preparar features de fecha y hora"""
        print("Preparando features de fecha y hora...")
        
        # Convertir a datetime
        df['fecha_origen_dt'] = pd.to_datetime(df['fecha_origen_recorrido'])
        df['fecha_destino_dt'] = pd.to_datetime(df['fecha_destino_recorrido'])
        df['fecha_alta_dt'] = pd.to_datetime(df['fecha_alta'])
        
        # Features de origen
        df['año_origen'] = df['fecha_origen_dt'].dt.year
        df['mes_origen'] = df['fecha_origen_dt'].dt.month
        df['dia_origen'] = df['fecha_origen_dt'].dt.day
        df['hora_origen'] = df['fecha_origen_dt'].dt.hour
        df['minuto_origen'] = df['fecha_origen_dt'].dt.minute
        df['segundo_origen'] = df['fecha_origen_dt'].dt.second
        df['dia_semana'] = df['fecha_origen_dt'].dt.dayofweek
        
        # Features de destino
        df['año_destino'] = df['fecha_destino_dt'].dt.year
        df['mes_destino'] = df['fecha_destino_dt'].dt.month
        df['dia_destino'] = df['fecha_destino_dt'].dt.day
        df['hora_destino'] = df['fecha_destino_dt'].dt.hour
        df['minuto_destino'] = df['fecha_destino_dt'].dt.minute
        df['segundo_destino'] = df['fecha_destino_dt'].dt.second
        
        # Features de fecha alta
        df['año_alta'] = df['fecha_alta_dt'].dt.year
        df['mes_alta'] = df['fecha_alta_dt'].dt.month
        
        # Features adicionales de tiempo
        df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
        df['estacion_del_anio'] = ((df['mes_origen'] % 12) // 3 + 1)
        
        return df
    
    def create_gender_features(self, df):
        """Crear features one-hot para género"""
        print("Creando features de género...")
        
        # One-hot encoding para género
        df['genero_FEMALE'] = (df['genero'] == 'FEMALE').astype(int)
        df['genero_MALE'] = (df['genero'] == 'MALE').astype(int)
        df['genero_OTHER'] = (~df['genero'].isin(['FEMALE', 'MALE'])).astype(int)
        
        return df
    
    def create_user_features(self, df):
        """Crear features de usuario"""
        print("Creando features de usuario...")
        
        # Usuario registrado (si tiene fecha de alta válida)
        df['usuario_registrado'] = (~df['fecha_alta'].isna()).astype(int)
        
        # Edad (manejar valores inválidos)
        df['edad_usuario'] = df['edad_usuario'].fillna(-1)
        df['edad_usuario'] = df['edad_usuario'].replace([np.inf, -np.inf], -1)
        
        # Codificar modelo de bicicleta (ICONIC=1, FIT=0, otros=2)
        df['modelo_bicicleta'] = df['modelo_bicicleta'].map({'ICONIC': 1, 'FIT': 0}).fillna(2)
        
        return df
    
    def create_station_clusters(self, df):
        """Crear clusters de estaciones y asignar barrios"""
        print("Creando clusters de estaciones y barrios...")
        
        # Obtener coordenadas únicas de estaciones
        estaciones = df[['id_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']].drop_duplicates()
        estaciones.columns = ['id_estacion', 'lat', 'long']
        
        # Crear clusters simples basados en coordenadas (ejemplo usando quantiles)
        lat_bins = pd.qcut(estaciones['lat'], q=4, labels=['S', 'SC', 'NC', 'N'])
        long_bins = pd.qcut(estaciones['long'], q=4, labels=['W', 'WC', 'EC', 'E'])
        
        estaciones['zona_cluster'] = (lat_bins.astype(str) + long_bins.astype(str)).factorize()[0]
        estaciones['barrio'] = estaciones['zona_cluster']  # Simplificado por ahora
        
        # Mapear a diccionarios
        self.estaciones_clusters = dict(zip(estaciones['id_estacion'], estaciones['zona_cluster']))
        self.estaciones_barrios = dict(zip(estaciones['id_estacion'], estaciones['barrio']))
        
        # Asignar a dataset
        df['zona_origen_cluster'] = df['id_estacion_origen'].map(self.estaciones_clusters).fillna(0)
        df['zona_destino_cluster'] = df['id_estacion_destino'].map(self.estaciones_clusters).fillna(0)
        df['barrio_origen'] = df['id_estacion_origen'].map(self.estaciones_barrios).fillna(0)
        df['barrio_destino'] = df['id_estacion_destino'].map(self.estaciones_barrios).fillna(0)
        
        return df
    
    def calculate_nearby_stations(self, df):
        """Calcular cantidad de estaciones cercanas"""
        print("Calculando estaciones cercanas...")
        
        # Para simplificar, usaremos el número de estaciones en el mismo cluster
        cluster_counts = df.groupby('zona_origen_cluster')['id_estacion_origen'].nunique().to_dict()
        
        df['cantidad_estaciones_cercanas_origen'] = df['zona_origen_cluster'].map(cluster_counts).fillna(1)
        df['cantidad_estaciones_cercanas_destino'] = df['zona_destino_cluster'].map(cluster_counts).fillna(1)
        
        return df
    
    def create_time_windows(self, df):
        """Crear ventanas de tiempo para agrupación"""
        print("Creando ventanas de tiempo...")
        
        # Usar ventana_arribo como base para crear features de intervalo
        df['fecha_intervalo'] = pd.to_datetime(df['ventana_arribo'])
        df['año_intervalo'] = df['fecha_intervalo'].dt.year
        df['mes_intervalo'] = df['fecha_intervalo'].dt.month
        df['dia_intervalo'] = df['fecha_intervalo'].dt.day
        df['hora_intervalo'] = df['fecha_intervalo'].dt.hour
        df['minuto_intervalo'] = df['fecha_intervalo'].dt.minute
        
        return df
    
    def create_lag_features(self, df):
        """Crear features de lag para estaciones destino"""
        print("Creando features de lag...")
        
        # Ordenar por fecha para crear lags temporales correctos
        df = df.sort_values(['fecha_destino_dt']).reset_index(drop=True)
        
        # Crear lags para las características de destino
        lag_features = ['id_estacion_destino', 'barrio_destino', 'cantidad_estaciones_cercanas_destino',
                       'año_destino', 'mes_destino', 'dia_destino', 'hora_destino', 
                       'minuto_destino', 'segundo_destino']
        
        for feature in lag_features:
            for lag in [1, 2, 3]:
                df[f'{feature}_LAG{lag}'] = df[feature].shift(lag)
        
        return df
    
    def calculate_arrival_departure_counts(self, df):
        """Calcular conteos de arribos y salidas por intervalo"""
        print("Calculando conteos de arribos y salidas...")
        
        # Agrupaciones por ventana de tiempo y estación
        arrivals = df.groupby(['ventana_arribo', 'id_estacion_destino']).size().reset_index(name='N_arribos_intervalo')
        departures = df.groupby(['ventana_despacho', 'id_estacion_origen']).size().reset_index(name='N_salidas_intervalo')
        
        # Merge con el dataset principal
        df = df.merge(
            arrivals.rename(columns={'ventana_arribo': 'ventana_arribo', 'id_estacion_destino': 'estacion_referencia'}),
            left_on=['ventana_arribo', 'id_estacion_destino'],
            right_on=['ventana_arribo', 'estacion_referencia'],
            how='left'
        )
        
        df = df.merge(
            departures.rename(columns={'ventana_despacho': 'ventana_arribo', 'id_estacion_origen': 'estacion_referencia'}),
            left_on=['ventana_arribo', 'id_estacion_destino'],
            right_on=['ventana_arribo', 'estacion_referencia'],
            how='left',
            suffixes=('', '_dep')
        )
        
        # Crear estacion_referencia si no existe
        if 'estacion_referencia' not in df.columns:
            df['estacion_referencia'] = df['id_estacion_destino']
        
        # Rellenar valores faltantes
        df['N_arribos_intervalo'] = df['N_arribos_intervalo'].fillna(0)
        df['N_salidas_intervalo'] = df['N_salidas_intervalo'].fillna(0)
        
        return df
    
    def create_lag_arrival_departure_counts(self, df):
        """Crear features de lag para conteos de arribos y salidas"""
        print("Creando lags de arribos y salidas...")
        
        # Ordenar por fecha y estación
        df = df.sort_values(['id_estacion_destino', 'fecha_intervalo']).reset_index(drop=True)
        
        # Crear lags para arribos y salidas
        for lag in range(1, 7):  # LAG1 a LAG6
            df[f'N_ARRIBOS_LAG{lag}'] = df.groupby('id_estacion_destino')['N_arribos_intervalo'].shift(lag)
            df[f'N_SALIDAS_LAG{lag}'] = df.groupby('id_estacion_destino')['N_salidas_intervalo'].shift(lag)
        
        # Calcular promedios de 2 intervalos anteriores
        df['N_SALIDAS_PROM_2INT'] = (df['N_SALIDAS_LAG1'] + df['N_SALIDAS_LAG2']) / 2
        df['N_ARRIBOS_PROM_2INT'] = (df['N_ARRIBOS_LAG1'] + df['N_ARRIBOS_LAG2']) / 2
        
        return df
    
    def select_final_features(self, df):
        """Seleccionar y ordenar las features finales"""
        print("Seleccionando features finales...")
        
        # Features objetivo según la especificación
        final_columns = [
            'id_recorrido', 'duracion_recorrido', 'id_estacion_origen', 'id_estacion_destino',
            'id_usuario', 'modelo_bicicleta', 'barrio_origen', 'barrio_destino', 'dia_semana',
            'es_finde', 'estacion_del_anio', 'edad_usuario', 'año_alta', 'mes_alta',
            'genero_FEMALE', 'genero_MALE', 'genero_OTHER', 'usuario_registrado',
            'zona_destino_cluster', 'zona_origen_cluster', 'cantidad_estaciones_cercanas_destino',
            'cantidad_estaciones_cercanas_origen', 'año_origen', 'mes_origen', 'dia_origen',
            'hora_origen', 'minuto_origen', 'segundo_origen', 'año_destino', 'mes_destino',
            'dia_destino', 'hora_destino', 'minuto_destino', 'segundo_destino',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo',
            'minuto_intervalo', 'fecha_intervalo', 'N_SALIDAS_PROM_2INT', 'N_ARRIBOS_PROM_2INT',
            'id_estacion_destino_LAG1', 'id_estacion_destino_LAG2', 'id_estacion_destino_LAG3',
            'barrio_destino_LAG1', 'barrio_destino_LAG2', 'barrio_destino_LAG3',
            'cantidad_estaciones_cercanas_destino_LAG1', 'cantidad_estaciones_cercanas_destino_LAG2',
            'cantidad_estaciones_cercanas_destino_LAG3', 'año_destino_LAG1', 'año_destino_LAG2',
            'año_destino_LAG3', 'mes_destino_LAG1', 'mes_destino_LAG2', 'mes_destino_LAG3',
            'dia_destino_LAG1', 'dia_destino_LAG2', 'dia_destino_LAG3', 'hora_destino_LAG1',
            'hora_destino_LAG2', 'hora_destino_LAG3', 'minuto_destino_LAG1', 'minuto_destino_LAG2',
            'minuto_destino_LAG3', 'segundo_destino_LAG1', 'segundo_destino_LAG2', 'segundo_destino_LAG3',
            'estacion_referencia', 'N_arribos_intervalo', 'N_salidas_intervalo',
            'N_ARRIBOS_LAG1', 'N_SALIDAS_LAG1', 'N_ARRIBOS_LAG2', 'N_SALIDAS_LAG2',
            'N_ARRIBOS_LAG3', 'N_SALIDAS_LAG3', 'N_ARRIBOS_LAG4', 'N_SALIDAS_LAG4',
            'N_ARRIBOS_LAG5', 'N_SALIDAS_LAG5', 'N_ARRIBOS_LAG6', 'N_SALIDAS_LAG6'
        ]
        
        # Asegurar que todas las columnas existen
        missing_cols = set(final_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Rellenar con 0 las columnas faltantes
            
        # Seleccionar solo las columnas finales
        df_final = df[final_columns].copy()
        
        # Rellenar NaN con valores apropiados
        df_final = df_final.fillna(0)
        
        return df_final
    
    def get_training_features(self):
        """Retornar la lista de features para entrenamiento"""
        return [
            'id_estacion_origen', 'id_usuario', 'modelo_bicicleta', 'barrio_origen',
            'dia_semana', 'es_finde', 'estacion_del_anio', 'edad_usuario', 'año_alta',
            'mes_alta', 'genero_FEMALE', 'genero_MALE', 'genero_OTHER',
            'usuario_registrado', 'zona_origen_cluster',
            'cantidad_estaciones_cercanas_origen', 'año_origen', 'mes_origen',
            'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo',
            'minuto_intervalo', 'N_SALIDAS_PROM_2INT', 'N_ARRIBOS_PROM_2INT',
            'id_estacion_destino_LAG1', 'id_estacion_destino_LAG2',
            'id_estacion_destino_LAG3', 'barrio_destino_LAG1', 'barrio_destino_LAG2',
            'barrio_destino_LAG3', 'cantidad_estaciones_cercanas_destino_LAG1',
            'cantidad_estaciones_cercanas_destino_LAG2',
            'cantidad_estaciones_cercanas_destino_LAG3', 'año_destino_LAG1',
            'año_destino_LAG2', 'año_destino_LAG3', 'mes_destino_LAG1',
            'mes_destino_LAG2', 'mes_destino_LAG3', 'dia_destino_LAG1',
            'dia_destino_LAG2', 'dia_destino_LAG3', 'hora_destino_LAG1',
            'hora_destino_LAG2', 'hora_destino_LAG3', 'minuto_destino_LAG1',
            'minuto_destino_LAG2', 'minuto_destino_LAG3', 'segundo_destino_LAG1',
            'segundo_destino_LAG2', 'segundo_destino_LAG3', 'N_ARRIBOS_LAG1',
            'N_SALIDAS_LAG1', 'N_ARRIBOS_LAG2', 'N_SALIDAS_LAG2', 'N_ARRIBOS_LAG3',
            'N_SALIDAS_LAG3', 'N_ARRIBOS_LAG4', 'N_SALIDAS_LAG4', 'N_ARRIBOS_LAG5',
            'N_SALIDAS_LAG5', 'N_ARRIBOS_LAG6', 'N_SALIDAS_LAG6'
        ]
    
    def transform_dataset(self, input_path, output_path):
        """Proceso completo de transformación"""
        print("=== Iniciando proceso de Feature Engineering ===")
        
        # Cargar datos
        df = self.load_data(input_path)
        
        # Aplicar todas las transformaciones
        df = self.prepare_datetime_features(df)
        df = self.create_gender_features(df)
        df = self.create_user_features(df)
        df = self.create_station_clusters(df)
        df = self.calculate_nearby_stations(df)
        df = self.create_time_windows(df)
        df = self.create_lag_features(df)
        df = self.calculate_arrival_departure_counts(df)
        df = self.create_lag_arrival_departure_counts(df)
        df = self.select_final_features(df)
        
        # Guardar resultado
        print(f"Guardando dataset transformado en: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("=== Proceso completado ===")
        print(f"Dataset final: {df.shape}")
        print(f"Columnas: {len(df.columns)}")
        
        return df

# Función principal para ejecutar la transformación
def main():
    """Ejecutar el proceso de feature engineering"""
    
    # Configuración
    input_path = 'data/processed/trips_con_ventanas.csv'
    output_path = 'data/processed/trips_features_engineered.csv'
    
    # Crear instancia del procesador
    feature_engineer = BikeDataFeatureEngineer()
    
    # Ejecutar transformación
    df_transformed = feature_engineer.transform_dataset(input_path, output_path)
    
    # Mostrar features de entrenamiento
    training_features = feature_engineer.get_training_features()
    print(f"\nFeatures para entrenamiento ({len(training_features)}):")
    for i, feature in enumerate(training_features):
        print(f"{i+1:2d}. {feature}")
    
    print(f"\nDataset listo para entrenamiento con XGBoost!")
    print(f"Archivo guardado: {output_path}")
    
    return df_transformed, training_features

if __name__ == "__main__":
    df_transformed, training_features = main() 