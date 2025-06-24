import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BikeDataFeatureEngineerNoLeakage:
    """
    Feature engineering corregido SIN DATA LEAKAGE para datos de bicicletas
    """
    
    def __init__(self):
        self.estaciones_barrios = {}
        self.estaciones_clusters = {}
        
    def load_data(self, filepath):
        """Cargar dataset de recorridos"""
        print("Cargando dataset...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Dataset cargado: {df.shape}")
        return df
    
    def prepare_basic_features(self, df):
        """Preparar features básicas sin data leakage"""
        print("Preparando features básicas...")
        
        # Convertir fechas
        df['fecha_origen_dt'] = pd.to_datetime(df['fecha_origen_recorrido'])
        df['fecha_destino_dt'] = pd.to_datetime(df['fecha_destino_recorrido'])
        df['fecha_alta_dt'] = pd.to_datetime(df['fecha_alta'], errors='coerce')
        df['ventana_arribo_dt'] = pd.to_datetime(df['ventana_arribo'])
        df['ventana_despacho_dt'] = pd.to_datetime(df['ventana_despacho'])
        
        # Features de recorrido (conocidas al momento del origen)
        df['año_origen'] = df['fecha_origen_dt'].dt.year
        df['mes_origen'] = df['fecha_origen_dt'].dt.month
        df['dia_origen'] = df['fecha_origen_dt'].dt.day
        df['hora_origen'] = df['fecha_origen_dt'].dt.hour
        df['minuto_origen'] = df['fecha_origen_dt'].dt.minute
        df['segundo_origen'] = df['fecha_origen_dt'].dt.second
        df['dia_semana'] = df['fecha_origen_dt'].dt.dayofweek
        
        # Features contextuales
        df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
        df['estacion_del_anio'] = ((df['mes_origen'] % 12) // 3 + 1)
        
        # Features de usuario (conocidas antes del viaje)
        df['genero_FEMALE'] = (df['genero'] == 'FEMALE').astype(int)
        df['genero_MALE'] = (df['genero'] == 'MALE').astype(int)
        df['genero_OTHER'] = (~df['genero'].isin(['FEMALE', 'MALE'])).astype(int)
        df['usuario_registrado'] = (~df['fecha_alta'].isna()).astype(int)
        df['edad_usuario'] = df['edad_usuario'].fillna(-1)
        df['edad_usuario'] = df['edad_usuario'].replace([np.inf, -np.inf], -1)
        df['modelo_bicicleta'] = df['modelo_bicicleta'].map({'ICONIC': 1, 'FIT': 0}).fillna(2)
        
        # Features de fecha de alta
        df['año_alta'] = df['fecha_alta_dt'].dt.year.fillna(0)
        df['mes_alta'] = df['fecha_alta_dt'].dt.month.fillna(0)
        
        return df
    
    def create_station_features(self, df):
        """Crear features de estaciones y clusters geográficos"""
        print("Creando features de estaciones...")
        
        # Obtener coordenadas únicas de estaciones
        estaciones_origen = df[['id_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']].drop_duplicates()
        estaciones_destino = df[['id_estacion_destino', 'lat_estacion_destino', 'long_estacion_destino']].drop_duplicates()
        
        # Combinar todas las estaciones
        estaciones_origen.columns = ['id_estacion', 'lat', 'long']
        estaciones_destino.columns = ['id_estacion', 'lat', 'long']
        estaciones = pd.concat([estaciones_origen, estaciones_destino]).drop_duplicates().dropna()
        
        # Crear clusters geográficos
        lat_bins = pd.qcut(estaciones['lat'], q=4, labels=False, duplicates='drop')
        long_bins = pd.qcut(estaciones['long'], q=4, labels=False, duplicates='drop')
        
        estaciones['zona_cluster'] = lat_bins * 4 + long_bins
        estaciones['barrio'] = estaciones['zona_cluster']  
        
        # Mapear a diccionarios
        self.estaciones_clusters = dict(zip(estaciones['id_estacion'], estaciones['zona_cluster']))
        self.estaciones_barrios = dict(zip(estaciones['id_estacion'], estaciones['barrio']))
        
        # Asignar clusters
        df['zona_origen_cluster'] = df['id_estacion_origen'].map(self.estaciones_clusters).fillna(0)
        df['zona_destino_cluster'] = df['id_estacion_destino'].map(self.estaciones_clusters).fillna(0)
        df['barrio_origen'] = df['id_estacion_origen'].map(self.estaciones_barrios).fillna(0)
        df['barrio_destino'] = df['id_estacion_destino'].map(self.estaciones_barrios).fillna(0)
        
        # Calcular estaciones cercanas por cluster
        cluster_counts = df.groupby('zona_origen_cluster')['id_estacion_origen'].nunique().to_dict()
        df['cantidad_estaciones_cercanas_origen'] = df['zona_origen_cluster'].map(cluster_counts).fillna(1)
        df['cantidad_estaciones_cercanas_destino'] = df['zona_destino_cluster'].map(cluster_counts).fillna(1)
        
        return df
    
    def create_time_window_features(self, df):
        """Crear features de ventana temporal de arribo"""
        print("Creando features de ventana temporal...")
        
        # Features de la ventana de arribo (esto es lo que queremos predecir)
        df['año_intervalo'] = df['ventana_arribo_dt'].dt.year
        df['mes_intervalo'] = df['ventana_arribo_dt'].dt.month
        df['dia_intervalo'] = df['ventana_arribo_dt'].dt.day
        df['hora_intervalo'] = df['ventana_arribo_dt'].dt.hour
        df['minuto_intervalo'] = df['ventana_arribo_dt'].dt.minute
        df['fecha_intervalo'] = df['ventana_arribo_dt']
        df['estacion_referencia'] = df['id_estacion_destino']
        
        return df
    
    def create_temporal_aggregations(self, df):
        """Crear agregaciones temporales SIN LEAKAGE"""
        print("Creando agregaciones temporales correctas...")
        
        # Crear tabla de conteos por ventana
        arribos_por_ventana = df.groupby(['ventana_arribo', 'id_estacion_destino']).size().reset_index(name='N_arribos_ventana')
        arribos_por_ventana['ventana_arribo_dt'] = pd.to_datetime(arribos_por_ventana['ventana_arribo'])
        
        salidas_por_ventana = df.groupby(['ventana_despacho', 'id_estacion_origen']).size().reset_index(name='N_salidas_ventana')
        salidas_por_ventana['ventana_despacho_dt'] = pd.to_datetime(salidas_por_ventana['ventana_despacho'])
        
        # Crear serie temporal completa
        min_fecha = min(arribos_por_ventana['ventana_arribo_dt'].min(), salidas_por_ventana['ventana_despacho_dt'].min())
        max_fecha = max(arribos_por_ventana['ventana_arribo_dt'].max(), salidas_por_ventana['ventana_despacho_dt'].max())
        
        todas_ventanas = pd.date_range(start=min_fecha, end=max_fecha, freq='30T')
        todas_estaciones = sorted(df['id_estacion_destino'].dropna().unique())
        
        # Producto cartesiano: estación x ventana
        serie_temporal = []
        for estacion in todas_estaciones:
            for ventana in todas_ventanas:
                serie_temporal.append({
                    'id_estacion': estacion,
                    'ventana_tiempo': ventana
                })
        
        df_serie = pd.DataFrame(serie_temporal)
        
        # Merge con conteos reales
        df_serie = df_serie.merge(
            arribos_por_ventana[['ventana_arribo_dt', 'id_estacion_destino', 'N_arribos_ventana']],
            left_on=['ventana_tiempo', 'id_estacion'],
            right_on=['ventana_arribo_dt', 'id_estacion_destino'],
            how='left'
        )
        df_serie['N_arribos_actual'] = df_serie['N_arribos_ventana'].fillna(0)
        
        df_serie = df_serie.merge(
            salidas_por_ventana[['ventana_despacho_dt', 'id_estacion_origen', 'N_salidas_ventana']],
            left_on=['ventana_tiempo', 'id_estacion'],
            right_on=['ventana_despacho_dt', 'id_estacion_origen'],
            how='left'
        )
        df_serie['N_salidas_actual'] = df_serie['N_salidas_ventana'].fillna(0)
        
        # Crear LAGs CORRECTOS por estación
        print("Creando LAGs correctos por estación y tiempo...")
        
        df_serie = df_serie.sort_values(['id_estacion', 'ventana_tiempo']).reset_index(drop=True)
        
        # LAGs de arribos y salidas por estación
        for lag in range(1, 7):
            df_serie[f'N_ARRIBOS_LAG{lag}'] = df_serie.groupby('id_estacion')['N_arribos_actual'].shift(lag)
            df_serie[f'N_SALIDAS_LAG{lag}'] = df_serie.groupby('id_estacion')['N_salidas_actual'].shift(lag)
        
        # Promedios de 2 intervalos previos
        df_serie['N_ARRIBOS_PROM_2INT'] = (df_serie['N_ARRIBOS_LAG1'] + df_serie['N_ARRIBOS_LAG2']) / 2
        df_serie['N_SALIDAS_PROM_2INT'] = (df_serie['N_SALIDAS_LAG1'] + df_serie['N_SALIDAS_LAG2']) / 2
        
        # Merge de vuelta con dataset original
        print("Haciendo merge con dataset original...")
        
        df['merge_key'] = df['ventana_arribo'] + '_' + df['id_estacion_destino'].astype(str)
        df_serie['merge_key'] = df_serie['ventana_tiempo'].astype(str) + '_' + df_serie['id_estacion'].astype(str)
        
        lag_columns = [col for col in df_serie.columns if 'LAG' in col or 'PROM' in col]
        merge_columns = ['merge_key', 'N_arribos_actual', 'N_salidas_actual'] + lag_columns
        
        df = df.merge(df_serie[merge_columns], on='merge_key', how='left')
        
        df['N_arribos_intervalo'] = df['N_arribos_actual'] 
        df['N_salidas_intervalo'] = df['N_salidas_actual']
        
        df = df.drop(['merge_key', 'N_arribos_actual', 'N_salidas_actual'], axis=1, errors='ignore')
        
        return df
    
    def create_destination_lags(self, df):
        """Crear LAGs de características de destino SIN LEAKAGE"""
        print("Creando LAGs de características de destino...")
        
        df = df.sort_values(['fecha_origen_dt']).reset_index(drop=True)
        
        lag_features = ['id_estacion_origen', 'barrio_origen', 'cantidad_estaciones_cercanas_origen']
        
        for feature in lag_features:
            for lag in [1, 2, 3]:
                df[f'{feature}_LAG{lag}'] = df[feature].shift(lag)
        
        return df
    
    def select_final_features(self, df):
        """Seleccionar features finales SIN LEAKAGE"""
        print("Seleccionando features finales...")
        
        safe_features = [
            'id_recorrido', 'duracion_recorrido', 'id_estacion_origen', 'id_estacion_destino',
            'id_usuario', 'modelo_bicicleta', 'estacion_referencia',
            'genero_FEMALE', 'genero_MALE', 'genero_OTHER', 'usuario_registrado',
            'edad_usuario', 'año_alta', 'mes_alta',
            'barrio_origen', 'zona_origen_cluster', 'cantidad_estaciones_cercanas_origen',
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'dia_semana', 'es_finde', 'estacion_del_anio',
            'barrio_destino', 'zona_destino_cluster', 'cantidad_estaciones_cercanas_destino',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            'fecha_intervalo',
            'N_ARRIBOS_LAG1', 'N_SALIDAS_LAG1', 'N_ARRIBOS_LAG2', 'N_SALIDAS_LAG2',
            'N_ARRIBOS_LAG3', 'N_SALIDAS_LAG3', 'N_ARRIBOS_LAG4', 'N_SALIDAS_LAG4',
            'N_ARRIBOS_LAG5', 'N_SALIDAS_LAG5', 'N_ARRIBOS_LAG6', 'N_SALIDAS_LAG6',
            'N_ARRIBOS_PROM_2INT', 'N_SALIDAS_PROM_2INT',
            'id_estacion_origen_LAG1', 'id_estacion_origen_LAG2', 'id_estacion_origen_LAG3',
            'barrio_origen_LAG1', 'barrio_origen_LAG2', 'barrio_origen_LAG3',
            'cantidad_estaciones_cercanas_origen_LAG1', 'cantidad_estaciones_cercanas_origen_LAG2',
            'cantidad_estaciones_cercanas_origen_LAG3',
            'N_arribos_intervalo', 'N_salidas_intervalo'
        ]
        
        available_features = [col for col in safe_features if col in df.columns]
        missing_features = set(safe_features) - set(available_features)
        
        if missing_features:
            print(f"⚠️  Features faltantes (se rellenarán con 0): {missing_features}")
            for col in missing_features:
                df[col] = 0
        
        df_final = df[safe_features].copy()
        df_final = df_final.fillna(0)
        
        return df_final
    
    def get_training_features(self):
        """Features para entrenamiento"""
        return [
            'id_usuario', 'modelo_bicicleta', 'genero_FEMALE', 'genero_MALE', 'genero_OTHER',
            'usuario_registrado', 'edad_usuario', 'año_alta', 'mes_alta',
            'id_estacion_origen', 'barrio_origen', 'zona_origen_cluster', 'cantidad_estaciones_cercanas_origen',
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'dia_semana', 'es_finde', 'estacion_del_anio',
            'barrio_destino', 'zona_destino_cluster', 'cantidad_estaciones_cercanas_destino',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            'N_ARRIBOS_LAG1', 'N_SALIDAS_LAG1', 'N_ARRIBOS_LAG2', 'N_SALIDAS_LAG2',
            'N_ARRIBOS_LAG3', 'N_SALIDAS_LAG3', 'N_ARRIBOS_LAG4', 'N_SALIDAS_LAG4',
            'N_ARRIBOS_LAG5', 'N_SALIDAS_LAG5', 'N_ARRIBOS_LAG6', 'N_SALIDAS_LAG6',
            'N_ARRIBOS_PROM_2INT', 'N_SALIDAS_PROM_2INT',
            'id_estacion_origen_LAG1', 'id_estacion_origen_LAG2', 'id_estacion_origen_LAG3',
            'barrio_origen_LAG1', 'barrio_origen_LAG2', 'barrio_origen_LAG3',
            'cantidad_estaciones_cercanas_origen_LAG1', 'cantidad_estaciones_cercanas_origen_LAG2',
            'cantidad_estaciones_cercanas_origen_LAG3'
        ]
    
    def transform_dataset(self, input_path, output_path):
        """Proceso completo de transformación SIN DATA LEAKAGE"""
        print("=== 🛡️ FEATURE ENGINEERING SIN DATA LEAKAGE ===")
        
        df = self.load_data(input_path)
        df = self.prepare_basic_features(df)
        df = self.create_station_features(df)
        df = self.create_time_window_features(df)
        df = self.create_temporal_aggregations(df)
        df = self.create_destination_lags(df)
        df = self.select_final_features(df)
        
        print(f"Guardando dataset SIN LEAKAGE en: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("=== ✅ PROCESO COMPLETADO SIN DATA LEAKAGE ===")
        print(f"Dataset final: {df.shape}")
        print(f"Target: N_arribos_intervalo (arribos en ventana [T, T+30])")
        
        return df

def main():
    """Función principal"""
    
    input_path = 'data/processed/trips_con_ventanas.csv'
    output_path = 'data/processed/trips_no_leakage.csv'
    
    feature_engineer = BikeDataFeatureEngineerNoLeakage()
    df_transformed = feature_engineer.transform_dataset(input_path, output_path)
    
    training_features = feature_engineer.get_training_features()
    print(f"\n🎯 Features para entrenamiento ({len(training_features)}):")
    for i, feature in enumerate(training_features):
        print(f"{i+1:2d}. {feature}")
    
    print(f"\n✅ Dataset SIN DATA LEAKAGE listo!")
    print(f"📄 Archivo guardado: {output_path}")
    
    return df_transformed, training_features

if __name__ == "__main__":
    df_transformed, training_features = main() 