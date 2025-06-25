import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BikeDataFeatureEngineerNoLeakage:
    """
    Clase para realizar feature engineering sin data leakage en datos de recorridos de bicicletas.
    Objetivo: Predecir arribos en [T, T+30] usando solo información de [T-30, T] sin información de arribos.
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
        """Preparar features de fecha y hora SOLO para información histórica (sin destino)"""
        print("Preparando features de fecha y hora...")
        
        # Convertir a datetime
        df['fecha_origen_dt'] = pd.to_datetime(df['fecha_origen_recorrido'])
        df['fecha_alta_dt'] = pd.to_datetime(df['fecha_alta'])
        
        # Features de origen (información histórica válida)
        df['año_origen'] = df['fecha_origen_dt'].dt.year
        df['mes_origen'] = df['fecha_origen_dt'].dt.month
        df['dia_origen'] = df['fecha_origen_dt'].dt.day
        df['hora_origen'] = df['fecha_origen_dt'].dt.hour
        df['minuto_origen'] = df['fecha_origen_dt'].dt.minute
        df['segundo_origen'] = df['fecha_origen_dt'].dt.second
        df['dia_semana'] = df['fecha_origen_dt'].dt.dayofweek
        
        # Features de fecha alta (información histórica válida)
        df['año_alta'] = df['fecha_alta_dt'].dt.year
        df['mes_alta'] = df['fecha_alta_dt'].dt.month
        
        # Features adicionales de tiempo (basadas en origen)
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
        
        # Asignar SOLO para origen (no destino para evitar leakage)
        df['zona_origen_cluster'] = df['id_estacion_origen'].map(self.estaciones_clusters).fillna(0)
        df['barrio_origen'] = df['id_estacion_origen'].map(self.estaciones_barrios).fillna(0)
        
        return df
    
    def calculate_nearby_stations(self, df):
        """Calcular cantidad de estaciones cercanas SOLO para origen"""
        print("Calculando estaciones cercanas...")
        
        # Para simplificar, usaremos el número de estaciones en el mismo cluster
        cluster_counts = df.groupby('zona_origen_cluster')['id_estacion_origen'].nunique().to_dict()
        
        df['cantidad_estaciones_cercanas_origen'] = df['zona_origen_cluster'].map(cluster_counts).fillna(1)
        
        return df
    
    def create_time_windows(self, df):
        """Crear ventanas de tiempo SOLO para información histórica"""
        print("Creando ventanas de tiempo...")
        
        # Usar ventana_despacho como base (información disponible en T-30, T)
        df['fecha_intervalo'] = pd.to_datetime(df['ventana_despacho'])
        df['año_intervalo'] = df['fecha_intervalo'].dt.year
        df['mes_intervalo'] = df['fecha_intervalo'].dt.month
        df['dia_intervalo'] = df['fecha_intervalo'].dt.day
        df['hora_intervalo'] = df['fecha_intervalo'].dt.hour
        df['minuto_intervalo'] = df['fecha_intervalo'].dt.minute
        
        return df
        
    def calculate_historical_departures(self, df):
        """Calcular conteos históricos de SALIDAS por intervalo (sin arribos para evitar leakage)"""
        print("Calculando conteos históricos de salidas...")
        
        # Solo contar salidas (despachos), NO arribos para evitar data leakage
        departures = df.groupby(['ventana_despacho', 'id_estacion_origen']).size().reset_index(name='N_salidas_historicas')
        
        # Merge con el dataset principal usando solo información de salidas
        df = df.merge(
            departures,
            left_on=['ventana_despacho', 'id_estacion_origen'],
            right_on=['ventana_despacho', 'id_estacion_origen'],
            how='left'
        )
        
        # Rellenar valores faltantes
        df['N_salidas_historicas'] = df['N_salidas_historicas'].fillna(0)
        
        return df
    
    def create_station_popularity_features(self, df):
        """Crear features de popularidad de estaciones basadas en datos históricos"""
        print("Creando features de popularidad de estaciones...")
        
        # Popularidad de estación origen (basada en salidas históricas)
        station_popularity = df.groupby('id_estacion_origen')['id_recorrido'].count().to_dict()
        df['popularidad_estacion_origen'] = df['id_estacion_origen'].map(station_popularity).fillna(0)
        
        # Hora promedio de uso de la estación origen
        df['hora_origen_num'] = df['hora_origen']  # Para cálculos
        station_avg_hour = df.groupby('id_estacion_origen')['hora_origen_num'].mean().to_dict()
        df['hora_promedio_estacion_origen'] = df['id_estacion_origen'].map(station_avg_hour).fillna(12)
        
        return df
    
    def calculate_target_arrivals(self, df):
        """Calcular variable objetivo: N_arribos_intervalo (lo que queremos predecir)"""
        print("Calculando variable objetivo: N_arribos_intervalo...")
        
        # IMPORTANTE: Esto es el TARGET, no una feature!
        # Contar arribos por ventana de arribo y estación destino
        arrivals = df.groupby(['ventana_arribo', 'id_estacion_destino']).size().reset_index(name='N_arribos_intervalo')
        
        # Merge con el dataset principal
        df = df.merge(
            arrivals,
            left_on=['ventana_arribo', 'id_estacion_destino'],
            right_on=['ventana_arribo', 'id_estacion_destino'],
            how='left'
        )
        
        # Rellenar valores faltantes con 0
        df['N_arribos_intervalo'] = df['N_arribos_intervalo'].fillna(0)
        
        print(f"✅ Variable objetivo creada. Rango: {df['N_arribos_intervalo'].min()} - {df['N_arribos_intervalo'].max()}")
        print(f"   Media de arribos por intervalo: {df['N_arribos_intervalo'].mean():.2f}")
        
        return df

    def select_final_features(self, df):
        """Seleccionar features finales SIN data leakage"""
        print("Seleccionando features finales sin data leakage...")
        
        # Features sin data leakage - solo información disponible en [T-30, T]
        final_columns = [
            # Identificadores
            'id_recorrido', 'id_estacion_origen', 'id_estacion_destino',
            'id_usuario', 'duracion_recorrido',
            
            # Features de usuario (información histórica válida)
            'modelo_bicicleta', 'edad_usuario', 'año_alta', 'mes_alta',
            'genero_FEMALE', 'genero_MALE', 'genero_OTHER', 'usuario_registrado',
            
            # Features de estación origen (información histórica válida)
            'barrio_origen', 'zona_origen_cluster', 'cantidad_estaciones_cercanas_origen',
            'popularidad_estacion_origen', 'hora_promedio_estacion_origen',
            
            # Features temporales (basadas en origen/despacho, información histórica válida)
            'dia_semana', 'es_finde', 'estacion_del_anio',
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            
            # Features de actividad histórica (solo salidas, información válida)
            'N_salidas_historicas',
            
            # Variable objetivo (TARGET - NO es feature!)
            'N_arribos_intervalo',
            
            # Mantener fecha_intervalo para referencia temporal
            'fecha_intervalo'
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
        """Retornar la lista de features para entrenamiento SIN data leakage"""
        return [
            # Features de estación origen
            'id_estacion_origen', 'barrio_origen', 'zona_origen_cluster',
            'cantidad_estaciones_cercanas_origen', 'popularidad_estacion_origen',
            'hora_promedio_estacion_origen',
            
            # Features de usuario
            'id_usuario', 'modelo_bicicleta', 'edad_usuario', 'año_alta', 'mes_alta',
            'genero_FEMALE', 'genero_MALE', 'genero_OTHER', 'usuario_registrado',
            
            # Features temporales (basadas en origen/despacho)
            'dia_semana', 'es_finde', 'estacion_del_anio',
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            
            # Features de actividad histórica
            'N_salidas_historicas'
        ]
    
    def transform_dataset(self, input_path, output_path):
        """Proceso completo de transformación SIN data leakage"""
        print("=== Iniciando proceso de Feature Engineering SIN DATA LEAKAGE ===")
        print("Objetivo: Predecir arribos en [T, T+30] usando información de [T-30, T] sin arribos")
        
        # Cargar datos
        df = self.load_data(input_path)
        
        # Aplicar transformaciones sin data leakage
        df = self.prepare_datetime_features(df)
        df = self.create_gender_features(df)
        df = self.create_user_features(df)
        df = self.create_station_clusters(df)
        df = self.calculate_nearby_stations(df)
        df = self.create_time_windows(df)
        df = self.calculate_historical_departures(df)
        df = self.create_station_popularity_features(df)
        
        # IMPORTANTE: Calcular target DESPUÉS de las features para evitar leakage
        df = self.calculate_target_arrivals(df)
        
        df = self.select_final_features(df)
        
        # Guardar resultado
        print(f"Guardando dataset transformado en: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("=== Proceso completado SIN DATA LEAKAGE ===")
        print(f"Dataset final: {df.shape}")
        print(f"Columnas: {len(df.columns)}")
        print("NOTA: Se eliminaron todas las features de lag y información de arribos para evitar data leakage")
        
        return df

# Función principal para ejecutar la transformación
def main():
    """Ejecutar el proceso de feature engineering sin data leakage"""
    
    # Configuración
    input_path = 'data/processed/trips_con_ventanas.csv'
    output_path = 'data/processed/trips_features_no_leakage.csv'
    
    # Crear instancia del procesador
    feature_engineer = BikeDataFeatureEngineerNoLeakage()
    
    # Ejecutar transformación
    df_transformed = feature_engineer.transform_dataset(input_path, output_path)
    
    # Mostrar features de entrenamiento
    training_features = feature_engineer.get_training_features()
    print(f"\nFeatures para entrenamiento SIN DATA LEAKAGE ({len(training_features)}):")
    for i, feature in enumerate(training_features):
        print(f"{i+1:2d}. {feature}")
    
    print(f"\nDataset listo para entrenamiento sin data leakage!")
    print(f"Archivo guardado: {output_path}")
    print("\n=== FEATURES ELIMINADAS PARA EVITAR DATA LEAKAGE ===")
    print("- Todas las features LAG (LAG1, LAG2, LAG3, etc.)")
    print("- Features de arribos (N_arribos_intervalo, N_ARRIBOS_LAG*, etc.)")
    print("- Features de destino (hora_destino, barrio_destino, etc.)")
    print("- Promedios basados en arribos (N_ARRIBOS_PROM_2INT, etc.)")
    
    return df_transformed, training_features

if __name__ == "__main__":
    df_transformed, training_features = main() 