import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

def asignar_ventanas_temporales(trips_df, time_window_minutes=30):
    """
    Añade columnas con la ventana temporal correspondiente al origen y destino del recorrido.
    Mantiene una fila por recorrido.
    CORREGIDO: Sin data leakage en las ventanas temporales.
    """
    trips_df = trips_df.copy()

    # Convertir fechas si no están en datetime
    trips_df['fecha_origen_recorrido'] = pd.to_datetime(trips_df['fecha_origen_recorrido'])
    trips_df['fecha_destino_recorrido'] = pd.to_datetime(trips_df['fecha_destino_recorrido'])

    # CORRECCIÓN: Asignar ventana de despacho sin agregar tiempo
    # Un viaje que sale a las 14:25 debe estar en la ventana 14:00-14:30
    trips_df['ventana_despacho'] = trips_df['fecha_origen_recorrido'].dt.floor(f'{time_window_minutes}min')

    # Asignar ventana de arribo: cae en la ventana actual
    # Un viaje que llega a las 14:55 debe estar en la ventana 14:30-15:00
    trips_df['ventana_arribo'] = trips_df['fecha_destino_recorrido'].dt.floor(f'{time_window_minutes}min')

    return trips_df

# Cargar tus datos
trips_enriched = pd.read_csv('data/processed/trips_enriched.csv')
# trips_verano = pd.read_csv('data/processed/trips_verano.csv')
trips_verano_con_ventanas = asignar_ventanas_temporales(trips_enriched, time_window_minutes=30)

# Guardar el resultado
trips_verano_con_ventanas.to_csv('data/processed/trips_con_ventanas.csv', index=False)
print("✅ Archivo guardado como trips_con_ventanas.csv - SIN DATA LEAKAGE")
print("🔧 Ventanas corregidas:")
print("   - ventana_despacho: fecha_origen redondeada hacia abajo")
print("   - ventana_arribo: fecha_destino redondeada hacia abajo")

class BikeDataFeatureEngineerNoLeakage:
    """
    Clase para realizar feature engineering sin data leakage en datos de recorridos de bicicletas.
    Objetivo: Predecir arribos en [T, T+30] usando solo información de [T-30, T] sin información de arribos.
    """
    
    def __init__(self):
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
        """Crear clusters de estaciones"""
        print("Creando clusters de estaciones...")
        
        # Obtener coordenadas únicas de estaciones
        estaciones = df[['id_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']].drop_duplicates()
        estaciones.columns = ['id_estacion', 'lat', 'long']
        
        # Crear clusters simples basados en coordenadas (ejemplo usando quantiles)
        lat_bins = pd.qcut(estaciones['lat'], q=4, labels=['S', 'SC', 'NC', 'N'])
        long_bins = pd.qcut(estaciones['long'], q=4, labels=['W', 'WC', 'EC', 'E'])
        
        estaciones['zona_cluster'] = (lat_bins.astype(str) + long_bins.astype(str)).factorize()[0]
        
        # Mapear a diccionarios
        self.estaciones_clusters = dict(zip(estaciones['id_estacion'], estaciones['zona_cluster']))
        
        # Asignar SOLO para origen (no destino para evitar leakage)
        df['zona_origen_cluster'] = df['id_estacion_origen'].map(self.estaciones_clusters).fillna(0)
        
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
    
    def create_lag_features_prev_n(self, df):
        """Crear features históricas específicas sin data leakage"""
        print("Creando features históricas específicas (_prev_n)...")
        print("Features objetivo: id_estacion_destino_prev_*, cantidad_estaciones_cercanas_destino_prev_*,")
        print("                  año/mes/dia/hora/minuto/segundo_destino_prev_*, N_ARRIBOS_prev_*, N_SALIDAS_prev_*")
        
        # Convertir ventanas a datetime para manipulación temporal
        df['ventana_arribo_dt'] = pd.to_datetime(df['ventana_arribo'])
        df['ventana_despacho_dt'] = pd.to_datetime(df['ventana_despacho'])
        
        # Preparar features de fecha/hora de destino (basadas en ventana de arribo)
        df['fecha_destino_dt'] = df['ventana_arribo_dt']
        df['año_destino'] = df['fecha_destino_dt'].dt.year
        df['mes_destino'] = df['fecha_destino_dt'].dt.month
        df['dia_destino'] = df['fecha_destino_dt'].dt.day
        df['hora_destino'] = df['fecha_destino_dt'].dt.hour
        df['minuto_destino'] = df['fecha_destino_dt'].dt.minute
        df['segundo_destino'] = df['fecha_destino_dt'].dt.second
        
        # Calcular zona cluster de destino (mapear estaciones destino a clusters)
        estaciones_destino = df[['id_estacion_destino', 'lat_estacion_destino', 'long_estacion_destino']].drop_duplicates()
        estaciones_destino.columns = ['id_estacion', 'lat', 'long']
        
        # Crear clusters para destino
        lat_bins = pd.qcut(estaciones_destino['lat'], q=4, labels=['S', 'SC', 'NC', 'N'], duplicates='drop')
        long_bins = pd.qcut(estaciones_destino['long'], q=4, labels=['W', 'WC', 'EC', 'E'], duplicates='drop')
        estaciones_destino['zona_cluster'] = (lat_bins.astype(str) + long_bins.astype(str)).factorize()[0]
        
        estaciones_clusters_destino = dict(zip(estaciones_destino['id_estacion'], estaciones_destino['zona_cluster']))
        df['zona_destino_cluster'] = df['id_estacion_destino'].map(estaciones_clusters_destino).fillna(0)
        
        # Calcular cantidad de estaciones cercanas para destino
        cluster_counts_destino = df.groupby('zona_destino_cluster')['id_estacion_destino'].nunique().to_dict()
        df['cantidad_estaciones_cercanas_destino'] = df['zona_destino_cluster'].map(cluster_counts_destino).fillna(1)
        
        # 1. ARRIBOS por ventana y estación
        print("   Calculando arribos históricos...")
        arrivals_by_window = df.groupby(['ventana_arribo', 'id_estacion_destino']).size().reset_index(name='N_ARRIBOS')
        arrivals_by_window['ventana_arribo_dt'] = pd.to_datetime(arrivals_by_window['ventana_arribo'])
        
        # 2. SALIDAS por ventana y estación
        print("   Calculando salidas históricas...")
        departures_by_window = df.groupby(['ventana_despacho', 'id_estacion_origen']).size().reset_index(name='N_SALIDAS')
        departures_by_window['ventana_despacho_dt'] = pd.to_datetime(departures_by_window['ventana_despacho'])
        
        # 3. Preparar datos históricos de destino (CRÍTICO: solo usar viajes ya completados)
        print("   Preparando datos históricos de destino (sin data leakage)...")
        destino_historico = df[['ventana_arribo', 'id_estacion_destino', 'año_destino', 'mes_destino', 
                               'dia_destino', 'hora_destino', 'minuto_destino', 'segundo_destino',
                               'cantidad_estaciones_cercanas_destino', 'ventana_arribo_dt']].copy()
        
        all_lag_features = []
        
        # Crear features LAG específicas para períodos 1-6 (arribos/salidas) y 1-3 (destino histórico)
        for lag in range(1, 7):  # prev_1 hasta prev_6
            print(f"   Creando features _prev_{lag} (T-{30*lag} min)...")
            
            # === N_ARRIBOS LAG ===
            # CORRECCIÓN: Para prev_X necesito datos de T-30*X minutos (ANTES, no después)
            arrivals_by_window[f'ventana_lag_{lag}'] = arrivals_by_window['ventana_arribo_dt'] - pd.Timedelta(minutes=30*lag)
            arrivals_lag = arrivals_by_window[['ventana_arribo', 'id_estacion_destino', f'ventana_lag_{lag}', 'N_ARRIBOS']].copy()
            arrivals_lag[f'ventana_lag_{lag}'] = arrivals_lag[f'ventana_lag_{lag}'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Hacer merge: buscar en el dataset principal las ventanas que coincidan con las lag calculadas
            df = df.merge(
                arrivals_lag[[f'ventana_lag_{lag}', 'id_estacion_destino', 'N_ARRIBOS']].rename(columns={'N_ARRIBOS': f'N_ARRIBOS_prev_{lag}'}),
                left_on=['ventana_arribo', 'id_estacion_destino'],
                right_on=[f'ventana_lag_{lag}', 'id_estacion_destino'],
                how='left'
            )
            df = df.drop(f'ventana_lag_{lag}', axis=1, errors='ignore')
            df[f'N_ARRIBOS_prev_{lag}'] = df[f'N_ARRIBOS_prev_{lag}'].fillna(0)
            # También crear con el nombre esperado por compatibilidad
            df[f'arribos_prev_{lag}'] = df[f'N_ARRIBOS_prev_{lag}']
            all_lag_features.extend([f'N_ARRIBOS_prev_{lag}', f'arribos_prev_{lag}'])
            
            # === N_SALIDAS LAG ===
            # CORRECCIÓN: Para prev_X necesito datos de T-30*X minutos (ANTES, no después)
            departures_by_window[f'ventana_lag_{lag}'] = departures_by_window['ventana_despacho_dt'] - pd.Timedelta(minutes=30*lag)
            departures_lag = departures_by_window[['ventana_despacho', 'id_estacion_origen', f'ventana_lag_{lag}', 'N_SALIDAS']].copy()
            departures_lag[f'ventana_lag_{lag}'] = departures_lag[f'ventana_lag_{lag}'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Hacer merge: buscar en el dataset principal las ventanas que coincidan con las lag calculadas
            df = df.merge(
                departures_lag[[f'ventana_lag_{lag}', 'id_estacion_origen', 'N_SALIDAS']].rename(columns={'N_SALIDAS': f'N_SALIDAS_prev_{lag}'}),
                left_on=['ventana_despacho', 'id_estacion_origen'],
                right_on=[f'ventana_lag_{lag}', 'id_estacion_origen'],
                how='left'
            )
            df = df.drop(f'ventana_lag_{lag}', axis=1, errors='ignore')
            df[f'N_SALIDAS_prev_{lag}'] = df[f'N_SALIDAS_prev_{lag}'].fillna(0)
            # También crear con el nombre esperado por compatibilidad
            df[f'salidas_prev_{lag}'] = df[f'N_SALIDAS_prev_{lag}']
            all_lag_features.extend([f'N_SALIDAS_prev_{lag}', f'salidas_prev_{lag}'])
            
            # === FEATURES DE DESTINO HISTÓRICO CON VENTANAS TEMPORALES (solo para lag 1-3) ===
            if lag <= 3:
                print(f"     Agregando features de destino histórico _prev_{lag}...")
                
                # Crear agregación temporal de destinos por ventana
                print(f"       Calculando destinos históricos para T-{30*lag} minutos...")
                
                # Agregar columna de ventana lag al dataset principal para el merge
                df[f'ventana_lag_{lag}'] = df['ventana_arribo_dt'] - pd.Timedelta(minutes=30*lag)
                df[f'ventana_lag_{lag}_str'] = df[f'ventana_lag_{lag}'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Crear agregación de destinos históricos por ventana y estación
                destino_stats = df.groupby(['ventana_arribo', 'id_estacion_destino']).agg({
                    'año_destino': 'first',
                    'mes_destino': 'first',
                    'dia_destino': 'first', 
                    'hora_destino': 'mean',
                    'minuto_destino': 'mean',
                    'segundo_destino': 'mean',
                    'cantidad_estaciones_cercanas_destino': 'first'
                }).reset_index()
                
                # Crear una copia de id_estacion_destino para el lag y renombrar otras columnas
                destino_stats[f'id_estacion_destino_prev_{lag}'] = destino_stats['id_estacion_destino']
                
                # Renombrar columnas para el lag (excepto id_estacion_destino que ya lo manejamos arriba)
                destino_rename = {
                    'año_destino': f'año_destino_prev_{lag}',
                    'mes_destino': f'mes_destino_prev_{lag}',
                    'dia_destino': f'dia_destino_prev_{lag}',
                    'hora_destino': f'hora_destino_prev_{lag}',
                    'minuto_destino': f'minuto_destino_prev_{lag}',
                    'segundo_destino': f'segundo_destino_prev_{lag}',
                    'cantidad_estaciones_cercanas_destino': f'cantidad_estaciones_cercanas_destino_prev_{lag}'
                }
                
                destino_stats_renamed = destino_stats.rename(columns=destino_rename)
                
                # Hacer merge con las ventanas lag calculadas
                df = df.merge(
                    destino_stats_renamed[['ventana_arribo', 'id_estacion_destino', f'id_estacion_destino_prev_{lag}'] + list(destino_rename.values())],
                    left_on=[f'ventana_lag_{lag}_str', 'id_estacion_destino'],
                    right_on=['ventana_arribo', 'id_estacion_destino'],
                    how='left',
                    suffixes=('', '_hist')
                )
                
                # Limpiar columnas temporales
                df = df.drop([f'ventana_lag_{lag}', f'ventana_lag_{lag}_str', 'ventana_arribo_hist', 'id_estacion_destino_hist'], axis=1, errors='ignore')
                
                # Rellenar valores faltantes y convertir tipos (incluir id_estacion_destino_prev_X)
                all_destino_cols = [f'id_estacion_destino_prev_{lag}'] + list(destino_rename.values())
                for col in all_destino_cols:
                    if col in df.columns:
                        if 'id_estacion' in col:
                            df[col] = df[col].fillna(0).astype(int)
                        else:
                            df[col] = df[col].fillna(0)
                    else:
                        # Si la columna no existe, crearla con valores 0
                        df[col] = 0
                
                # Agregar a la lista de features
                all_lag_features.extend(all_destino_cols)
        
        print(f"✅ Features LAG específicas creadas: {len(all_lag_features)} features")
        print(f"   N_ARRIBOS_prev_1 a prev_6: {len([f for f in all_lag_features if 'N_ARRIBOS_prev_' in f])} features")
        print(f"   N_SALIDAS_prev_1 a prev_6: {len([f for f in all_lag_features if 'N_SALIDAS_prev_' in f])} features")
        print(f"   Destino histórico prev_1 a prev_3: {len([f for f in all_lag_features if 'destino_prev_' in f])} features")
        
        # Mostrar estadísticas de algunas features LAG
        sample_features = [f for f in all_lag_features if 'prev_1' in f and f in df.columns]
        for feature in sample_features[:5]:  # Solo mostrar las primeras 5
            print(f"   {feature}: media={df[feature].mean():.2f}, max={df[feature].max()}")
        
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
            'zona_origen_cluster', 'cantidad_estaciones_cercanas_origen',
            'popularidad_estacion_origen', 'hora_promedio_estacion_origen',
            
            # Features temporales (basadas en origen/despacho, información histórica válida)
            'dia_semana', 'es_finde', 'estacion_del_anio',
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            
            # Features de actividad histórica (solo salidas, información válida)
            'N_salidas_historicas',
            
            # Features LAG específicas solicitadas (información anterior válida)
            # Estación destino histórica (prev_1 a prev_3)
            'id_estacion_destino_prev_1', 'id_estacion_destino_prev_2', 'id_estacion_destino_prev_3',
            'cantidad_estaciones_cercanas_destino_prev_1', 'cantidad_estaciones_cercanas_destino_prev_2', 'cantidad_estaciones_cercanas_destino_prev_3',
            
            # Features temporales de destino histórico (prev_1 a prev_3)
            'año_destino_prev_1', 'año_destino_prev_2', 'año_destino_prev_3',
            'mes_destino_prev_1', 'mes_destino_prev_2', 'mes_destino_prev_3',
            'dia_destino_prev_1', 'dia_destino_prev_2', 'dia_destino_prev_3',
            'hora_destino_prev_1', 'hora_destino_prev_2', 'hora_destino_prev_3',
            'minuto_destino_prev_1', 'minuto_destino_prev_2', 'minuto_destino_prev_3',
            'segundo_destino_prev_1', 'segundo_destino_prev_2', 'segundo_destino_prev_3',
            
            # Arribos y salidas históricas (prev_1 a prev_6) - ambos nombres por compatibilidad
            'N_ARRIBOS_prev_1', 'N_ARRIBOS_prev_2', 'N_ARRIBOS_prev_3', 'N_ARRIBOS_prev_4', 'N_ARRIBOS_prev_5', 'N_ARRIBOS_prev_6',
            'arribos_prev_1', 'arribos_prev_2', 'arribos_prev_3', 'arribos_prev_4', 'arribos_prev_5', 'arribos_prev_6',
            'N_SALIDAS_prev_1', 'N_SALIDAS_prev_2', 'N_SALIDAS_prev_3', 'N_SALIDAS_prev_4', 'N_SALIDAS_prev_5', 'N_SALIDAS_prev_6',
            'salidas_prev_1', 'salidas_prev_2', 'salidas_prev_3', 'salidas_prev_4', 'salidas_prev_5', 'salidas_prev_6',
    
            
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
            'id_estacion_origen', 'zona_origen_cluster',
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
            'N_salidas_historicas',
            
            # Features LAG específicas solicitadas
            # Estación destino histórica (prev_1 a prev_3)
            'id_estacion_destino_prev_1', 'id_estacion_destino_prev_2', 'id_estacion_destino_prev_3',
            'cantidad_estaciones_cercanas_destino_prev_1', 'cantidad_estaciones_cercanas_destino_prev_2', 'cantidad_estaciones_cercanas_destino_prev_3',
            
            # Features temporales de destino histórico (prev_1 a prev_3)
            'año_destino_prev_1', 'año_destino_prev_2', 'año_destino_prev_3',
            'mes_destino_prev_1', 'mes_destino_prev_2', 'mes_destino_prev_3',
            'dia_destino_prev_1', 'dia_destino_prev_2', 'dia_destino_prev_3',
            'hora_destino_prev_1', 'hora_destino_prev_2', 'hora_destino_prev_3',
            'minuto_destino_prev_1', 'minuto_destino_prev_2', 'minuto_destino_prev_3',
            'segundo_destino_prev_1', 'segundo_destino_prev_2', 'segundo_destino_prev_3',
            
            # Arribos y salidas históricas (prev_1 a prev_6) - nombres compatibles
            'arribos_prev_1', 'arribos_prev_2', 'arribos_prev_3', 'arribos_prev_4', 'arribos_prev_5', 'arribos_prev_6',
            'salidas_prev_1', 'salidas_prev_2', 'salidas_prev_3', 'salidas_prev_4', 'salidas_prev_5', 'salidas_prev_6',
            
            # Features adicionales LAG
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
        
        # Crear features LAG históricas (sin data leakage)
        df = self.create_lag_features_prev_n(df)
        
        # IMPORTANTE: Calcular target DESPUÉS de las features para evitar leakage
        df = self.calculate_target_arrivals(df)
        
        df = self.select_final_features(df)
        
        # Guardar resultado
        print(f"Guardando dataset transformado en: {output_path}")
        df.to_csv(output_path, index=False)
        
        print("=== Proceso completado ===")
        print(f"Dataset final: {df.shape}")
        print(f"Columnas: {len(df.columns)}")
        print("NOTA: Se incluyeron features LAG históricas (arribos_prev_1 a prev_6) sin data leakage")
        
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
    
    print(f"\nDataset listo para entrenamiento!")
    print(f"Archivo guardado: {output_path}")
    print("\n=== FEATURES LAG ESPECÍFICAS INCLUIDAS ===")
    print("🏢 DESTINO HISTÓRICO: id_estacion_destino_prev_1 a prev_3")
    print("📍 ESTACIONES CERCANAS: cantidad_estaciones_cercanas_destino_prev_1 a prev_3")
    print("📅 TIEMPO DESTINO: año/mes/dia/hora/minuto/segundo_destino_prev_1 a prev_3")
    print("🎯 ARRIBOS: N_ARRIBOS_prev_1 a prev_6 (arribos en períodos [T-30 a T-180])")
    print("🚀 SALIDAS: N_SALIDAS_prev_1 a prev_6 (salidas en períodos [T-30 a T-180])")
    print("✅ Solo usa información de períodos anteriores al objetivo (sin data leakage)")
    lag_features_count = len([f for f in training_features if 'prev_' in f])
    print(f"📏 Total de features LAG específicas: {lag_features_count} features")
    print(f"   - Destino histórico: {len([f for f in training_features if 'destino_prev_' in f])} features")
    print(f"   - Arribos/Salidas: {len([f for f in training_features if 'ARRIBOS_prev_' in f or 'SALIDAS_prev_' in f])} features")
    
    return df_transformed, training_features

if __name__ == "__main__":
    df_transformed, training_features = main() 