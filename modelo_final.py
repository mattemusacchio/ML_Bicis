#!/usr/bin/env python3
"""
PROYECTO FINAL - PREDICCIÓN DE BICICLETAS PÚBLICAS GCBA
Modelo M0 - Predictor de Arribos

Autores: Matteo Musacchio, Tiziano Demarco

Este script implementa el modelo final M0 para predecir la cantidad de arribos 
de bicicletas en las estaciones del sistema de bicicletas públicas del GCBA.

Uso:
    python modelo_m0_predictor.py --input datos_test.csv --output predicciones.csv
    
Formato de entrada (dos opciones):
    1. Datos preprocesados: CSV con 13 features del modelo
    2. Datos raw: CSV con viajes que requieren preprocesamiento completo
"""

import argparse
import pickle
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_time_series_dataset_fast(trips_df, time_window_minutes=30):
    """
    Versión optimizada de transformación de dataset de viajes a series temporales.
    """
    trips_df = trips_df.copy()

    trips_df['fecha_origen_recorrido'] = pd.to_datetime(trips_df['fecha_origen_recorrido'])
    trips_df['fecha_destino_recorrido'] = pd.to_datetime(trips_df['fecha_destino_recorrido'])
    
    # 1. Crear columnas de ventana para despachos (origen) y arribos (destino)
    trips_df['timestamp_origen_window'] = (trips_df['fecha_origen_recorrido'].dt.floor(f'{time_window_minutes}min') + pd.Timedelta(minutes=time_window_minutes))
    trips_df['timestamp_destino_window'] = trips_df['fecha_destino_recorrido'].dt.floor(f'{time_window_minutes}min')
    trips_df['timestamp_destino_prev'] = (
    trips_df['fecha_destino_recorrido'].dt.floor(f'{time_window_minutes}min') + pd.Timedelta(minutes=time_window_minutes))
    arribos_prev = trips_df.groupby(['timestamp_destino_prev','id_estacion_destino']).size().reset_index(name='arribos_prev_count').rename(columns={'timestamp_destino_prev':'timestamp','id_estacion_destino':'id_estacion'})

    # 2. Obtener rango de timestamps
    fecha_min = trips_df['timestamp_origen_window'].min()
    fecha_max = trips_df['timestamp_destino_window'].max()
    timestamps = pd.date_range(start=fecha_min, end=fecha_max, freq=f'{time_window_minutes}min')
    
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
    agg_dict = {'id_estacion_origen': 'count'}  # Siempre contar despachos
    
    # Agregar columnas opcionales si existen
    if 'duracion_recorrido' in trips_df.columns:
        agg_dict['duracion_recorrido'] = ['mean', 'std', 'count']
    if 'edad_usuario' in trips_df.columns:
        agg_dict['edad_usuario'] = ['mean', 'std']
    if 'genero' in trips_df.columns:
        agg_dict['genero'] = lambda x: (x == 'FEMALE').sum() / len(x) if len(x) > 0 else 0
    if 'modelo_bicicleta' in trips_df.columns:
        agg_dict['modelo_bicicleta'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 'UNKNOWN'
    
    despachos = trips_df.groupby(['timestamp_origen_window', 'id_estacion_origen']).agg(agg_dict).reset_index()
    
    # Aplanar columnas si hay múltiples niveles
    if isinstance(despachos.columns, pd.MultiIndex):
        despachos.columns = ['_'.join(col).strip('_') for col in despachos.columns.values]
    
    # Renombrar columnas
    rename_dict = {
        'timestamp_origen_window': 'timestamp',
        'id_estacion_origen': 'id_estacion',
        'id_estacion_origen_count': 'despachos_count',
        'genero': 'proporcion_mujeres',
        'modelo_bicicleta': 'modelo_mas_comun'
    }
    despachos = despachos.rename(columns=rename_dict)

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
    
    # Rellenar columnas opcionales si existen
    if 'duracion_recorrido_mean' in ts_df.columns:
        ts_df['duracion_recorrido_mean'] = ts_df['duracion_recorrido_mean'].fillna(0)
        ts_df['duracion_recorrido_std'] = ts_df['duracion_recorrido_std'].fillna(0)
        ts_df['duracion_recorrido_count'] = ts_df['duracion_recorrido_count'].fillna(0).astype(int)
    if 'edad_usuario_mean' in ts_df.columns:
        ts_df['edad_usuario_mean'] = ts_df['edad_usuario_mean'].fillna(0)
        ts_df['edad_usuario_std'] = ts_df['edad_usuario_std'].fillna(0)
    if 'proporcion_mujeres' in ts_df.columns:
        ts_df['proporcion_mujeres'] = ts_df['proporcion_mujeres'].fillna(0)
    if 'modelo_mas_comun' in ts_df.columns:
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
    features_to_shift = ['despachos_count', 'arribos_count']
    
    # Agregar features opcionales si existen
    if 'duracion_recorrido_mean' in ts_df.columns:
        features_to_shift.extend(['duracion_recorrido_mean', 'duracion_recorrido_std', 'duracion_recorrido_count'])
    if 'edad_usuario_mean' in ts_df.columns:
        features_to_shift.extend(['edad_usuario_mean', 'edad_usuario_std'])
    if 'proporcion_mujeres' in ts_df.columns:
        features_to_shift.append('proporcion_mujeres')

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


class ModeloM0Predictor:
    """
    Clase principal para cargar y usar el modelo M0 entrenado.
    
    El modelo M0 es un XGBoost Regressor entrenado para predecir arribos
    de bicicletas usando features temporales, geográficas y de demanda.
    """
    
    def __init__(self, modelo_path="models/xgb_model_M0.pkl"):
        """
        Inicializa el predictor cargando el modelo entrenado.
        
        Args:
            modelo_path (str): Ruta al archivo pickle del modelo M0
        """
        self.modelo_path = modelo_path
        self.model_info = None
        self.model = None
        self.features_requeridas = None
        self.target_real = None  # Para guardar valores reales cuando estén disponibles
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """
        Carga el modelo M0 entrenado desde el archivo pickle.
        
        Raises:
            FileNotFoundError: Si no se encuentra el archivo del modelo
            Exception: Si hay error al cargar el modelo
        """
        try:
            print(f"📦 Cargando modelo M0 desde: {self.modelo_path}")
            
            with open(self.modelo_path, 'rb') as f:
                self.model_info = pickle.load(f)
            
            self.model = self.model_info['model']
            self.features_requeridas = self.model_info['features']
            
            print(f"✅ Modelo M0 cargado exitosamente")
            print(f"   📊 Métricas de entrenamiento:")
            for metric, value in self.model_info['metrics'].items():
                print(f"   {metric}: {value:.4f}")
            print(f"   📅 Fecha de entrenamiento: {self.model_info.get('train_date', 'No disponible')}")
            print(f"   🔧 Features requeridas: {len(self.features_requeridas)}")
            
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el modelo en {self.modelo_path}")
            print("   Verifica que el modelo M0 esté entrenado y guardado.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {str(e)}")
            sys.exit(1)
    
    def detectar_tipo_datos(self, df):
        """
        Detecta si los datos son raw (viajes) o preprocesados (features).
        
        Args:
            df (pd.DataFrame): DataFrame con los datos de entrada
            
        Returns:
            str: 'raw' o 'processed'
        """
        # Verificar si tiene todas las features del modelo
        if all(feature in df.columns for feature in self.features_requeridas):
            return 'processed'
        
        # Verificar si tiene columnas típicas de viajes raw
        raw_columns = ['fecha_origen_recorrido', 'fecha_destino_recorrido', 'id_estacion_origen', 'id_estacion_destino']
        if any(col in df.columns for col in raw_columns):
            return 'raw'
        
        return 'unknown'
    
    def enriquecer_con_usuarios(self, trips_df, usuarios_path="data/raw/usuarios_ecobici_2024.csv"):
        """
        Enriquece los datos de viajes con información de usuarios.
        
        Args:
            trips_df (pd.DataFrame): DataFrame con viajes
            usuarios_path (str): Ruta al archivo de usuarios
            
        Returns:
            pd.DataFrame: DataFrame enriquecido
        """
        try:
            if os.path.exists(usuarios_path):
                print(f"📊 Enriqueciendo con datos de usuarios desde {usuarios_path}")
                usuarios = pd.read_csv(usuarios_path)
                usuarios['id_usuario'] = usuarios['id_usuario'].astype(float)
                trips_df['id_usuario'] = trips_df['id_usuario'].astype(float)
                
                trips_enriched = trips_df.merge(
                    usuarios[['id_usuario', 'edad_usuario', 'fecha_alta', 'hora_alta']],
                    on='id_usuario',
                    how='left'
                )
                print(f"   ✅ Datos enriquecidos: {trips_enriched.shape}")
                return trips_enriched
            else:
                print(f"⚠️ No se encontró archivo de usuarios en {usuarios_path}")
                print("   Continuando sin enriquecimiento...")
                return trips_df
        except Exception as e:
            print(f"⚠️ Error al enriquecer con usuarios: {str(e)}")
            print("   Continuando sin enriquecimiento...")
            return trips_df
    
    def preprocesar_datos_raw(self, df):
        """
        Aplica el pipeline completo de preprocesamiento a datos raw.
        
        Args:
            df (pd.DataFrame): DataFrame con datos raw de viajes
            
        Returns:
            pd.DataFrame: DataFrame preprocesado listo para predicción
        """
        print(f"🔄 Aplicando pipeline de preprocesamiento completo...")
        
        # 1. Enriquecer con usuarios (si está disponible)
        df_enriched = self.enriquecer_con_usuarios(df)
        
        # 2. Aplicar pipeline de series temporales
        print(f"📊 Transformando a series temporales...")
        df_ts = create_time_series_dataset_fast(df_enriched, time_window_minutes=30)
        
        # 3. Seleccionar features finales (las que usa el modelo M0)
        features_finales = [
            'timestamp', 'id_estacion', 'lat_estacion', 'long_estacion', 'despachos_count',
            'edad_usuario_mean', 'edad_usuario_std', 'proporcion_mujeres',
            'hora', 'dia_semana', 'es_fin_semana', 'mes', 'dia_mes', 'año'
        ]
        
        # ⭐ GUARDAR ARRIBOS_COUNT SI ESTÁ DISPONIBLE PARA EVALUACIÓN
        if 'arribos_count' in df_ts.columns:
            features_finales.append('arribos_count')
            print(f"   ✅ Guardando valores reales de arribos para evaluación posterior")
        
        # Verificar qué features están disponibles
        features_disponibles = [f for f in features_finales if f in df_ts.columns]
        features_faltantes = [f for f in features_finales if f not in df_ts.columns]
        
        if features_faltantes:
            print(f"⚠️ Features faltantes después del preprocesamiento: {features_faltantes}")
        
        df_final = df_ts[features_disponibles].copy()
        
        print(f"✅ Preprocesamiento completado: {df_final.shape}")
        return df_final
    
    def validar_datos_entrada(self, df):
        """
        Valida que los datos de entrada tengan el formato correcto.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos de entrada
            
        Returns:
            tuple: (es_válido, mensaje_error)
        """
        tipo_datos = self.detectar_tipo_datos(df)
        
        if tipo_datos == 'processed':
            # Verificar columnas requeridas para datos preprocesados
            columnas_faltantes = set(self.features_requeridas) - set(df.columns)
            if columnas_faltantes:
                return False, f"Columnas faltantes: {list(columnas_faltantes)}"
            
            # Verificar que no haya valores nulos en features críticas
            features_criticas = ['id_estacion', 'timestamp']
            for feature in features_criticas:
                if feature in df.columns and df[feature].isnull().any():
                    return False, f"La columna '{feature}' contiene valores nulos"
            
            # Verificar tipos de datos básicos
            if not pd.api.types.is_numeric_dtype(df['id_estacion']):
                return False, "id_estacion debe ser numérico"
            
            return True, "Datos preprocesados válidos"
            
        elif tipo_datos == 'raw':
            # Verificar columnas básicas para datos raw
            columnas_raw_basicas = ['fecha_origen_recorrido', 'fecha_destino_recorrido']
            columnas_faltantes = set(columnas_raw_basicas) - set(df.columns)
            if columnas_faltantes:
                return False, f"Columnas raw faltantes: {list(columnas_faltantes)}"
            
            return True, "Datos raw válidos para preprocesamiento"
        
        else:
            return False, "No se pudo determinar el tipo de datos (raw o processed)"
    
    def preparar_features(self, df):
        """
        Prepara las features para la predicción.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos de entrada
            
        Returns:
            tuple: (X, df_procesado) donde X son las features y df_procesado incluye target si existe
        """
        tipo_datos = self.detectar_tipo_datos(df)
        
        if tipo_datos == 'raw':
            print(f"📊 Detectados datos raw - aplicando preprocesamiento completo...")
            df_procesado = self.preprocesar_datos_raw(df)
        else:
            print(f"📊 Detectados datos preprocesados - aplicando preparación directa...")
            df_procesado = df.copy()
        
        # ⭐ GUARDAR TARGET REAL SI ESTÁ DISPONIBLE
        if 'arribos_count' in df_procesado.columns:
            self.target_real = df_procesado['arribos_count'].copy()
            print(f"   💾 Guardados {len(self.target_real)} valores reales para evaluación")
        else:
            self.target_real = None
            print(f"   ℹ️ No hay valores reales disponibles (solo predicción)")
        
        # Asegurar que timestamp esté en formato datetime
        if 'timestamp' in df_procesado.columns:
            df_procesado['timestamp'] = pd.to_datetime(df_procesado['timestamp'])
        
        # Seleccionar solo las features que el modelo necesita
        features_disponibles = [f for f in self.features_requeridas if f in df_procesado.columns]
        features_faltantes = [f for f in self.features_requeridas if f not in df_procesado.columns]
        
        if features_faltantes:
            print(f"⚠️ Features faltantes para el modelo: {features_faltantes}")
            # Crear features faltantes con valores por defecto
            for feature in features_faltantes:
                df_procesado[feature] = 0
                print(f"   Creando {feature} con valor por defecto: 0")
        
        X = df_procesado[self.features_requeridas]
        
        # Rellenar valores nulos con 0 (estrategia conservadora)
        X = X.fillna(0)
        
        return X, df_procesado
    
    def predecir(self, df):
        """
        Realiza predicciones usando el modelo M0.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos de entrada
            
        Returns:
            tuple: (predicciones, df_procesado)
        """
        print(f"🔮 Realizando predicciones para {len(df)} registros...")
        
        # Validar datos
        es_valido, mensaje = self.validar_datos_entrada(df)
        if not es_valido:
            raise ValueError(f"Datos de entrada inválidos: {mensaje}")
        
        # Preparar features
        X, df_procesado = self.preparar_features(df)
        
        print(f"   📋 Features preparadas: {X.shape}")
        print(f"   🎯 Features utilizadas: {list(X.columns)}")
        
        # Hacer predicciones
        predicciones = self.model.predict(X)
        
        # Asegurar que las predicciones sean no negativas (arribos no pueden ser < 0)
        predicciones = np.maximum(0, predicciones)
        
        print(f"✅ Predicciones completadas")
        print(f"   📈 Rango de predicciones: {predicciones.min():.2f} - {predicciones.max():.2f}")
        print(f"   📊 Promedio de arribos predichos: {predicciones.mean():.2f}")
        
        return predicciones, df_procesado
    
    def generar_submission(self, df_procesado, predicciones, archivo_salida):
        """
        Genera el archivo de submission con las predicciones.
        
        Args:
            df_procesado (pd.DataFrame): DataFrame procesado con datos de entrada
            predicciones (np.array): Array con predicciones
            archivo_salida (str): Ruta del archivo de salida
        """
        print(f"💾 Generando archivo de submission...")
        
        submission = pd.DataFrame({
            'id_estacion': df_procesado['id_estacion'],
            'timestamp': df_procesado['timestamp'],
            'arribos_predichos': predicciones
        })
        
        # Redondear predicciones a enteros (arribos son conteos)
        submission['arribos_predichos'] = submission['arribos_predichos'].round().astype(int)
        
        # Guardar archivo
        submission.to_csv(archivo_salida, index=False)
        
        print(f"✅ Submission guardado en: {archivo_salida}")
        print(f"   📊 Estadísticas del submission:")
        print(f"   Total de registros: {len(submission)}")
        print(f"   Estaciones únicas: {submission['id_estacion'].nunique()}")
        print(f"   Rango temporal: {submission['timestamp'].min()} a {submission['timestamp'].max()}")
        print(f"   Total arribos predichos: {submission['arribos_predichos'].sum()}")
    
    def evaluar_predicciones(self, predicciones):
        """
        Evalúa las predicciones contra los valores reales si están disponibles.
        
        Args:
            predicciones (np.array): Array con predicciones
            
        Returns:
            dict: Métricas de evaluación o None si no hay valores reales
        """
        if self.target_real is not None:
            print(f"📊 Evaluando predicciones contra valores reales...")
            
            y_true = self.target_real.values
            y_pred = predicciones
            
            # Asegurar que ambos arrays tengan la misma longitud
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            
            # Calcular métricas
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"   📈 Métricas de evaluación:")
            print(f"   MSE: {mse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   R²: {r2:.4f}")
            
            # Estadísticas adicionales
            errores = np.abs(y_true - y_pred)
            print(f"   📊 Estadísticas de error:")
            print(f"   Error promedio: {errores.mean():.2f}")
            print(f"   Error mediano: {np.median(errores):.2f}")
            print(f"   Error máximo: {errores.max():.2f}")
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'error_promedio': errores.mean(),
                'error_mediano': np.median(errores),
                'error_maximo': errores.max()
            }
        else:
            print(f"ℹ️ No hay valores reales disponibles para evaluación")
            return None


def main():
    """
    Función principal del script.
    """
    parser = argparse.ArgumentParser(
        description="Modelo M0 - Predictor de arribos de bicicletas GCBA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
    # Predicción con datos preprocesados
    python modelo_m0_predictor.py --input test_processed.csv --output predicciones.csv
    
    # Predicción con datos raw (viajes)
    python modelo_m0_predictor.py --input test_raw.csv --output predicciones.csv --usuarios data/raw/usuarios_ecobici_2024.csv
    
    # Evaluación con datos que incluyen valores reales
    python modelo_m0_predictor.py --input datos_con_target.csv --output predicciones.csv --evaluar
        """
    )
    
    parser.add_argument('--input', '-i', 
                        required=True,
                        help='Archivo CSV con datos de entrada (raw o preprocesados)')
    
    parser.add_argument('--output', '-o', 
                        required=True,
                        help='Archivo CSV de salida con predicciones')
    
    parser.add_argument('--modelo', '-m',
                        default='models/xgb_model_M0.pkl',
                        help='Ruta al modelo M0 (default: models/xgb_model_M0.pkl)')
    
    parser.add_argument('--usuarios', '-u',
                        default='data/raw/usuarios_ecobici_2024.csv',
                        help='Ruta al archivo de usuarios para enriquecimiento')
    
    parser.add_argument('--evaluar', '-e',
                        action='store_true',
                        help='Evaluar predicciones si hay valores reales disponibles')
    
    args = parser.parse_args()
    
    print("🚴‍♀️ MODELO M0 - PREDICTOR DE ARRIBOS DE BICICLETAS GCBA")
    print("=" * 60)
    print(f"📅 Ejecutado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Archivo de entrada: {args.input}")
    print(f"📁 Archivo de salida: {args.output}")
    print(f"🤖 Modelo: {args.modelo}")
    print(f"👥 Usuarios: {args.usuarios}")
    print()
    
    try:
        # Verificar que existe el archivo de entrada
        if not os.path.exists(args.input):
            print(f"❌ Error: No se encontró el archivo de entrada: {args.input}")
            sys.exit(1)
        
        # Cargar datos de entrada
        print(f"📖 Cargando datos de entrada...")
        df = pd.read_csv(args.input, parse_dates=['fecha_origen_recorrido', 'fecha_destino_recorrido'], low_memory=False)
        print(f"✅ Datos cargados: {df.shape}")
        print(f"   Columnas: {list(df.columns)}")
        print()
        
        # Inicializar predictor
        predictor = ModeloM0Predictor(args.modelo)
        print()
        
        # Realizar predicciones
        predicciones, df_procesado = predictor.predecir(df)
        print()
        
        # Evaluar si se solicita
        if args.evaluar:
            metricas = predictor.evaluar_predicciones(predicciones)
            print()
        
        # Generar archivo de submission
        predictor.generar_submission(df_procesado, predicciones, args.output)
        print()
        
        print("🎉 ¡Proceso completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 