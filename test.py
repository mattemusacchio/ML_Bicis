# =============================================================================
# PROYECTO FINAL - MACHINE LEARNING 
# PREDICCIÓN DE ARRIBOS DE BICICLETAS PÚBLICAS GCBA
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Para el modelado
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# =============================================================================
# PASO 1: CARGA Y EXPLORACIÓN INICIAL DE DATOS
# =============================================================================

print("🚴 CARGANDO DATOS DE BICICLETAS PÚBLICAS BA...")

# Cargar datos de trips 2024
trips_df = pd.read_csv('data/raw/trips_2024.csv')
usuarios_df = pd.read_csv('data/raw/usuarios_ecobici_2024.csv')

print(f"📊 Datos cargados:")
print(f"   - Trips: {trips_df.shape[0]:,} registros, {trips_df.shape[1]} columnas")
print(f"   - Usuarios: {usuarios_df.shape[0]:,} registros, {usuarios_df.shape[1]} columnas")

# =============================================================================
# EXPLORACIÓN DE ESTRUCTURA DE DATOS
# =============================================================================

print("\n🔍 EXPLORANDO ESTRUCTURA DE TRIPS...")
print(f"Columnas de trips: {list(trips_df.columns)}")
print(f"\nPrimeras filas:")
print(trips_df.head())

print(f"\nInfo de trips:")
print(trips_df.info())

print("\n🔍 EXPLORANDO ESTRUCTURA DE USUARIOS...")
print(f"Columnas de usuarios: {list(usuarios_df.columns)}")
print(f"\nPrimeras filas:")
print(usuarios_df.head())

# =============================================================================
# PASO 2: PREPROCESAMIENTO DE DATOS
# =============================================================================

print("\n🛠️ PREPROCESANDO DATOS...")

# Convertir fechas a datetime
trips_df['fecha_origen_recorrido'] = pd.to_datetime(trips_df['fecha_origen_recorrido'])
trips_df['fecha_destino_recorrido'] = pd.to_datetime(trips_df['fecha_destino_recorrido'])

# Filtrar solo datos hasta agosto 2024 como especifica el enunciado
trips_df = trips_df[trips_df['fecha_origen_recorrido'] <= '2024-08-31']

print(f"📅 Datos filtrados hasta agosto 2024: {trips_df.shape[0]:,} registros")

# Verificar rango de fechas
print(f"   - Fecha mínima: {trips_df['fecha_origen_recorrido'].min()}")
print(f"   - Fecha máxima: {trips_df['fecha_origen_recorrido'].max()}")

# =============================================================================
# ANÁLISIS EXPLORATORIO
# =============================================================================

print("\n📈 ANÁLISIS EXPLORATORIO...")

# Estadísticas básicas de duración
print(f"Duración promedio de viajes: {trips_df['duracion_recorrido'].mean():.2f} segundos")
print(f"Duración mediana: {trips_df['duracion_recorrido'].median():.2f} segundos")

# Cantidad de estaciones únicas
n_estaciones_origen = trips_df['id_estacion_origen'].nunique()
n_estaciones_destino = trips_df['id_estacion_destino'].nunique()
print(f"Estaciones origen únicas: {n_estaciones_origen}")
print(f"Estaciones destino únicas: {n_estaciones_destino}")

# =============================================================================
# PASO 3: FEATURE ENGINEERING - CREACIÓN DE VENTANAS TEMPORALES
# =============================================================================

print("\n⚙️ CREANDO FEATURES TEMPORALES...")

# Definir delta T (usaremos 30 minutos como ejemplo)
DELTA_T_MINUTES = 30

# Agregar features temporales
trips_df['año'] = trips_df['fecha_origen_recorrido'].dt.year
trips_df['mes'] = trips_df['fecha_origen_recorrido'].dt.month
trips_df['dia'] = trips_df['fecha_origen_recorrido'].dt.day
trips_df['hora'] = trips_df['fecha_origen_recorrido'].dt.hour
trips_df['dia_semana'] = trips_df['fecha_origen_recorrido'].dt.dayofweek
trips_df['es_fin_de_semana'] = trips_df['dia_semana'].isin([5, 6]).astype(int)

# Crear ventanas temporales de 30 minutos
trips_df['timestamp_rounded'] = trips_df['fecha_origen_recorrido'].dt.floor(f'{DELTA_T_MINUTES}min')

print(f"✅ Features temporales creadas con ventanas de {DELTA_T_MINUTES} minutos")

# =============================================================================
# PASO 4: CREACIÓN DEL DATASET PARA ML - AGREGACIÓN POR VENTANAS
# =============================================================================

print("\n🎯 CREANDO DATASET PARA MACHINE LEARNING...")

# Obtener lista de todas las estaciones
todas_las_estaciones = pd.concat([
    trips_df[['id_estacion_origen', 'nombre_estacion_origen', 'lat_estacion_origen', 'long_estacion_origen']].rename(columns={
        'id_estacion_origen': 'id_estacion',
        'nombre_estacion_origen': 'nombre_estacion', 
        'lat_estacion_origen': 'lat_estacion',
        'long_estacion_origen': 'long_estacion'
    }),
    trips_df[['id_estacion_destino', 'nombre_estacion_destino', 'lat_estacion_destino', 'long_estacion_destino']].rename(columns={
        'id_estacion_destino': 'id_estacion',
        'nombre_estacion_destino': 'nombre_estacion',
        'lat_estacion_destino': 'lat_estacion', 
        'long_estacion_destino': 'long_estacion'
    })
]).drop_duplicates(subset=['id_estacion'])

print(f"📍 Total de estaciones identificadas: {len(todas_las_estaciones)}")

# =============================================================================
# AGREGACIÓN DE PARTIDAS POR VENTANA TEMPORAL
# =============================================================================

print("\n📤 AGREGANDO PARTIDAS POR VENTANA TEMPORAL...")

# Contar partidas por estación y ventana temporal
partidas_por_ventana = (trips_df.groupby(['timestamp_rounded', 'id_estacion_origen'])
                       .agg({
                           'id_recorrido': 'count',
                           'hora': 'first',
                           'dia_semana': 'first', 
                           'es_fin_de_semana': 'first',
                           'mes': 'first',
                           'dia': 'first'
                       })
                       .rename(columns={'id_recorrido': 'partidas'})
                       .reset_index())

print(f"✅ Partidas agregadas: {len(partidas_por_ventana)} registros")

# =============================================================================
# AGREGACIÓN DE ARRIBOS POR VENTANA TEMPORAL  
# =============================================================================

print("\n📥 AGREGANDO ARRIBOS POR VENTANA TEMPORAL...")

# Para los arribos, necesitamos usar la fecha de destino
trips_df['timestamp_destino_rounded'] = trips_df['fecha_destino_recorrido'].dt.floor(f'{DELTA_T_MINUTES}min')

# Contar arribos por estación y ventana temporal
arribos_por_ventana = (trips_df.groupby(['timestamp_destino_rounded', 'id_estacion_destino'])
                      .agg({
                          'id_recorrido': 'count'
                      })
                      .rename(columns={'id_recorrido': 'arribos'})
                      .reset_index()
                      .rename(columns={'timestamp_destino_rounded': 'timestamp_rounded',
                                     'id_estacion_destino': 'id_estacion'}))

print(f"✅ Arribos agregados: {len(arribos_por_ventana)} registros")

# =============================================================================
# PASO 5: CREACIÓN DE FEATURES Y TARGETS 
# =============================================================================

print("\n🎯 CREANDO FEATURES (X) Y TARGETS (Y)...")

# Obtener todas las combinaciones de timestamp y estación posibles
timestamps_unicos = pd.date_range(
    start=trips_df['timestamp_rounded'].min(),
    end=trips_df['timestamp_rounded'].max(), 
    freq=f'{DELTA_T_MINUTES}min'
)

# Crear un DataFrame base con todas las combinaciones
base_df = pd.DataFrame([
    (ts, estacion) 
    for ts in timestamps_unicos 
    for estacion in todas_las_estaciones['id_estacion'].unique()
], columns=['timestamp', 'id_estacion'])

print(f"📊 Dataset base creado: {len(base_df)} registros")
print(f"   - Ventanas temporales: {len(timestamps_unicos)}")
print(f"   - Estaciones: {todas_las_estaciones['id_estacion'].nunique()}")

# =============================================================================
# MERGE DE PARTIDAS Y ARRIBOS
# =============================================================================

print("\n🔗 COMBINANDO PARTIDAS Y ARRIBOS...")

# Merge con partidas
dataset = base_df.merge(
    partidas_por_ventana.rename(columns={'timestamp_rounded': 'timestamp', 'id_estacion_origen': 'id_estacion'}),
    on=['timestamp', 'id_estacion'], 
    how='left'
)

# Merge con arribos (estos serán nuestros targets futuros)
dataset = dataset.merge(
    arribos_por_ventana.rename(columns={'timestamp_rounded': 'timestamp'}),
    on=['timestamp', 'id_estacion'],
    how='left'
)

# Rellenar NaN con 0 (no hubo partidas/arribos)
dataset['partidas'] = dataset['partidas'].fillna(0)
dataset['arribos'] = dataset['arribos'].fillna(0)

print(f"✅ Dataset combinado: {len(dataset)} registros")

# =============================================================================
# AGREGAR INFORMACIÓN DE ESTACIONES
# =============================================================================

print("\n📍 AGREGANDO INFORMACIÓN GEOGRÁFICA...")

dataset = dataset.merge(
    todas_las_estaciones[['id_estacion', 'lat_estacion', 'long_estacion', 'nombre_estacion']],
    on='id_estacion',
    how='left'
)

# Agregar features temporales al dataset final
dataset['hora'] = dataset['timestamp'].dt.hour
dataset['dia_semana'] = dataset['timestamp'].dt.dayofweek  
dataset['es_fin_de_semana'] = dataset['dia_semana'].isin([5, 6]).astype(int)
dataset['mes'] = dataset['timestamp'].dt.month
dataset['dia'] = dataset['timestamp'].dt.day

print(f"✅ Features geográficas y temporales agregadas")

# =============================================================================
# PASO 6: CREACIÓN DE FEATURES DE PARTIDAS PASADAS PARA PREDICCIÓN
# =============================================================================

print("\n⏰ CREANDO FEATURES DE PARTIDAS PASADAS...")

# Ordenar por timestamp para crear las features de lag
dataset = dataset.sort_values(['id_estacion', 'timestamp'])

# Crear features de lag (partidas en ventanas anteriores)
for lag in [1, 2, 3, 6]:  # 1, 2, 3 y 6 ventanas atrás (30, 60, 90, 180 mins)
    dataset[f'partidas_lag_{lag}'] = dataset.groupby('id_estacion')['partidas'].shift(lag)

# Features agregadas de partidas pasadas
dataset['partidas_rolling_mean_3'] = dataset.groupby('id_estacion')['partidas'].rolling(window=3, min_periods=1).mean().values
dataset['partidas_rolling_sum_6'] = dataset.groupby('id_estacion')['partidas'].rolling(window=6, min_periods=1).sum().values

print(f"✅ Features de lag creadas")

# =============================================================================
# PASO 7: CREACIÓN DE TARGETS FUTUROS
# =============================================================================

print("\n🎯 CREANDO TARGETS FUTUROS...")

# Crear targets futuros (arribos en las próximas ventanas)
dataset['arribos_futuro_1'] = dataset.groupby('id_estacion')['arribos'].shift(-1)  # Próximos 30 min
dataset['arribos_futuro_2'] = dataset.groupby('id_estacion')['arribos'].shift(-2)  # Próximos 60 min

# Target principal: arribos en los próximos 30 minutos
dataset['target'] = dataset['arribos_futuro_1']

print(f"✅ Targets futuros creados")

# =============================================================================
# PASO 8: FEATURE ENGINEERING ADICIONAL
# =============================================================================

print("\n🔧 FEATURE ENGINEERING ADICIONAL...")

# Agregar información de usuarios si está disponible
if not usuarios_df.empty:
    # Obtener estadísticas de usuarios por viaje
    user_stats = trips_df.groupby(['timestamp_rounded', 'id_estacion_origen'])['id_usuario'].agg(['count', 'nunique']).reset_index()
    user_stats.columns = ['timestamp', 'id_estacion', 'total_viajes_usuarios', 'usuarios_unicos']
    
    dataset = dataset.merge(user_stats, on=['timestamp', 'id_estacion'], how='left')
    dataset['total_viajes_usuarios'] = dataset['total_viajes_usuarios'].fillna(0)
    dataset['usuarios_unicos'] = dataset['usuarios_unicos'].fillna(0)
    
    print(f"✅ Features de usuarios agregadas")

# Features de ubicación relativa
if not dataset.empty:
    centro_lat = dataset['lat_estacion'].mean()
    centro_long = dataset['long_estacion'].mean()
    
    dataset['distancia_al_centro'] = np.sqrt(
        (dataset['lat_estacion'] - centro_lat)**2 + 
        (dataset['long_estacion'] - centro_long)**2
    )
    
    print(f"✅ Features de ubicación relativa agregadas")

# =============================================================================
# PASO 9: PREPARACIÓN DE DATOS PARA MODELADO
# =============================================================================

print("\n🎓 PREPARANDO DATOS PARA MODELADO...")

# Seleccionar features para el modelo
feature_columns = [
    'hora', 'dia_semana', 'es_fin_de_semana', 'mes', 'dia',
    'lat_estacion', 'long_estacion', 'distancia_al_centro',
    'partidas', 'partidas_lag_1', 'partidas_lag_2', 'partidas_lag_3', 'partidas_lag_6',
    'partidas_rolling_mean_3', 'partidas_rolling_sum_6'
]

# Agregar features de usuario si existen
if 'total_viajes_usuarios' in dataset.columns:
    feature_columns.extend(['total_viajes_usuarios', 'usuarios_unicos'])

# Filtrar registros válidos (sin NaN en target y con suficientes lags)
dataset_clean = dataset.dropna(subset=['target'] + feature_columns)

print(f"📊 Dataset limpio: {len(dataset_clean)} registros")
print(f"📊 Features seleccionadas: {len(feature_columns)}")

# =============================================================================
# ANÁLISIS DE DISTRIBUCIÓN DE TARGETS
# =============================================================================

print("\n📈 ANÁLISIS DE DISTRIBUCIÓN DE TARGETS...")

print(f"Estadísticas de arribos futuros:")
print(f"   - Media: {dataset_clean['target'].mean():.2f}")
print(f"   - Mediana: {dataset_clean['target'].median():.2f}")
print(f"   - Desviación estándar: {dataset_clean['target'].std():.2f}")
print(f"   - Máximo: {dataset_clean['target'].max()}")

# =============================================================================
# PASO 10: SPLIT TEMPORAL DE DATOS
# =============================================================================

print("\n✂️ DIVIDIENDO DATOS TEMPORALMENTE...")

# Split temporal: últimas 2 semanas para test
cutoff_date = dataset_clean['timestamp'].max() - timedelta(weeks=2)

train_data = dataset_clean[dataset_clean['timestamp'] <= cutoff_date]
test_data = dataset_clean[dataset_clean['timestamp'] > cutoff_date]

print(f"📅 Split temporal:")
print(f"   - Train: {len(train_data)} registros (hasta {cutoff_date.strftime('%Y-%m-%d')})")
print(f"   - Test: {len(test_data)} registros (desde {cutoff_date.strftime('%Y-%m-%d')})")

# Preparar X y y
X_train = train_data[feature_columns]
y_train = train_data['target']
X_test = test_data[feature_columns]
y_test = test_data['target']

print(f"✅ Datos preparados para entrenamiento")

# =============================================================================
# PASO 11: ESCALADO DE FEATURES
# =============================================================================

print("\n📏 ESCALANDO FEATURES...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Features escaladas")

# =============================================================================
# PASO 12: ENTRENAMIENTO DE MODELOS
# =============================================================================

print("\n🤖 ENTRENANDO MODELOS...")

# Diccionario de modelos a probar
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

# Entrenar y evaluar cada modelo
results = {}

for name, model in models.items():
    print(f"\n🔄 Entrenando {name}...")
    
    # Entrenar modelo
    if name in ['Ridge Regression', 'Lasso Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'model': model,
        'predictions': y_pred
    }
    
    print(f"   ✅ {name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

# =============================================================================
# PASO 13: EVALUACIÓN Y COMPARACIÓN DE MODELOS
# =============================================================================

print("\n📊 RESUMEN DE RESULTADOS:")
print("="*60)

results_df = pd.DataFrame({
    name: {metric: results[name][metric] for metric in ['MAE', 'RMSE', 'R2']}
    for name in results.keys()
}).round(3)

print(results_df)

# Encontrar el mejor modelo
best_model_name = min(results.keys(), key=lambda x: results[x]['MAE'])
best_model = results[best_model_name]['model']

print(f"\n🏆 MEJOR MODELO: {best_model_name}")
print(f"   - MAE: {results[best_model_name]['MAE']:.3f}")
print(f"   - RMSE: {results[best_model_name]['RMSE']:.3f}")
print(f"   - R2: {results[best_model_name]['R2']:.3f}")

# =============================================================================
# PASO 14: ANÁLISIS DE IMPORTANCIA DE FEATURES
# =============================================================================

print("\n🎯 ANÁLISIS DE IMPORTANCIA DE FEATURES...")

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features más importantes:")
    print(feature_importance.head(10))
    
elif hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': abs(best_model.coef_)
    }).sort_values('coefficient', ascending=False)
    
    print("\nTop 10 features con mayores coeficientes:")
    print(feature_importance.head(10))

# =============================================================================
# PASO 15: VALIDACIÓN CON SERIES TEMPORALES
# =============================================================================

print("\n📈 VALIDACIÓN CON TIME SERIES SPLIT...")

# Usar TimeSeriesSplit para validación más robusta
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []

# Preparar datos ordenados por tiempo para CV
train_data_sorted = train_data.sort_values('timestamp')
X_cv = train_data_sorted[feature_columns]
y_cv = train_data_sorted['target']

# Validación cruzada temporal
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_cv)):
    X_train_cv, X_val_cv = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
    y_train_cv, y_val_cv = y_cv.iloc[train_idx], y_cv.iloc[val_idx]
    
    # Entrenar modelo del best performer
    if best_model_name in ['Ridge Regression', 'Lasso Regression']:
        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler.transform(X_val_cv)
        temp_model = type(best_model)(**best_model.get_params())
        temp_model.fit(X_train_cv_scaled, y_train_cv)
        y_pred_cv = temp_model.predict(X_val_cv_scaled)
    else:
        temp_model = type(best_model)(**best_model.get_params())
        temp_model.fit(X_train_cv, y_train_cv)
        y_pred_cv = temp_model.predict(X_val_cv)
    
    mae_cv = mean_absolute_error(y_val_cv, y_pred_cv)
    cv_scores.append(mae_cv)
    
    print(f"   Fold {fold+1}: MAE = {mae_cv:.3f}")

print(f"\n✅ Validación cruzada completada:")
print(f"   - MAE promedio: {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")

# =============================================================================
# PASO 16: FUNCIÓN DE PREDICCIÓN
# =============================================================================

print("\n🔮 CREANDO FUNCIÓN DE PREDICCIÓN...")

def predecir_arribos_futuro(timestamp_inicio, estaciones_partidas, modelo=best_model, 
                           scaler_obj=scaler, usar_escalado=best_model_name in ['Ridge Regression', 'Lasso Regression']):
    """
    Predice arribos de bicicletas para todas las estaciones en los próximos 30 minutos
    
    Parameters:
    timestamp_inicio: datetime - momento desde el cual predecir
    estaciones_partidas: dict - {id_estacion: cantidad_partidas} en últimos 30 min
    """
    
    predicciones = {}
    
    for id_estacion in todas_las_estaciones['id_estacion'].unique():
        # Obtener info de la estación
        estacion_info = todas_las_estaciones[todas_las_estaciones['id_estacion'] == id_estacion].iloc[0]
        
        # Crear features para la predicción
        features = {
            'hora': timestamp_inicio.hour,
            'dia_semana': timestamp_inicio.weekday(),
            'es_fin_de_semana': 1 if timestamp_inicio.weekday() >= 5 else 0,
            'mes': timestamp_inicio.month,
            'dia': timestamp_inicio.day,
            'lat_estacion': estacion_info['lat_estacion'],
            'long_estacion': estacion_info['long_estacion'],
            'distancia_al_centro': np.sqrt((estacion_info['lat_estacion'] - centro_lat)**2 + 
                                         (estacion_info['long_estacion'] - centro_long)**2),
            'partidas': estaciones_partidas.get(id_estacion, 0),
            'partidas_lag_1': 0,  # Simplificado para el ejemplo
            'partidas_lag_2': 0,
            'partidas_lag_3': 0,
            'partidas_lag_6': 0,
            'partidas_rolling_mean_3': estaciones_partidas.get(id_estacion, 0),
            'partidas_rolling_sum_6': estaciones_partidas.get(id_estacion, 0)
        }
        
        # Agregar features de usuario si existen
        if 'total_viajes_usuarios' in feature_columns:
            features['total_viajes_usuarios'] = estaciones_partidas.get(id_estacion, 0)
            features['usuarios_unicos'] = min(estaciones_partidas.get(id_estacion, 0), 1)
        
        # Convertir a array para predicción
        X_pred = np.array([[features[col] for col in feature_columns]])
        
        # Hacer predicción
        if usar_escalado:
            X_pred_scaled = scaler_obj.transform(X_pred)
            pred = modelo.predict(X_pred_scaled)[0]
        else:
            pred = modelo.predict(X_pred)[0]
        
        predicciones[id_estacion] = max(0, round(pred))  # No puede ser negativo
    
    return predicciones

print("✅ Función de predicción creada")

# =============================================================================
# PASO 17: EJEMPLO DE USO
# =============================================================================

print("\n🧪 EJEMPLO DE PREDICCIÓN...")

# Crear un ejemplo de partidas por estación en los últimos 30 min
ejemplo_timestamp = datetime(2024, 8, 15, 14, 30)  # Ejemplo: 15 de agosto 2024, 14:30
ejemplo_partidas = {
    list(todas_las_estaciones['id_estacion'])[0]: 5,
    list(todas_las_estaciones['id_estacion'])[1]: 3,
    list(todas_las_estaciones['id_estacion'])[2]: 8,
    # Resto con 0 partidas
}

# Completar con 0s para todas las estaciones
for id_est in todas_las_estaciones['id_estacion']:
    if id_est not in ejemplo_partidas:
        ejemplo_partidas[id_est] = 0

# Hacer predicción
predicciones_ejemplo = predecir_arribos_futuro(ejemplo_timestamp, ejemplo_partidas)

print(f"🎯 Predicción para {ejemplo_timestamp.strftime('%Y-%m-%d %H:%M')}:")
print("Top 10 estaciones con más arribos predichos:")

# Mostrar top 10 predicciones
pred_sorted = sorted(predicciones_ejemplo.items(), key=lambda x: x[1], reverse=True)
for i, (id_estacion, arribos) in enumerate(pred_sorted[:10]):
    nombre = todas_las_estaciones[todas_las_estaciones['id_estacion']==id_estacion]['nombre_estacion'].iloc[0]
    print(f"   {i+1}. Estación {id_estacion} ({nombre[:30]}...): {arribos} arribos")

# =============================================================================
# PASO 18: ANÁLISIS DE PATRONES TEMPORALES
# =============================================================================

print("\n📊 ANÁLISIS DE PATRONES TEMPORALES...")

# Análisis por hora del día
patron_hora = dataset_clean.groupby('hora')['target'].mean()
print("\nPromedio de arribos por hora del día:")
for hora, arribos in patron_hora.items():
    print(f"   {hora:02d}:00 - {arribos:.1f} arribos promedio")

# Análisis por día de la semana
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
patron_dia = dataset_clean.groupby('dia_semana')['target'].mean()
print("\nPromedio de arribos por día de la semana:")
for dia_num, arribos in patron_dia.items():
    print(f"   {dias_semana[dia_num]}: {arribos:.1f} arribos promedio")

# =============================================================================
# PASO 19: MÉTRICAS DE NEGOCIO
# =============================================================================

print("\n💼 MÉTRICAS DE NEGOCIO...")

# Calcular precisión en predicción de alta demanda (top 20% de arribos)
threshold_alta_demanda = y_test.quantile(0.8)
y_test_alta = (y_test >= threshold_alta_demanda).astype(int)
y_pred_best = results[best_model_name]['predictions']
y_pred_alta = (y_pred_best >= threshold_alta_demanda).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score

precision_alta = precision_score(y_test_alta, y_pred_alta, zero_division=0)
recall_alta = recall_score(y_test_alta, y_pred_alta, zero_division=0)
f1_alta = f1_score(y_test_alta, y_pred_alta, zero_division=0)

print(f"📈 Predicción de alta demanda (top 20%):")
print(f"   - Precision: {precision_alta:.3f}")
print(f"   - Recall: {recall_alta:.3f}")
print(f"   - F1-Score: {f1_alta:.3f}")

# Error relativo promedio
error_relativo = abs(y_test - y_pred_best) / (y_test + 1)  # +1 para evitar división por 0
print(f"\n📊 Error relativo promedio: {error_relativo.mean():.1%}")

# =============================================================================
# PASO 20: CONCLUSIONES Y PRÓXIMOS PASOS
# =============================================================================

print("\n" + "="*80)
print("🎉 RESUMEN FINAL DEL PROYECTO")
print("="*80)

print(f"\n📊 DATASET PROCESADO:")
print(f"   - Registros totales procesados: {len(dataset_clean):,}")
print(f"   - Estaciones monitoreadas: {todas_las_estaciones['id_estacion'].nunique()}")
print(f"   - Ventanas temporales de: {DELTA_T_MINUTES} minutos")
print(f"   - Período analizado: {trips_df['fecha_origen_recorrido'].min().strftime('%Y-%m-%d')} a {trips_df['fecha_origen_recorrido'].max().strftime('%Y-%m-%d')}")

print(f"\n🤖 MEJOR MODELO: {best_model_name}")
print(f"   - Error absoluto medio: {results[best_model_name]['MAE']:.2f} arribos")
print(f"   - Error cuadrático medio: {results[best_model_name]['RMSE']:.2f} arribos")
print(f"   - R² Score: {results[best_model_name]['R2']:.3f}")
print(f"   - Error relativo promedio: {error_relativo.mean():.1%}")

print(f"\n🎯 APLICACIONES PRÁCTICAS:")
print(f"   - Redistribución proactiva de bicicletas")
print(f"   - Optimización de recursos de mantenimiento")
print(f"   - Planificación de capacidad por estación")
print(f"   - Alertas tempranas de alta/baja demanda")

print(f"\n🔮 PREDICCIÓN EJEMPLO:")
print(f"   - Para el {ejemplo_timestamp.strftime('%Y-%m-%d %H:%M')}")
print(f"   - Total arribos predichos: {sum(predicciones_ejemplo.values())} bicicletas")
print(f"   - Estación con mayor demanda predicha: {max(predicciones_ejemplo.values())} arribos")

print(f"\n📈 PRÓXIMOS PASOS RECOMENDADOS:")
print(f"   1. Implementar modelos más sofisticados (LSTM, XGBoost)")
print(f"   2. Incluir datos meteorológicos y eventos especiales")
print(f"   3. Desarrollar API para predicciones en tiempo real")
print(f"   4. Crear dashboard de monitoreo operativo")
print(f"   5. Validar con datos de septiembre 2024 en adelante")

print("\n🚴 ¡PROYECTO COMPLETADO EXITOSAMENTE! 🚴")
print("="*80)
