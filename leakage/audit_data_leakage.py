import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def audit_data_leakage(df_path):
    """
    Auditoría completa para detectar data leakage en el dataset de bicicletas
    """
    print("=== 🕵️ AUDITORÍA DE DATA LEAKAGE ===")
    
    # Cargar dataset
    df = pd.read_csv(df_path)
    print(f"Dataset cargado: {df.shape}")
    
    # 1. VERIFICAR ORDENACIÓN TEMPORAL
    print("\n1. 🕒 VERIFICANDO ORDENACIÓN TEMPORAL...")
    
    # Convertir fechas
    df['fecha_intervalo_dt'] = pd.to_datetime(df['fecha_intervalo'])
    df['ventana_arribo_dt'] = pd.to_datetime(df['ventana_arribo']) if 'ventana_arribo' in df.columns else df['fecha_intervalo_dt']
    
    # Verificar por estación
    sample_stations = df['estacion_referencia'].unique()[:5]
    
    for station in sample_stations:
        station_data = df[df['estacion_referencia'] == station].sort_values('fecha_intervalo_dt')
        
        # Verificar LAG1
        if 'N_ARRIBOS_LAG1' in df.columns:
            lag1_future_leak = 0
            for i in range(1, min(10, len(station_data))):
                current_time = station_data.iloc[i]['fecha_intervalo_dt']
                lag1_value = station_data.iloc[i]['N_ARRIBOS_LAG1']
                prev_time = station_data.iloc[i-1]['fecha_intervalo_dt']
                prev_arribos = station_data.iloc[i-1]['N_arribos_intervalo']
                
                # El LAG1 debería ser igual a los arribos del período anterior
                if not pd.isna(lag1_value) and not pd.isna(prev_arribos):
                    if abs(lag1_value - prev_arribos) > 0.1:
                        lag1_future_leak += 1
            
            print(f"   Estación {station}: {lag1_future_leak}/10 inconsistencias en LAG1")
    
    # 2. VERIFICAR CORRELACIONES SOSPECHOSAS
    print("\n2. 📊 VERIFICANDO CORRELACIONES...")
    
    target = 'N_arribos_intervalo'
    lag_features = [col for col in df.columns if 'LAG' in col and 'ARRIBOS' in col]
    
    correlations = {}
    for feature in lag_features:
        if feature in df.columns:
            corr = df[target].corr(df[feature])
            correlations[feature] = corr
            
    # Ordenar por correlación
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("   Correlaciones TARGET vs LAG features:")
    for feature, corr in sorted_corr[:10]:
        status = "🚨 SOSPECHOSO" if abs(corr) > 0.8 else "✅ OK"
        print(f"   {feature}: {corr:.4f} {status}")
    
    # 3. VERIFICAR VENTANAS TEMPORALES
    print("\n3. ⏰ VERIFICANDO VENTANAS TEMPORALES...")
    
    # Verificar si hay overlaps temporales
    df_sample = df.sample(1000).copy()
    df_sample['fecha_origen_dt'] = pd.to_datetime(df_sample['fecha_origen_recorrido'])
    df_sample['fecha_destino_dt'] = pd.to_datetime(df_sample['fecha_destino_recorrido'])
    
    future_info_count = 0
    for idx, row in df_sample.iterrows():
        intervalo_time = row['fecha_intervalo_dt']
        destino_time = row['fecha_destino_dt']
        
        # El intervalo debería ser DESPUÉS del destino (o igual)
        if intervalo_time < destino_time:
            future_info_count += 1
    
    print(f"   Casos con información del futuro: {future_info_count}/1000")
    
    # 4. VERIFICAR SHIFT CORRECTO
    print("\n4. 🔄 VERIFICANDO SHIFT CORRECTO...")
    
    # Tomar una estación específica y verificar manualmente
    test_station = df['estacion_referencia'].value_counts().index[0]
    station_data = df[df['estacion_referencia'] == test_station].copy()
    station_data = station_data.sort_values('fecha_intervalo_dt').reset_index(drop=True)
    
    print(f"   Verificando estación {test_station} con {len(station_data)} registros:")
    
    # Verificar primeros 5 registros
    for i in range(min(5, len(station_data))):
        row = station_data.iloc[i]
        fecha = row['fecha_intervalo_dt']
        arribos_actual = row['N_arribos_intervalo']
        arribos_lag1 = row['N_ARRIBOS_LAG1'] if 'N_ARRIBOS_LAG1' in station_data.columns else 'N/A'
        
        print(f"   [{i}] {fecha}: Actual={arribos_actual}, LAG1={arribos_lag1}")
    
    # 5. VERIFICAR FEATURES TEMPORALES
    print("\n5. 📅 VERIFICANDO FEATURES TEMPORALES...")
    
    temporal_features = ['año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo']
    
    inconsistencies = 0
    sample_data = df.sample(100)
    
    for idx, row in sample_data.iterrows():
        fecha_intervalo = pd.to_datetime(row['fecha_intervalo'])
        
        checks = [
            row['año_intervalo'] == fecha_intervalo.year,
            row['mes_intervalo'] == fecha_intervalo.month,
            row['dia_intervalo'] == fecha_intervalo.day,
            row['hora_intervalo'] == fecha_intervalo.hour
        ]
        
        if not all(checks):
            inconsistencies += 1
    
    print(f"   Inconsistencias en features temporales: {inconsistencies}/100")
    
    # 6. RECOMENDACIONES
    print("\n6. 💡 RECOMENDACIONES...")
    
    high_corr_features = [f for f, c in sorted_corr if abs(c) > 0.8]
    
    recommendations = []
    
    if future_info_count > 50:
        recommendations.append("🚨 Revisar definición de ventanas temporales")
    
    if len(high_corr_features) > 0:
        recommendations.append(f"🚨 Investigar features con correlación alta: {high_corr_features}")
    
    if inconsistencies > 10:
        recommendations.append("🚨 Revisar cálculo de features temporales")
    
    recommendations.append("✅ Implementar shift correcto por estación y ventana temporal")
    recommendations.append("✅ Asegurar que LAG1 = arribos del intervalo [T-30, T]")
    recommendations.append("✅ Verificar que target = arribos del intervalo [T, T+30]")
    
    print("   Acciones recomendadas:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return {
        'correlations': correlations,
        'future_info_count': future_info_count,
        'temporal_inconsistencies': inconsistencies,
        'recommendations': recommendations
    }

def create_corrected_lags(df_original_path, output_path):
    """
    Crear versión corregida sin data leakage
    """
    print("\n=== 🔧 CREANDO VERSIÓN CORREGIDA ===")
    
    # Cargar datos originales (antes del feature engineering problemático)
    df = pd.read_csv(df_original_path)
    
    # Convertir fechas
    df['ventana_arribo_dt'] = pd.to_datetime(df['ventana_arribo'])
    df['ventana_despacho_dt'] = pd.to_datetime(df['ventana_despacho'])
    
    # 1. Crear conteos por ventana PRIMERO
    print("1. Calculando conteos por ventana temporal...")
    
    # Conteos de arribos por ventana de 30 min y estación
    arribos_ventana = df.groupby(['ventana_arribo', 'id_estacion_destino']).size().reset_index(name='N_arribos_ventana')
    
    # Conteos de salidas por ventana de 30 min y estación  
    salidas_ventana = df.groupby(['ventana_despacho', 'id_estacion_origen']).size().reset_index(name='N_salidas_ventana')
    
    # 2. Crear serie temporal completa por estación
    print("2. Creando serie temporal completa...")
    
    # Obtener rango completo de ventanas
    min_ventana = min(df['ventana_arribo'].min(), df['ventana_despacho'].min())
    max_ventana = max(df['ventana_arribo'].max(), df['ventana_despacho'].max())
    
    # Crear todas las ventanas posibles
    ventanas_completas = pd.date_range(start=min_ventana, end=max_ventana, freq='30T')
    estaciones = sorted(df['id_estacion_destino'].dropna().unique())
    
    # Producto cartesiano: todas las combinaciones estación-ventana
    ventana_estacion = []
    for estacion in estaciones:
        for ventana in ventanas_completas:
            ventana_estacion.append({
                'id_estacion': estacion,
                'ventana_tiempo': ventana,
                'N_arribos_actual': 0,
                'N_salidas_actual': 0
            })
    
    df_temporal = pd.DataFrame(ventana_estacion)
    
    # 3. Merge con conteos reales
    print("3. Haciendo merge con conteos reales...")
    
    # Merge arribos
    df_temporal = df_temporal.merge(
        arribos_ventana.rename(columns={'ventana_arribo': 'ventana_tiempo', 'id_estacion_destino': 'id_estacion'}),
        on=['ventana_tiempo', 'id_estacion'],
        how='left'
    )
    df_temporal['N_arribos_actual'] = df_temporal['N_arribos_ventana'].fillna(0)
    
    # Merge salidas  
    df_temporal = df_temporal.merge(
        salidas_ventana.rename(columns={'ventana_despacho': 'ventana_tiempo', 'id_estacion_origen': 'id_estacion'}),
        on=['ventana_tiempo', 'id_estacion'], 
        how='left'
    )
    df_temporal['N_salidas_actual'] = df_temporal['N_salidas_ventana'].fillna(0)
    
    # 4. Crear LAGS CORRECTOS
    print("4. Creando LAGs correctos por estación...")
    
    df_temporal = df_temporal.sort_values(['id_estacion', 'ventana_tiempo']).reset_index(drop=True)
    
    # Crear lags por estación
    for lag in range(1, 7):
        df_temporal[f'N_ARRIBOS_LAG{lag}'] = df_temporal.groupby('id_estacion')['N_arribos_actual'].shift(lag)
        df_temporal[f'N_SALIDAS_LAG{lag}'] = df_temporal.groupby('id_estacion')['N_salidas_actual'].shift(lag)
    
    # Calcular promedios
    df_temporal['N_ARRIBOS_PROM_2INT'] = (df_temporal['N_ARRIBOS_LAG1'] + df_temporal['N_ARRIBOS_LAG2']) / 2
    df_temporal['N_SALIDAS_PROM_2INT'] = (df_temporal['N_SALIDAS_LAG1'] + df_temporal['N_SALIDAS_LAG2']) / 2
    
    # 5. Merge de vuelta con dataset original
    print("5. Merge con dataset original...")
    
    # Preparar merge keys
    df['merge_key'] = df['ventana_arribo'] + '_' + df['id_estacion_destino'].astype(str)
    df_temporal['merge_key'] = df_temporal['ventana_tiempo'].astype(str) + '_' + df_temporal['id_estacion'].astype(str)
    
    # Merge
    df_final = df.merge(
        df_temporal[['merge_key', 'N_arribos_actual'] + [col for col in df_temporal.columns if 'LAG' in col or 'PROM' in col]],
        on='merge_key',
        how='left'
    )
    
    # Renombrar target
    df_final['N_arribos_intervalo'] = df_final['N_arribos_actual']
    
    # Guardar
    df_final.to_csv(output_path, index=False)
    print(f"✅ Dataset corregido guardado en: {output_path}")
    
    return df_final

if __name__ == "__main__":
    # Ejecutar auditoría
    audit_results = audit_data_leakage('data/processed/trips_features_engineered_fixed.csv')
    
    # Crear versión corregida
    df_corrected = create_corrected_lags(
        'data/processed/trips_con_ventanas.csv',
        'data/processed/trips_no_leakage.csv'
    ) 