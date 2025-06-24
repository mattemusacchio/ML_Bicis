import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def audit_sample_data_leakage():
    """
    Auditoría simple para detectar data leakage en muestra del dataset
    """
    print("=== 🔍 AUDITORÍA SIMPLIFICADA DE DATA LEAKAGE ===")
    
    # Cargar solo una muestra pequeña
    print("Cargando muestra del dataset...")
    df = pd.read_csv('data/processed/trips_features_engineered_fixed.csv', nrows=10000)
    print(f"Muestra cargada: {df.shape}")
    
    # 1. VERIFICAR CORRELACIONES SOSPECHOSAS
    print("\n1. 📊 VERIFICANDO CORRELACIONES...")
    
    target = 'N_arribos_intervalo'
    lag_features = [col for col in df.columns if 'LAG' in col and 'ARRIBOS' in col]
    
    print(f"Target: {target}")
    print(f"LAG features encontradas: {lag_features}")
    
    correlations = {}
    for feature in lag_features:
        if feature in df.columns and target in df.columns:
            corr = df[target].corr(df[feature])
            correlations[feature] = corr
            
    # Ordenar por correlación
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]) if not pd.isna(x[1]) else 0, reverse=True)
    
    print("   Correlaciones TARGET vs LAG features:")
    for feature, corr in sorted_corr[:10]:
        if not pd.isna(corr):
            status = "🚨 SOSPECHOSO" if abs(corr) > 0.8 else "✅ OK"
            print(f"   {feature}: {corr:.4f} {status}")
    
    # 2. VERIFICAR ESTRUCTURA TEMPORAL
    print("\n2. ⏰ VERIFICANDO ESTRUCTURA TEMPORAL...")
    
    if 'fecha_intervalo' in df.columns:
        df['fecha_intervalo_dt'] = pd.to_datetime(df['fecha_intervalo'])
        
        # Verificar muestra de ordenación temporal
        sample_data = df.sample(min(100, len(df))).copy()
        
        if 'estacion_referencia' in df.columns:
            # Tomar una estación específica
            station_id = df['estacion_referencia'].value_counts().index[0]
            station_data = df[df['estacion_referencia'] == station_id].copy()
            station_data = station_data.sort_values('fecha_intervalo_dt').head(10)
            
            print(f"   Verificando estación {station_id}:")
            print("   Fecha_Intervalo | Target | LAG1 | LAG2")
            print("   " + "-"*50)
            
            for i, (idx, row) in enumerate(station_data.iterrows()):
                fecha = row['fecha_intervalo_dt']
                target_val = row.get('N_arribos_intervalo', 'N/A')
                lag1_val = row.get('N_ARRIBOS_LAG1', 'N/A')
                lag2_val = row.get('N_ARRIBOS_LAG2', 'N/A')
                
                print(f"   {fecha} | {target_val:4.0f} | {lag1_val:4.0f} | {lag2_val:4.0f}")
    
    # 3. VERIFICAR LÓGICA DE LAG
    print("\n3. 🔄 VERIFICANDO LÓGICA DE LAG...")
    
    # Verificar si LAG1 realmente corresponde al período anterior
    lag_issues = 0
    
    if 'estacion_referencia' in df.columns and 'N_ARRIBOS_LAG1' in df.columns:
        # Tomar estaciones con suficientes datos
        station_counts = df['estacion_referencia'].value_counts()
        test_stations = station_counts.head(3).index
        
        for station in test_stations:
            station_data = df[df['estacion_referencia'] == station].copy()
            station_data = station_data.sort_values('fecha_intervalo_dt').head(5)
            
            print(f"\n   Estación {station}:")
            for i in range(1, min(len(station_data), 4)):
                current_target = station_data.iloc[i]['N_arribos_intervalo']
                current_lag1 = station_data.iloc[i]['N_ARRIBOS_LAG1']
                prev_target = station_data.iloc[i-1]['N_arribos_intervalo']
                
                # LAG1 debería ser igual al target del período anterior
                if not pd.isna(current_lag1) and not pd.isna(prev_target):
                    if abs(current_lag1 - prev_target) > 0.1:
                        lag_issues += 1
                        print(f"     ❌ LAG1={current_lag1:.1f} ≠ Prev_Target={prev_target:.1f}")
                    else:
                        print(f"     ✅ LAG1={current_lag1:.1f} = Prev_Target={prev_target:.1f}")
    
    # 4. ANALIZAR IMPORTANCIA DE FEATURES
    print("\n4. 📈 IMPORTANCIA DE FEATURES:")
    
    high_corr_features = [f for f, c in sorted_corr if not pd.isna(c) and abs(c) > 0.7]
    medium_corr_features = [f for f, c in sorted_corr if not pd.isna(c) and 0.3 <= abs(c) <= 0.7]
    
    print(f"   Features con correlación alta (>0.7): {len(high_corr_features)}")
    for f in high_corr_features[:5]:
        corr_val = correlations[f]
        print(f"     - {f}: {corr_val:.3f}")
    
    print(f"   Features con correlación media (0.3-0.7): {len(medium_corr_features)}")
    
    # 5. DETECTAR PROBLEMAS ESPECÍFICOS
    print("\n5. 🚨 PROBLEMAS DETECTADOS:")
    
    problems = []
    
    if len(high_corr_features) > 0:
        top_feature = sorted_corr[0][0]
        top_corr = sorted_corr[0][1]
        if abs(top_corr) > 0.9:
            problems.append(f"⚠️  {top_feature} tiene correlación extremadamente alta ({top_corr:.3f}) - posible leakage")
    
    if lag_issues > 0:
        problems.append(f"⚠️  {lag_issues} inconsistencias en lógica de LAG detectadas")
    
    # Verificar si hay features del futuro
    future_features = [col for col in df.columns if any(term in col.lower() for term in ['destino', 'arribo']) 
                      and 'LAG' not in col and col != target]
    
    if len(future_features) > 10:
        problems.append(f"⚠️  Posibles features del futuro: {len(future_features)} features de destino/arribo")
    
    if not problems:
        problems.append("✅ No se detectaron problemas obvios en la muestra")
    
    for problem in problems:
        print(f"   {problem}")
    
    # 6. RECOMENDACIONES
    print("\n6. 💡 RECOMENDACIONES:")
    
    recommendations = [
        "🔧 Verificar que LAG1 = arribos del intervalo [T-30, T]",
        "🔧 Asegurar que target = arribos del intervalo [T, T+30]",
        "🔧 Confirmar que shift se hace POR ESTACIÓN y TIEMPO",
        "🔧 Eliminar features que usen información posterior a T"
    ]
    
    if len(high_corr_features) > 0:
        recommendations.insert(0, f"🚨 URGENTE: Revisar {high_corr_features[0]} - correlación sospechosa")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return {
        'correlations': correlations,
        'lag_issues': lag_issues,
        'high_corr_features': high_corr_features,
        'problems': problems
    }

def create_minimal_fixed_dataset():
    """
    Crear una versión simplificada sin los problemas más obvios
    """
    print("\n=== 🛠️ CREANDO DATASET CORREGIDO MÍNIMO ===")
    
    # Cargar muestra
    df = pd.read_csv('data/processed/trips_features_engineered_fixed.csv', nrows=50000)
    
    # Eliminar features claramente problemáticas
    drop_columns = []
    
    # Eliminar features que claramente vienen del futuro
    for col in df.columns:
        if any(term in col.lower() for term in ['destino_lag', 'arribo']) and 'LAG' not in col:
            if col != 'N_arribos_intervalo':  # Mantener target
                drop_columns.append(col)
    
    print(f"Eliminando {len(drop_columns)} features problemáticas...")
    df = df.drop(columns=drop_columns, errors='ignore')
    
    # Guardar versión corregida
    output_path = 'data/processed/trips_minimal_fixed.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset mínimo corregido guardado: {output_path}")
    print(f"Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    # Ejecutar auditoría simplificada
    results = audit_sample_data_leakage()
    
    # Crear versión corregida mínima
    df_fixed = create_minimal_fixed_dataset() 