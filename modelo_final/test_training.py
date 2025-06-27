#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from src.feature_engineering import BikeFeatureEngineering
from src.train_model import BikeXGBoostTrainer

def test_training():
    """Probar entrenamiento con muestra pequeña"""
    
    print("=== PRUEBA DE ENTRENAMIENTO XGBOOST ===")
    
    # Crear muestra pequeña
    print("📥 Creando muestra pequeña...")
    df_sample = pd.read_csv('data/processed/trips_enriched.csv', nrows=5000)
    
    # Feature engineering
    print("🔧 Aplicando feature engineering...")
    fe = BikeFeatureEngineering(time_window_minutes=30, n_clusters=8)  # Menos clusters para muestra
    df_processed = fe.transform(
        'data/processed/trips_enriched.csv', 
        'data/processed/trips_test_features.csv'
    )
    
    # Tomar solo las primeras 5000 filas procesadas
    df_small = df_processed.head(5000).copy()
    print(f"Muestra procesada: {df_small.shape}")
    
    # Test de features válidas
    print("\n🔍 Verificando features...")
    trainer = BikeXGBoostTrainer()
    valid_features = trainer.identify_valid_features(df_small)
    
    print(f"✅ Features válidas identificadas: {len(valid_features)}")
    print("Ejemplos de features válidas:")
    for i, feat in enumerate(valid_features[:10]):
        print(f"  {i+1}. {feat}")
    
    # Verificar variable objetivo
    target_stats = df_small['N_arribos_intervalo'].describe()
    print(f"\n📊 Variable objetivo stats:")
    print(f"  Count: {target_stats['count']}")
    print(f"  Mean:  {target_stats['mean']:.2f}")
    print(f"  Min:   {target_stats['min']}")
    print(f"  Max:   {target_stats['max']}")
    
    # Test de split temporal
    print("\n⏰ Probando split temporal...")
    train_df, val_df, test_df = trainer.temporal_split(df_small, train_ratio=0.6, val_ratio=0.2)
    
    print("✅ Split temporal exitoso")
    
    # Test rápido de entrenamiento (pocos rounds)
    print("\n🚀 Entrenamiento rápido de prueba...")
    model_params = {
        'objective': 'count:poisson',
        'eval_metric': 'poisson-nloglik',
        'max_depth': 4,
        'learning_rate': 0.3,
        'random_state': 42,
        'n_jobs': -1,
        'max_delta_step': 0.7
    }
    
    try:
        # Solo preparar datos sin entrenar el modelo completo
        X_train, y_train = trainer.prepare_data(train_df)
        X_test, y_test = trainer.prepare_data(test_df)
        
        print(f"✅ Datos preparados exitosamente:")
        print(f"   Train: {X_train.shape}")
        print(f"   Test:  {X_test.shape}")
        print(f"   Features: {len(trainer.feature_names)}")
        
        print("\n🎯 Todo listo para entrenamiento completo!")
        
    except Exception as e:
        print(f"❌ Error en preparación: {str(e)}")
        raise

if __name__ == "__main__":
    test_training() 