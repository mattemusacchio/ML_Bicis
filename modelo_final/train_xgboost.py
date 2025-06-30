#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.train_model import BikeXGBoostTrainer

def main():
    """Entrenar modelo XGBoost para predicción de arribos"""
    
    # Configuración
    input_file = '../data/processed/trips_final_features.csv'
    model_output = '../models/xgboost_bike_arrivals.pkl'
    
    # Verificar que existe el archivo de entrada
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo {input_file}")
        print("Ejecuta primero el feature engineering con run_pipeline.py")
        return
    
    print("=== ENTRENAMIENTO XGBOOST - PREDICCIÓN DE ARRIBOS ===")
    
    # Cargar datos
    print("Cargando datos procesados...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Dataset cargado: {df.shape}")
    
    # Verificar variable objetivo
    if 'N_arribos_intervalo' not in df.columns:
        print("Error: Variable objetivo N_arribos_intervalo no encontrada")
        return
    
    print(f"Variable objetivo stats:")
    print(f"  - Min: {df['N_arribos_intervalo'].min()}")
    print(f"  - Max: {df['N_arribos_intervalo'].max()}")
    print(f"  - Media: {df['N_arribos_intervalo'].mean():.2f}")
    print(f"  - Std: {df['N_arribos_intervalo'].std():.2f}")
    
    # Crear entrenador
    trainer = BikeXGBoostTrainer(poisson_max_delta_step=0.7)
    
    # Parámetros del modelo
    model_params = {
        'objective': 'count:poisson',
        'eval_metric': 'poisson-nloglik',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'max_delta_step': 0.7,
        'tree_method': 'gpu_hist'
    }
    
    # Entrenar modelo
    try:
        results, test_data = trainer.train(df, model_params)
        
        # Crear directorio de modelos si no existe
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        
        # Guardar modelo
        trainer.save_model(model_output, results)
        
        # Mostrar feature importance
        print("\n=== FEATURE IMPORTANCE (TOP 20) ===")
        importance = trainer.model.get_score(importance_type='weight')
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        for i, (feature, score) in enumerate(sorted_importance[:20]):
            print(f"{i+1:2d}. {feature:<35} {score:>6}")
        
        print(f"\n✅ Entrenamiento completado exitosamente")
        print(f"   Modelo guardado: {model_output}")
        print(f"   Features usadas: {results['n_features']}")
        print(f"   Test RMSE: {results['test_rmse']:.4f}")
        print(f"   Test MAE: {results['test_mae']:.4f}")
        
        # Analizar predicciones
        X_test, y_test, test_pred = test_data
        print(f"\n📊 Análisis de predicciones (test set):")
        print(f"   Real - Min: {y_test.min():.2f}, Max: {y_test.max():.2f}, Media: {y_test.mean():.2f}")
        print(f"   Pred - Min: {test_pred.min():.2f}, Max: {test_pred.max():.2f}, Media: {test_pred.mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main() 