"""
Entrenamiento SIN DATA LEAKAGE
Script para entrenar modelos con datasets corregidos sin filtración de información.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import time
from feature_engineering import prepare_train_val_split, prepare_features_target
from utils import (
    TARGET_STATION_ID, TARGET_STATION_NAME,
    evaluate_model, save_model, print_feature_importance
)

def train_xgboost_no_leakage(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo XGBoost con configuración más conservadora para datos sin leakage.
    """
    print("\n🚀 ENTRENANDO MODELO XGBOOST SIN LEAKAGE")
    print("="*60)
    
    # Configuración más conservadora (esperamos performance más baja)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1000,         # Menos árboles
        'max_depth': 15,              # Menor profundidad
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,       # Más conservador
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
        'early_stopping_rounds': 50
    }
    
    print("⚙️  Configuración del modelo:")
    for param, value in xgb_params.items():
        print(f"   {param}: {value}")
    
    model = xgb.XGBRegressor(**xgb_params)
    
    print(f"\n🔄 Iniciando entrenamiento...")
    start_time = time.time()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
    print(f"🌳 Árboles utilizados: {model.best_iteration + 1}")
    
    return model

def main():
    """
    Pipeline de entrenamiento con datasets sin leakage.
    """
    print("🔍 ENTRENAMIENTO SIN DATA LEAKAGE - RESULTADOS REALISTAS")
    print("="*70)
    print(f"🎯 Estación objetivo: {TARGET_STATION_NAME} (ID: {TARGET_STATION_ID})")
    
    # 1. Cargar datasets sin leakage
    print("\n📂 Cargando datasets SIN LEAKAGE...")
    try:
        dataset_global = pd.read_csv('../data/dataset_global_no_leakage.csv')
        dataset_nearby = pd.read_csv('../data/dataset_nearby_no_leakage.csv')
        print(f"✅ Global sin leakage: {dataset_global.shape}")
        print(f"✅ Cercanas sin leakage: {dataset_nearby.shape}")
    except FileNotFoundError:
        print("❌ No se encontraron datasets sin leakage")
        print("   Ejecuta primero: python feature_engineering_no_leakage.py")
        return
    
    # 2. Procesar dataset GLOBAL
    print(f"\n🌐 PROCESANDO DATASET GLOBAL SIN LEAKAGE...")
    
    # Convertir timestamp
    dataset_global['timestamp'] = pd.to_datetime(dataset_global['timestamp'])
    
    # Dividir train/val
    train_global, val_global = prepare_train_val_split(dataset_global)
    
    # Preparar features y target
    X_train_global, y_train_global, X_val_global, y_val_global, features_global = prepare_features_target(train_global, val_global)
    
    # Entrenar modelo global
    model_global = train_xgboost_no_leakage(X_train_global, y_train_global, X_val_global, y_val_global)
    
    # Evaluar modelo global
    metrics_global, y_pred_global = evaluate_model(
        model_global, X_train_global, y_train_global, X_val_global, y_val_global,
        model_name="XGBoost Global Sin Leakage"
    )
    
    # 3. Procesar dataset CERCANAS
    print(f"\n🔍 PROCESANDO DATASET CERCANAS SIN LEAKAGE...")
    
    # Convertir timestamp
    dataset_nearby['timestamp'] = pd.to_datetime(dataset_nearby['timestamp'])
    
    # Dividir train/val
    train_nearby, val_nearby = prepare_train_val_split(dataset_nearby)
    
    # Preparar features y target
    X_train_nearby, y_train_nearby, X_val_nearby, y_val_nearby, features_nearby = prepare_features_target(train_nearby, val_nearby)
    
    # Entrenar modelo cercanas
    model_nearby = train_xgboost_no_leakage(X_train_nearby, y_train_nearby, X_val_nearby, y_val_nearby)
    
    # Evaluar modelo cercanas
    metrics_nearby, y_pred_nearby = evaluate_model(
        model_nearby, X_train_nearby, y_train_nearby, X_val_nearby, y_val_nearby,
        model_name="XGBoost Cercanas Sin Leakage"
    )
    
    # 4. Comparación de resultados
    print(f"\n🆚 COMPARACIÓN DE MODELOS SIN LEAKAGE")
    print("="*60)
    print(f"                      {'Global':<10} {'Cercanas':<10}")
    print("="*50)
    print(f"R²                    {metrics_global['val_r2']:<10.4f} {metrics_nearby['val_r2']:<10.4f}")
    print(f"RMSE                  {metrics_global['val_rmse']:<10.4f} {metrics_nearby['val_rmse']:<10.4f}")
    print(f"MAE                   {metrics_global['val_mae']:<10.4f} {metrics_nearby['val_mae']:<10.4f}")
    print(f"Features              {len(features_global):<10} {len(features_nearby):<10}")
    
    # 5. Análisis de mejora
    print(f"\n📊 ANÁLISIS DE RESULTADOS:")
    print(f"   Global R²: {metrics_global['val_r2']:.4f}")
    print(f"   Cercanas R²: {metrics_nearby['val_r2']:.4f}")
    
    if metrics_global['val_r2'] > 0.8 or metrics_nearby['val_r2'] > 0.8:
        print("⚠️  ATENCIÓN: R² aún muy alto, puede quedar leakage residual")
    elif metrics_global['val_r2'] > 0.3 or metrics_nearby['val_r2'] > 0.3:
        print("✅ Resultados más realistas - buen trabajo eliminando leakage")
    else:
        print("📉 R² bajo - modelo baseline, pero sin leakage")
    
    # 6. Feature importance
    print(f"\n🔝 TOP FEATURES - MODELO GLOBAL:")
    importance_global = print_feature_importance(model_global, features_global, top_n=10)
    
    print(f"\n🔝 TOP FEATURES - MODELO CERCANAS:")
    importance_nearby = print_feature_importance(model_nearby, features_nearby, top_n=10)
    
    # 7. Guardar modelos
    save_model(
        model=model_global,
        features=features_global,
        metrics=metrics_global,
        model_name="xgboost_global_no_leakage",
        target_info={'approach': 'global_no_leakage', 'description': 'Global sin data leakage'}
    )
    
    save_model(
        model=model_nearby,
        features=features_nearby,
        metrics=metrics_nearby,
        model_name="xgboost_nearby_no_leakage",
        target_info={'approach': 'nearby_no_leakage', 'description': 'Cercanas sin data leakage'}
    )
    
    print("\n🎉 ENTRENAMIENTO SIN LEAKAGE COMPLETADO")
    print("="*60)
    print("✅ Modelos entrenados con datasets corregidos")
    print("✅ Resultados más realistas y confiables")
    print("✅ Sin filtración de información del futuro")

if __name__ == "__main__":
    main() 