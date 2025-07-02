"""
Entrenamiento con Features Globales
Entrena un modelo XGBoost usando despachos de TODAS las estaciones como features.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import time
from feature_engineering import prepare_features_target
from utils import (
    TARGET_STATION_ID, TARGET_STATION_NAME,
    evaluate_model, save_model, print_feature_importance
)

def train_xgboost_global(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo XGBoost optimizado para el problema de regresión.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
    
    Returns:
        modelo entrenado
    """
    print("\n🚀 ENTRENANDO MODELO XGBOOST CON FEATURES GLOBALES")
    print("="*60)
    
    # Configuración optimizada para regresión
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,          # L1 regularization
        'reg_lambda': 1.0,         # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
        'early_stopping_rounds': 100
    }
    
    print("⚙️  Configuración del modelo:")
    for param, value in xgb_params.items():
        print(f"   {param}: {value}")
    
    # Crear el modelo
    model = xgb.XGBRegressor(**xgb_params)
    
    print(f"\n🔄 Iniciando entrenamiento...")
    start_time = time.time()
    
    # Entrenar con early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
    print(f"🌳 Árboles utilizados: {model.best_iteration + 1}")
    
    return model

def analyze_predictions(y_val, y_pred, model_name="Global"):
    """
    Analiza las predicciones en detalle.
    
    Args:
        y_val: Valores reales
        y_pred: Predicciones del modelo
        model_name: Nombre del modelo para logging
    """
    print(f"\n🔍 ANÁLISIS DETALLADO DE PREDICCIONES - {model_name.upper()}")
    print("="*60)
    
    # Estadísticas básicas
    print(f"📊 VALORES REALES:")
    print(f"   Min: {y_val.min()}")
    print(f"   Max: {y_val.max()}")
    print(f"   Media: {y_val.mean():.2f}")
    print(f"   Std: {y_val.std():.2f}")
    
    print(f"\n📈 PREDICCIONES:")
    print(f"   Min: {y_pred.min():.2f}")
    print(f"   Max: {y_pred.max():.2f}")
    print(f"   Media: {y_pred.mean():.2f}")
    print(f"   Std: {y_pred.std():.2f}")
    
    # Análisis de errores
    errors = y_pred - y_val
    abs_errors = np.abs(errors)
    
    print(f"\n❌ ERRORES:")
    print(f"   Error medio: {errors.mean():.2f}")
    print(f"   Error absoluto medio: {abs_errors.mean():.2f}")
    print(f"   Error máximo: {abs_errors.max():.2f}")
    
    # Percentiles de errores
    print(f"\n📏 PERCENTILES DE ERRORES ABSOLUTOS:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"   P{p}: {np.percentile(abs_errors, p):.2f}")
    
    # Análisis por rangos de valores
    print(f"\n📊 PERFORMANCE POR RANGOS:")
    ranges = [(0, 2), (2, 5), (5, 10), (10, float('inf'))]
    
    for low, high in ranges:
        if high == float('inf'):
            mask = y_val >= low
            range_name = f">= {low}"
        else:
            mask = (y_val >= low) & (y_val < high)
            range_name = f"{low}-{high}"
        
        if mask.sum() > 0:
            range_mae = abs_errors[mask].mean()
            range_count = mask.sum()
            print(f"   Rango {range_name}: MAE = {range_mae:.2f} ({range_count} samples)")

def main():
    """
    Pipeline principal de entrenamiento con features globales.
    """
    print("🌐 ENTRENAMIENTO CON FEATURES GLOBALES")
    print("="*70)
    print(f"🎯 Estación objetivo: {TARGET_STATION_NAME} (ID: {TARGET_STATION_ID})")
    
    # 1. Cargar datos
    print("\n📂 Cargando datasets...")
    train_df = pd.read_csv('../data/train_global.csv')
    val_df = pd.read_csv('../data/val_global.csv')
    
    print(f"✅ Train: {train_df.shape}")
    print(f"✅ Val: {val_df.shape}")
    
    # Información sobre el target
    print(f"\n🎯 INFORMACIÓN DEL TARGET:")
    print(f"   Train - Arribos promedio: {train_df['target_arribos'].mean():.2f}")
    print(f"   Train - Arribos std: {train_df['target_arribos'].std():.2f}")
    print(f"   Val - Arribos promedio: {val_df['target_arribos'].mean():.2f}")
    print(f"   Val - Arribos std: {val_df['target_arribos'].std():.2f}")
    
    # 2. Preparar features y target
    X_train, y_train, X_val, y_val, feature_names = prepare_features_target(train_df, val_df)
    
    # 3. Entrenar modelo
    model = train_xgboost_global(X_train, y_train, X_val, y_val)
    
    # 4. Evaluar modelo
    metrics, y_pred = evaluate_model(
        model, X_train, y_train, X_val, y_val, 
        model_name="XGBoost Global"
    )
    
    # 5. Análisis detallado de predicciones
    analyze_predictions(y_val, y_pred, "Global")
    
    # 6. Feature importance
    importance_df = print_feature_importance(model, feature_names, top_n=20)
    
    # 7. Guardar modelo
    target_info = {
        'approach': 'global_features',
        'description': 'Usa despachos de todas las estaciones como features',
        'num_features': len(feature_names),
        'train_period': f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}",
        'val_period': f"{val_df['timestamp'].min()} to {val_df['timestamp'].max()}"
    }
    
    save_model(
        model=model,
        features=feature_names,
        metrics=metrics,
        model_name="xgboost_global_features",
        target_info=target_info
    )
    
    # 8. Guardar feature importance
    if importance_df is not None:
        importance_df.to_csv('../data/feature_importance_global.csv', index=False)
        print(f"💾 Feature importance guardada: ../data/feature_importance_global.csv")
    
    print("\n🎉 ENTRENAMIENTO COMPLETADO")
    print("="*50)
    print(f"📊 Resultado principal:")
    print(f"   🎯 R² validación: {metrics['val_r2']:.4f}")
    print(f"   📏 RMSE validación: {metrics['val_rmse']:.4f}")
    print(f"   📐 MAE validación: {metrics['val_mae']:.4f}")
    
    if metrics['val_r2'] > 0.5:
        print("✅ ¡Excelente performance!")
    elif metrics['val_r2'] > 0.3:
        print("👍 Buena performance")
    elif metrics['val_r2'] > 0.1:
        print("⚠️  Performance moderada")
    else:
        print("❌ Performance baja - revisar features o hiperparámetros")
    
    print(f"\n🔗 Próximo paso:")
    print("   Ejecutar: python train_nearby_features.py")

if __name__ == "__main__":
    main() 