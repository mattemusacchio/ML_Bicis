"""
Entrenamiento con Features de Estaciones Cercanas
Entrena un modelo XGBoost usando despachos solo de estaciones geográficamente cercanas.
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

def train_xgboost_nearby(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo XGBoost optimizado para features de estaciones cercanas.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
    
    Returns:
        modelo entrenado
    """
    print("\n🚀 ENTRENANDO MODELO XGBOOST CON ESTACIONES CERCANAS")
    print("="*60)
    
    # Configuración ajustada para menor número de features
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 800,         # Menos árboles para evitar overfitting
        'max_depth': 6,              # Menor profundidad
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,     # Mayor sampling de features
        'min_child_weight': 2,
        'reg_alpha': 0.05,           # Menos regularización L1
        'reg_lambda': 0.8,           # Menos regularización L2
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1,
        'early_stopping_rounds': 80
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

def compare_with_global(metrics_nearby, global_model_path="../models/xgboost_global_features.pkl"):
    """
    Compara las métricas del modelo de estaciones cercanas con el modelo global.
    
    Args:
        metrics_nearby: Métricas del modelo de estaciones cercanas
        global_model_path: Ruta al modelo global guardado
    """
    print(f"\n🆚 COMPARACIÓN CON MODELO GLOBAL")
    print("="*50)
    
    try:
        import pickle
        with open(global_model_path, 'rb') as f:
            global_info = pickle.load(f)
        
        metrics_global = global_info['metrics']
        
        print(f"📊 MÉTRICAS COMPARATIVAS:")
        print(f"                      {'Global':<10} {'Cercanas':<10} {'Diferencia':<10}")
        print("="*50)
        
        for metric in ['val_r2', 'val_rmse', 'val_mae']:
            global_val = metrics_global[metric]
            nearby_val = metrics_nearby[metric]
            diff = nearby_val - global_val
            
            # Determinar si es mejor o peor
            if metric == 'val_r2':  # Para R², mayor es mejor
                better = "✅" if diff > 0 else "❌"
            else:  # Para RMSE y MAE, menor es mejor
                better = "✅" if diff < 0 else "❌"
            
            print(f"{metric:<15} {global_val:<10.4f} {nearby_val:<10.4f} {diff:>+9.4f} {better}")
        
        # Análisis general
        r2_diff = metrics_nearby['val_r2'] - metrics_global['val_r2']
        if r2_diff > 0.01:
            print(f"\n🎉 ¡Modelo CERCANAS es MEJOR! (+{r2_diff:.3f} R²)")
        elif r2_diff > -0.01:
            print(f"\n🤝 Ambos modelos tienen performance similar")
        else:
            print(f"\n🌐 Modelo GLOBAL es mejor ({r2_diff:.3f} R²)")
        
        # Analizar feature count
        global_features = len(global_info['features'])
        nearby_features = len(metrics_nearby.get('features', []))
        
        print(f"\n📊 COMPLEJIDAD:")
        print(f"   Global: {global_features} features")
        print(f"   Cercanas: {nearby_features} features")
        print(f"   Reducción: {((global_features - nearby_features) / global_features * 100):.1f}%")
        
    except FileNotFoundError:
        print("⚠️  No se encontró el modelo global para comparar")
    except Exception as e:
        print(f"⚠️  Error al cargar modelo global: {e}")

def analyze_predictions(y_val, y_pred, model_name="Cercanas"):
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

def main():
    """
    Pipeline principal de entrenamiento con estaciones cercanas.
    """
    print("🔍 ENTRENAMIENTO CON ESTACIONES CERCANAS")
    print("="*70)
    print(f"🎯 Estación objetivo: {TARGET_STATION_NAME} (ID: {TARGET_STATION_ID})")
    
    # 1. Cargar datos
    print("\n📂 Cargando datasets...")
    train_df = pd.read_csv('../data/train_nearby.csv')
    val_df = pd.read_csv('../data/val_nearby.csv')
    
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
    model = train_xgboost_nearby(X_train, y_train, X_val, y_val)
    
    # 4. Evaluar modelo
    metrics, y_pred = evaluate_model(
        model, X_train, y_train, X_val, y_val, 
        model_name="XGBoost Cercanas"
    )
    
    # 5. Análisis detallado de predicciones
    analyze_predictions(y_val, y_pred, "Cercanas")
    
    # 6. Feature importance
    importance_df = print_feature_importance(model, feature_names, top_n=20)
    
    # 7. Comparar con modelo global
    metrics['features'] = feature_names  # Agregar features para comparación
    compare_with_global(metrics)
    
    # 8. Guardar modelo
    target_info = {
        'approach': 'nearby_features',
        'description': 'Usa despachos solo de estaciones geográficamente cercanas',
        'num_features': len(feature_names),
        'train_period': f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}",
        'val_period': f"{val_df['timestamp'].min()} to {val_df['timestamp'].max()}"
    }
    
    save_model(
        model=model,
        features=feature_names,
        metrics=metrics,
        model_name="xgboost_nearby_features",
        target_info=target_info
    )
    
    # 9. Guardar feature importance
    if importance_df is not None:
        importance_df.to_csv('../data/feature_importance_nearby.csv', index=False)
        print(f"💾 Feature importance guardada: ../data/feature_importance_nearby.csv")
    
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
    print("   Abrir notebook: ../notebooks/analisis_estacion_individual.ipynb")

if __name__ == "__main__":
    main() 