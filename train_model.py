import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import time
import os
import gc
from joblib import Parallel, delayed

def prepare_features_and_targets(train_df, val_df):
    print("Preparando features y targets...")

    target_cols = [col for col in train_df.columns if col.startswith('arribos_count_')]
    exclude_cols = ['timestamp'] + target_cols
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Convertir a float32 para reducir el consumo de RAM
    X_train = train_df[feature_cols].astype(np.float32)
    y_train = train_df[target_cols].astype(np.float32)
    X_val = val_df[feature_cols].astype(np.float32)
    y_val = val_df[target_cols].astype(np.float32)

    print(f"X_train: {X_train.shape} - dtype: {X_train.dtypes.iloc[0]}")
    print(f"y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_val: {y_val.shape}")

    return X_train, y_train, X_val, y_val, feature_cols, target_cols

def entrenar_target(i, X_train, y_train, target_cols, xgb_params):
    print(f"Entrenando modelo para target: {target_cols[i]}")
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train.iloc[:, i])
    gc.collect()
    return model

def train_xgboost_model(X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("ENTRENAMIENTO XGBOOST")
    print("="*60)

    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': 1,
        'verbosity': 1,
        'tree_method': 'gpu_hist'
    }

    print("Configuración XGBoost:")
    for param, value in xgb_params.items():
        print(f"  {param}: {value}")

    n_cores = 4
    print(f"Entrenamiento paralelo usando {n_cores} núcleos...\n")

    start_time = time.time()

    models = Parallel(n_jobs=n_cores)(
        delayed(entrenar_target)(i, X_train, y_train, target_cols, xgb_params)
        for i in range(y_train.shape[1])
    )

    training_time = time.time() - start_time
    print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")

    # Limpiar memoria
    gc.collect()

    return models

def evaluate_model(models, X_train, y_train, X_val, y_val, target_cols):
    print("\n" + "="*60)
    print("EVALUACIÓN DEL MODELO")
    print("="*60)

    print("Generando predicciones...")

    # Usar float32 para ahorrar RAM
    y_train_pred = np.zeros_like(y_train, dtype=np.float32)
    y_val_pred = np.zeros_like(y_val, dtype=np.float32)

    for i, model in enumerate(models):
        y_train_pred[:, i] = model.predict(X_train)
        y_val_pred[:, i] = model.predict(X_val)

    # Métricas globales
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n📊 MÉTRICAS GLOBALES:")
    print(f"{'Conjunto':<12} {'MSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 50)
    print(f"{'Train':<12} {train_mse:<12.4f} {train_mae:<12.4f} {train_r2:<12.4f}")
    print(f"{'Validación':<12} {val_mse:<12.4f} {val_mae:<12.4f} {val_r2:<12.4f}")

    print(f"\n📍 ANÁLISIS POR ESTACIÓN (Top 10):")
    station_totals = y_val.sum().sort_values(ascending=False).head(10)
    print(f"{'Estación':<15} {'MSE':<10} {'MAE':<10} {'R²':<10} {'Total Arribos':<15}")
    print("-" * 70)

    for station_col, total_arribos in station_totals.items():
        idx = target_cols.index(station_col)
        mse = mean_squared_error(y_val.iloc[:, idx], y_val_pred[:, idx])
        mae = mean_absolute_error(y_val.iloc[:, idx], y_val_pred[:, idx])
        r2 = r2_score(y_val.iloc[:, idx], y_val_pred[:, idx])
        station_name = station_col.replace('arribos_count_', '')[:12]
        print(f"{station_name:<15} {mse:<10.3f} {mae:<10.3f} {r2:<10.3f} {total_arribos:<15.0f}")

    print(f"\n🔍 ANÁLISIS DE OVERFITTING:")
    overfitting_mse = (train_mse - val_mse) / val_mse * 100
    overfitting_mae = (train_mae - val_mae) / val_mae * 100
    print(f"Diferencia MSE: {overfitting_mse:.2f}% ({'Overfitting' if overfitting_mse < -10 else 'OK'})")
    print(f"Diferencia MAE: {overfitting_mae:.2f}% ({'Overfitting' if overfitting_mae < -10 else 'OK'})")

    # Liberar memoria
    gc.collect()

    return {
        'train_mse': train_mse, 'train_mae': train_mae, 'train_r2': train_r2,
        'val_mse': val_mse, 'val_mae': val_mae, 'val_r2': val_r2,
        'y_train_pred': y_train_pred, 'y_val_pred': y_val_pred
    }

# ========================== INICIO DEL PIPELINE =============================

print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO XGBOOST")
print("="*60)

train_norm = pd.read_csv('data/processed/train_norm.csv')
val_norm = pd.read_csv('data/processed/val_norm.csv')

X_train, y_train, X_val, y_val, feature_cols, target_cols = prepare_features_and_targets(train_norm, val_norm)

# Liberar memoria intermedia
del train_norm, val_norm
gc.collect()

xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
results = evaluate_model(xgb_model, X_train, y_train, X_val, y_val, target_cols)

print("\n💾 Guardando modelo...")

model_info = {
    'model': xgb_model,
    'feature_columns': feature_cols,
    'target_columns': target_cols,
    'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'metrics': {
        'val_mse': results['val_mse'],
        'val_mae': results['val_mae'],
        'val_r2': results['val_r2']
    }
}

with open('data/processed/xgb_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("✅ Modelo guardado en: data/processed/xgb_model.pkl")
print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
print(f"Modelo entrenado con {len(feature_cols)} features para predecir {len(target_cols)} targets")
print(f"R² en validación: {results['val_r2']:.4f}")
