import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class XGBoostBikePredictorNoLeakage:
    """
    Predictor XGBoost para bicicletas SIN DATA LEAKAGE
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.model = None
        self.training_features = None
        self.target = 'N_arribos_intervalo'
        
    def load_and_prepare_data(self, filepath, sample_size=10000000):
        """Cargar y preparar datos sin leakage"""
        print("=== 🛡️ CARGANDO DATOS SIN DATA LEAKAGE ===")
        
        # Cargar muestra del dataset
        print(f"Cargando muestra de {sample_size} registros...")
        df = pd.read_csv(filepath, nrows=sample_size)
        print(f"Dataset cargado: {df.shape}")
        
        # Verificar columnas requeridas
        required_cols = [self.target, 'fecha_intervalo', 'estacion_referencia']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️ Columnas faltantes: {missing_cols}")
            
        # Convertir fecha
        if 'fecha_intervalo' in df.columns:
            df['fecha_intervalo'] = pd.to_datetime(df['fecha_intervalo'])
        
        return df
    
    def select_safe_features(self, df):
        """Seleccionar solo features seguras (sin data leakage)"""
        print("Seleccionando features sin data leakage...")
        
        # Features seguras definidas manualmente
        safe_features = [
            # Features de usuario (conocidas antes del viaje)
            'id_usuario', 'modelo_bicicleta', 'genero_FEMALE', 'genero_MALE', 'genero_OTHER',
            'usuario_registrado', 'edad_usuario', 'año_alta', 'mes_alta',
            
            # Features de origen (conocidas al inicio)
            'id_estacion_origen', 'barrio_origen', 'zona_origen_cluster', 
            'cantidad_estaciones_cercanas_origen',
            
            # Features temporales del origen
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen',
            'dia_semana', 'es_finde', 'estacion_del_anio',
            
            # Features de destino PLANIFICADO (no futuro real)
            'barrio_destino', 'zona_destino_cluster', 'cantidad_estaciones_cercanas_destino',
            
            # Features de ventana temporal (estructura, no contenido)
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
            
            # LAGs históricos (información del pasado)
            'N_ARRIBOS_LAG1', 'N_SALIDAS_LAG1', 'N_ARRIBOS_LAG2', 'N_SALIDAS_LAG2',
            'N_ARRIBOS_LAG3', 'N_SALIDAS_LAG3', 'N_ARRIBOS_LAG4', 'N_SALIDAS_LAG4',
            'N_ARRIBOS_LAG5', 'N_SALIDAS_LAG5', 'N_ARRIBOS_LAG6', 'N_SALIDAS_LAG6',
            'N_ARRIBOS_PROM_2INT', 'N_SALIDAS_PROM_2INT',
            
            # LAGs de viajes anteriores
            'id_estacion_origen_LAG1', 'id_estacion_origen_LAG2', 'id_estacion_origen_LAG3',
            'barrio_origen_LAG1', 'barrio_origen_LAG2', 'barrio_origen_LAG3',
            'cantidad_estaciones_cercanas_origen_LAG1', 'cantidad_estaciones_cercanas_origen_LAG2',
            'cantidad_estaciones_cercanas_origen_LAG3'
        ]
        
        # Filtrar features que realmente existen
        available_features = [col for col in safe_features if col in df.columns]
        missing_features = [col for col in safe_features if col not in df.columns]
        
        print(f"Features disponibles: {len(available_features)}")
        print(f"Features faltantes: {len(missing_features)}")
        
        if len(missing_features) > 0:
            print(f"Features faltantes: {missing_features[:10]}...")
        
        # Asegurar tipos numéricos
        for col in available_features:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        self.training_features = available_features
        return available_features
    
    def temporal_split(self, df):
        """División temporal correcta para evitar data leakage"""
        print("Realizando división temporal...")
        
        if 'fecha_intervalo' not in df.columns:
            print("⚠️ No se puede hacer división temporal, usando división aleatoria")
            return train_test_split(df, test_size=0.2, random_state=42)
        
        # Ordenar por fecha
        df = df.sort_values('fecha_intervalo').reset_index(drop=True)
        
        # División temporal: 80% primeras fechas para train, 20% últimas para test
        split_idx = int(len(df) * 0.8)
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"Train: {len(train_df)} registros ({train_df['fecha_intervalo'].min()} - {train_df['fecha_intervalo'].max()})")
        print(f"Test: {len(test_df)} registros ({test_df['fecha_intervalo'].min()} - {test_df['fecha_intervalo'].max()})")
        
        return train_df, test_df
    
    def audit_features(self, X_train, y_train):
        """Auditar features para detectar posible leakage"""
        print("\n=== 🔍 AUDITORÍA DE FEATURES ===")
        
        # Calcular correlaciones
        correlations = {}
        for feature in X_train.columns:
            corr = y_train.corr(X_train[feature])
            if not pd.isna(corr):
                correlations[feature] = abs(corr)
        
        # Ordenar por correlación
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 10 correlaciones con target:")
        suspicious_features = []
        
        for i, (feature, corr) in enumerate(sorted_corr[:10]):
            status = ""
            if corr > 0.9:
                status = "🚨 EXTREMA"
                suspicious_features.append(feature)
            elif corr > 0.8:
                status = "⚠️  ALTA"
                suspicious_features.append(feature)
            elif corr > 0.5:
                status = "🔶 MEDIA"
            else:
                status = "✅ OK"
            
            print(f"  {i+1:2d}. {feature:<30} {corr:.3f} {status}")
        
        if suspicious_features:
            print(f"\n⚠️ Features sospechosas detectadas: {suspicious_features}")
            print("Considere investigar estas features para posible data leakage")
        else:
            print("\n✅ No se detectaron correlaciones sospechosas")
        
        return correlations
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Entrenar modelo XGBoost"""
        print("\n=== 🚀 ENTRENANDO MODELO XGBOOST ===")
        
        # Configuración del modelo
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Configurar GPU si está disponible
        if self.use_gpu:
            try:
                params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0
                })
                print("✅ Usando GPU para entrenamiento")
            except:
                print("⚠️  GPU no disponible, usando CPU")
                self.use_gpu = False
        
        # Crear y entrenar modelo
        self.model = xgb.XGBRegressor(**params)
        
        # Entrenar con early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100,
            early_stopping_rounds=50
        )
        
        print("✅ Entrenamiento completado")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluar modelo y mostrar métricas"""
        print("\n=== 📊 EVALUACIÓN DEL MODELO ===")
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        
        # Métricas
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE:  {mae:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        
        # Verificar si las métricas son realistas
        if r2 > 0.95:
            print("🚨 ¡ALERTA! R² extremadamente alto - posible data leakage")
        elif r2 > 0.8:
            print("⚠️  R² alto - verificar posible overfitting")
        elif r2 > 0.5:
            print("✅ R² razonable para este tipo de problema")
        else:
            print("🔶 R² bajo - considerar más features o mejor feature engineering")
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def plot_feature_importance(self, top_n=20):
        """Graficar importancia de features"""
        print(f"\n=== 📈 IMPORTANCIA DE FEATURES (Top {top_n}) ===")
        
        if self.model is None:
            print("⚠️ Modelo no entrenado")
            return
        
        # Obtener importancia
        importance = self.model.feature_importances_
        feature_names = self.training_features
        
        # Crear DataFrame y ordenar
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Mostrar top features
        print("Top features:")
        for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        # Verificar distribución de importancia
        top_feature_importance = importance_df.iloc[0]['importance']
        if top_feature_importance > 0.5:
            print(f"🚨 ¡ALERTA! Feature dominante: {importance_df.iloc[0]['feature']} ({top_feature_importance:.3f})")
        
        # Graficar
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df.head(top_n),
            y='feature',
            x='importance'
        )
        plt.title(f'Feature Importance - Top {top_n}')
        plt.tight_layout()
        plt.savefig('feature_importance_no_leakage.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def save_model(self, filepath='models/xgboost_no_leakage.pkl'):
        """Guardar modelo entrenado"""
        if self.model is None:
            print("⚠️ No hay modelo para guardar")
            return
        
        # Crear metadata
        metadata = {
            'training_features': self.training_features,
            'target': self.target,
            'use_gpu': self.use_gpu,
            'model_type': 'XGBoost_No_Leakage'
        }
        
        # Guardar modelo
        joblib.dump(self.model, filepath)
        joblib.dump(metadata, filepath.replace('.pkl', '_metadata.pkl'))
        
        print(f"✅ Modelo guardado: {filepath}")
        
    def run_complete_pipeline(self, data_path, sample_size=100000000):
        """Ejecutar pipeline completo"""
        print("=== 🎯 PIPELINE COMPLETO SIN DATA LEAKAGE ===")
        
        # 1. Cargar datos
        df = self.load_and_prepare_data(data_path, sample_size)
        
        # 2. Seleccionar features seguras
        feature_cols = self.select_safe_features(df)
        
        if len(feature_cols) == 0:
            print("❌ No se encontraron features válidas")
            return None
        
        # 3. División temporal
        train_df, test_df = self.temporal_split(df)
        
        # 4. Preparar X, y
        X_train = train_df[feature_cols]
        y_train = train_df[self.target]
        X_test = test_df[feature_cols]
        y_test = test_df[self.target]
        
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        
        # 5. Auditar features
        correlations = self.audit_features(X_train, y_train)
        
        # 6. Entrenar modelo
        model = self.train_model(X_train, y_train, X_test, y_test)
        
        # 7. Evaluar
        results = self.evaluate_model(X_test, y_test)
        
        # 8. Feature importance
        importance_df = self.plot_feature_importance()
        
        # 9. Guardar modelo
        self.save_model()
        
        return {
            'model': model,
            'results': results,
            'importance': importance_df,
            'correlations': correlations
        }

def main():
    """Función principal"""
    
    # Configuración
    data_path = 'data/processed/trips_features_engineered_fixed.csv'
    sample_size = 100000000  # Usar muestra más pequeña para evitar memoria
    
    # Crear predictor
    predictor = XGBoostBikePredictorNoLeakage(use_gpu=True)
    
    # Ejecutar pipeline
    results = predictor.run_complete_pipeline(data_path, sample_size)
    
    if results:
        print("\n=== ✅ PIPELINE COMPLETADO ===")
        print(f"R²: {results['results']['r2']:.4f}")
        print("Modelo guardado en: models/xgboost_no_leakage.pkl")
    
    return results

if __name__ == "__main__":
    results = main() 