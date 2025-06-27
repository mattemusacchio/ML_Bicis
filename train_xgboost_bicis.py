import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class XGBoostBikePredictor:
    """
    Clase para entrenar modelo XGBoost para predicción de arribos de bicicletas
    """
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.model = None
        self.feature_names = None
        self.target_column = 'N_arribos_intervalo'
        self.training_features = [
            'id_estacion_origen', 'id_usuario', 'modelo_bicicleta', 'barrio_origen',
            'dia_semana', 'es_finde', 'estacion_del_anio', 'edad_usuario', 'año_alta',
            'mes_alta', 'genero_FEMALE', 'genero_MALE', 'genero_OTHER',
            'usuario_registrado', 'zona_origen_cluster',
            'cantidad_estaciones_cercanas_origen', 'año_origen', 'mes_origen',
            'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo',
            'minuto_intervalo', 'N_SALIDAS_PROM_2INT', 'N_ARRIBOS_PROM_2INT',
            'id_estacion_destino_LAG1', 'id_estacion_destino_LAG2',
            'id_estacion_destino_LAG3', 'barrio_destino_LAG1', 'barrio_destino_LAG2',
            'barrio_destino_LAG3', 'cantidad_estaciones_cercanas_destino_LAG1',
            'cantidad_estaciones_cercanas_destino_LAG2',
            'cantidad_estaciones_cercanas_destino_LAG3', 'año_destino_LAG1',
            'año_destino_LAG2', 'año_destino_LAG3', 'mes_destino_LAG1',
            'mes_destino_LAG2', 'mes_destino_LAG3', 'dia_destino_LAG1',
            'dia_destino_LAG2', 'dia_destino_LAG3', 'hora_destino_LAG1',
            'hora_destino_LAG2', 'hora_destino_LAG3', 'minuto_destino_LAG1',
            'minuto_destino_LAG2', 'minuto_destino_LAG3', 'segundo_destino_LAG1',
            'segundo_destino_LAG2', 'segundo_destino_LAG3', 'N_ARRIBOS_LAG1',
            'N_SALIDAS_LAG1', 'N_ARRIBOS_LAG2', 'N_SALIDAS_LAG2', 'N_ARRIBOS_LAG3',
            'N_SALIDAS_LAG3', 'N_ARRIBOS_LAG4', 'N_SALIDAS_LAG4', 'N_ARRIBOS_LAG5',
            'N_SALIDAS_LAG5', 'N_ARRIBOS_LAG6', 'N_SALIDAS_LAG6'
        ]
        
    def load_data(self, filepath):
        """Cargar datos transformados"""
        print("Cargando dataset transformado...")
        df = pd.read_csv(filepath)
        print(f"Dataset cargado: {df.shape}")
        
        # Verificar que todas las features están presentes
        missing_features = set(self.training_features) - set(df.columns)
        if missing_features:
            print(f"⚠️  Features faltantes: {missing_features}")
            # Crear features faltantes con valor 0
            for feature in missing_features:
                df[feature] = 0
                
        return df
    
    def prepare_data(self, df):
        """Preparar datos para entrenamiento"""
        print("Preparando datos para entrenamiento...")
        
        # Filtrar filas con valores válidos en target
        df_clean = df.dropna(subset=[self.target_column])
        
        # Separar features y target
        X = df_clean[self.training_features].copy()
        y = df_clean[self.target_column].copy()
        
        # Manejar valores infinitos y NaN
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Convertir columnas categóricas a numéricas si es necesario
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"⚠️  Convertiendo columna categórica '{col}' a numérica...")
                # Si es categórica, usar factorize para convertir a números
                X[col] = pd.Categorical(X[col]).codes
        
        # Asegurar tipos numéricos
        X = X.astype(float)
        y = y.astype(float)
        
        print(f"Datos preparados: X={X.shape}, y={y.shape}")
        print(f"Target stats: min={y.min()}, max={y.max()}, mean={y.mean():.2f}")
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Dividir datos en entrenamiento y prueba"""
        print("Dividiendo datos...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        print(f"Entrenamiento: X={X_train.shape}, y={y_train.shape}")
        print(f"Prueba: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_xgb_params(self):
        """Obtener parámetros optimizados para XGBoost"""
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'n_estimators': 1000,
            'max_depth': 12,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'verbosity': 1
        }
        
        # Configurar GPU si está disponible
        if self.use_gpu:
            try:
                base_params.update({
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'predictor': 'gpu_predictor'
                })
                print("🚀 GPU habilitada para XGBoost")
            except:
                print("⚠️  GPU no disponible, usando CPU")
                self.use_gpu = False
        
        return base_params
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Entrenar el modelo XGBoost"""
        print("=== ENTRENANDO MODELO XGBOOST ===")
        
        # Obtener parámetros
        params = self.get_xgb_params()
        
        # Crear el modelo
        self.model = xgb.XGBRegressor(**params)
        
        # Entrenar con validación
        print("Iniciando entrenamiento...")
        start_time = datetime.now()
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100
        )
        
        training_time = datetime.now() - start_time
        print(f"✅ Entrenamiento completado en: {training_time}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluar el modelo entrenado"""
        print("=== EVALUANDO MODELO ===")
        
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
        
        # Guardar métricas
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics, y_pred
    
    def plot_feature_importance(self, top_n=20):
        """Visualizar importancia de features"""
        if self.model is None:
            print("Modelo no entrenado")
            return
            
        # Obtener importancias
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plotear top N features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Features más Importantes')
        plt.xlabel('Importancia')
        plt.tight_layout()
        plt.show()
        
        return feature_importance_df
    
    def plot_predictions(self, y_test, y_pred, sample_size=1000):
        """Visualizar predicciones vs valores reales"""
        # Muestrear para visualización
        if len(y_test) > sample_size:
            indices = np.random.choice(len(y_test), sample_size, replace=False)
            y_test_sample = y_test.iloc[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_test_sample = y_test
            y_pred_sample = y_pred
        
        # Crear subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_test_sample, y_pred_sample, alpha=0.6)
        axes[0].plot([y_test_sample.min(), y_test_sample.max()], 
                     [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
        axes[0].set_xlabel('Valores Reales')
        axes[0].set_ylabel('Predicciones')
        axes[0].set_title('Predicciones vs Valores Reales')
        
        # Residuos
        residuals = y_test_sample - y_pred_sample
        axes[1].scatter(y_pred_sample, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicciones')
        axes[1].set_ylabel('Residuos')
        axes[1].set_title('Gráfico de Residuos')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path='models/xgboost_bicis_model.pkl', 
                   metadata_path='models/xgboost_bicis_metadata.pkl'):
        """Guardar modelo y metadatos"""
        print("Guardando modelo...")
        
        # Crear directorio si no existe
        import os
        os.makedirs('models', exist_ok=True)
        
        # Guardar modelo
        joblib.dump(self.model, model_path)
        
        # Guardar metadatos
        metadata = {
            'feature_names': self.feature_names,
            'training_features': self.training_features,
            'target_column': self.target_column,
            'use_gpu': self.use_gpu,
            'model_type': 'XGBRegressor',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Modelo guardado en: {model_path}")
        print(f"✅ Metadatos guardados en: {metadata_path}")
    
    def load_model(self, model_path='models/xgboost_bicis_model.pkl',
                   metadata_path='models/xgboost_bicis_metadata.pkl'):
        """Cargar modelo y metadatos"""
        print("Cargando modelo...")
        
        # Cargar modelo
        self.model = joblib.load(model_path)
        
        # Cargar metadatos
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.training_features = metadata['training_features']
        self.target_column = metadata['target_column']
        
        print(f"✅ Modelo cargado desde: {model_path}")
        
        return metadata

def main():
    """Función principal para entrenar el modelo"""
    
    # Configuración
    data_path = 'data/processed/trips_features_engineered.csv'
    use_gpu = True
    
    print("=== ENTRENAMIENTO DE MODELO XGBOOST PARA PREDICCIÓN DE ARRIBOS ===")
    
    # Crear predictor
    predictor = XGBoostBikePredictor(use_gpu=use_gpu)
    
    # Cargar y preparar datos
    df = predictor.load_data(data_path)
    X, y = predictor.prepare_data(df)
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Entrenar modelo
    model = predictor.train_model(X_train, X_test, y_train, y_test)
    
    # Evaluar modelo
    metrics, y_pred = predictor.evaluate_model(X_test, y_test)
    
    # Visualizaciones
    feature_importance = predictor.plot_feature_importance(top_n=20)
    predictor.plot_predictions(y_test, y_pred)
    
    # Guardar modelo
    predictor.save_model()
    
    print("\n=== RESUMEN ===")
    print(f"Modelo entrenado exitosamente")
    print(f"Features utilizadas: {len(predictor.training_features)}")
    print(f"Samples de entrenamiento: {len(X_train)}")
    print(f"Samples de prueba: {len(X_test)}")
    print("\nMétricas finales:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return predictor, metrics, feature_importance

if __name__ == "__main__":
    predictor, metrics, feature_importance = main() 