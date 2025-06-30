import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

class BikeXGBoostTrainer:
    """
    Entrenador de XGBoost para predicción de arribos con split temporal
    """
    
    def __init__(self, poisson_max_delta_step=0.7):
        self.model = None
        self.feature_names = None
        self.poisson_max_delta_step = poisson_max_delta_step
        
    def identify_valid_features(self, df):
        """Identificar features válidas (sin información del futuro)"""
        
        # Features que contienen información del futuro (NO usar)
        future_features = [
            'año_destino', 'mes_destino', 'dia_destino', 'hora_destino', 
            'minuto_destino', 'segundo_destino',
            'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 
            'minuto_intervalo', 'fecha_intervalo','id_estacion_destino', 'duracion_recorrido',
            'N_arribos_intervalo',  # Variable objetivo
            'N_salidas_intervalo'   # También futuro
        ]
        
        # Features válidas (información disponible en tiempo T)
        valid_features = [
            # IDs y features básicas
            'id_recorrido', 'id_estacion_origen', 
            'id_usuario', 'modelo_bicicleta', 'estacion_referencia',
            
            # Features temporales de origen (conocidas)
            'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
            'dia_semana', 'es_finde', 'estacion_del_año',
            
            # Features de usuario (conocidas)
            'edad_usuario', 'año_alta', 'mes_alta', 
            'genero_FEMALE', 'genero_MALE', 'genero_OTHER', 'usuario_registrado',
            
            # Features geográficas (conocidas)
            'zona_destino_cluster', 'zona_origen_cluster',
            'cantidad_estaciones_cercanas_destino', 'cantidad_estaciones_cercanas_origen',
            
            # Features históricas (información del pasado)
            'id_estacion_destino_prev_1', 'id_estacion_destino_prev_2', 'id_estacion_destino_prev_3',
            'barrio_destino_prev_1', 'barrio_destino_prev_2', 'barrio_destino_prev_3',
            'cantidad_estaciones_cercanas_destino_prev_1', 'cantidad_estaciones_cercanas_destino_prev_2',
            'cantidad_estaciones_cercanas_destino_prev_3',
            'mes_destino_prev_1', 'mes_destino_prev_2', 'mes_destino_prev_3',
            'dia_destino_prev_1', 'dia_destino_prev_2', 'dia_destino_prev_3',
            'hora_destino_prev_1', 'hora_destino_prev_2', 'hora_destino_prev_3',
            'minuto_destino_prev_1', 'minuto_destino_prev_2', 'minuto_destino_prev_3',
            'segundo_destino_prev_1', 'segundo_destino_prev_2', 'segundo_destino_prev_3',
            'N_ARRIBOS_prev_1', 'N_SALIDAS_prev_1', 'N_ARRIBOS_prev_2', 'N_SALIDAS_prev_2',
            'N_ARRIBOS_prev_3', 'N_SALIDAS_prev_3', 'N_ARRIBOS_prev_4', 'N_SALIDAS_prev_4',
            'N_ARRIBOS_prev_5', 'N_SALIDAS_prev_5', 'N_ARRIBOS_prev_6', 'N_SALIDAS_prev_6'
        ]
        
        # Filtrar solo las que existen en el dataset
        available_features = [f for f in valid_features if f in df.columns]
        
        print(f"Features válidas identificadas: {len(available_features)}")
        print(f"Features del futuro excluidas: {len(future_features)}")
        
        return available_features
    
    def temporal_split(self, df, train_ratio=0.7, val_ratio=0.15):
        """Split temporal del dataset (no random)"""
        
        # Crear columna de fecha de referencia para ordenar temporalmente
        if 'fecha_intervalo' in df.columns:
            df['fecha_ref'] = pd.to_datetime(df['fecha_intervalo'])
        else:
            # Usar fecha origen como referencia
            df['fecha_ref'] = pd.to_datetime(df['año_origen'].astype(str) + '-' + 
                                           df['mes_origen'].astype(str) + '-' + 
                                           df['dia_origen'].astype(str))
        
        # Ordenar temporalmente
        df_sorted = df.sort_values('fecha_ref').reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split temporal
        train_df = df_sorted.iloc[:n_train].copy()
        val_df = df_sorted.iloc[n_train:n_train+n_val].copy()
        test_df = df_sorted.iloc[n_train+n_val:].copy()
        
        print(f"Split temporal realizado:")
        print(f"  Train: {len(train_df):,} filas ({train_ratio:.1%})")
        print(f"  Val:   {len(val_df):,} filas ({val_ratio:.1%})")
        print(f"  Test:  {len(test_df):,} filas ({1-train_ratio-val_ratio:.1%})")
        
        if len(train_df) > 0 and len(test_df) > 0:
            print(f"  Fechas train: {train_df['fecha_ref'].min()} a {train_df['fecha_ref'].max()}")
            print(f"  Fechas test:  {test_df['fecha_ref'].min()} a {test_df['fecha_ref'].max()}")
        
        return train_df, val_df, test_df
    
    def prepare_data(self, df):
        """Preparar datos para entrenamiento"""
        
        # Identificar features válidas
        feature_columns = self.identify_valid_features(df)
        self.feature_names = feature_columns
        
        # Variable objetivo
        target = 'N_arribos_intervalo'
        
        # Verificar que existe la variable objetivo
        if target not in df.columns:
            raise ValueError(f"Variable objetivo {target} no encontrada en el dataset")
        
        # Preparar X e y
        X = df[feature_columns].copy()
        y = df[target].copy()
        
        # Manejar valores faltantes
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Asegurar que y sea no negativo (requerido para Poisson)
        y = np.maximum(y, 0)
        
        print(f"Datos preparados: {X.shape[0]} filas, {X.shape[1]} features")
        print(f"Target stats: min={y.min()}, max={y.max()}, mean={y.mean():.2f}")
        
        return X, y
    
    def train(self, df, model_params=None):
        """Entrenar modelo XGBoost con loss de Poisson"""
        
        print("=== INICIANDO ENTRENAMIENTO XGBOOST ===")
        
        # Parámetros por defecto
        if model_params is None:
            model_params = {
                'objective': 'count:poisson',  # Loss de Poisson
                'eval_metric': 'poisson-nloglik',
                'max_depth': 22,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'max_delta_step': self.poisson_max_delta_step,  # Importante para Poisson
                'tree_method': 'gpu_hist'  # Usar GPU para entrenamiento
            }
        
        # Split temporal
        train_df, val_df, test_df = self.temporal_split(df)
        
        # Preparar datos
        X_train, y_train = self.prepare_data(train_df)
        X_val, y_val = self.prepare_data(val_df)
        X_test, y_test = self.prepare_data(test_df)
        
        # Crear datasets de XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=self.feature_names)
        
        # Entrenar modelo
        print("Entrenando XGBoost...")
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        self.model = xgb.train(
            params=model_params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # Evaluación
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        test_pred = self.model.predict(dtest)
        
        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("\n=== RESULTADOS DEL ENTRENAMIENTO ===")
        print(f"RMSE - Train: {train_rmse:.4f}, Val: {val_rmse:.4f}, Test: {test_rmse:.4f}")
        print(f"MAE  - Train: {train_mae:.4f}, Val: {val_mae:.4f}, Test: {test_mae:.4f}")
        print(f"R2   - Train: {train_r2:.4f}, Val: {val_r2:.4f}, Test: {test_r2:.4f}")
        
        # Guardar resultados
        results = {
            'model': self.model,
            'feature_names': self.feature_names,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'model_params': model_params,
            'n_features': len(self.feature_names)
        }
        
        return results, (X_test, y_test, test_pred)
    
    def save_model(self, filepath, results):
        """Guardar modelo entrenado"""
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'results': results
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath):
        """Cargar modelo entrenado"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        print(f"Modelo cargado desde: {filepath}")
        return model_data['results'] 