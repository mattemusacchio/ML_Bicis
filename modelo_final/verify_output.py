#!/usr/bin/env python3

import pandas as pd

def verify_columns():
    """Verificar que las columnas de salida coincidan con las requeridas"""
    
    # Columnas solicitadas (objetivo)
    expected_columns = [
        'id_recorrido', 'duracion_recorrido', 'id_estacion_origen', 'id_estacion_destino',
        'id_usuario', 'modelo_bicicleta', 'dia_semana', 'es_finde', 'estacion_del_año',
        'edad_usuario', 'año_alta', 'mes_alta', 'genero_FEMALE', 'genero_MALE', 'genero_OTHER',
        'usuario_registrado', 'zona_destino_cluster', 'zona_origen_cluster',
        'cantidad_estaciones_cercanas_destino', 'cantidad_estaciones_cercanas_origen',
        'año_origen', 'mes_origen', 'dia_origen', 'hora_origen', 'minuto_origen', 'segundo_origen',
        'año_destino', 'mes_destino', 'dia_destino', 'hora_destino', 'minuto_destino', 'segundo_destino',
        'año_intervalo', 'mes_intervalo', 'dia_intervalo', 'hora_intervalo', 'minuto_intervalo',
        'fecha_intervalo',
        'id_estacion_destino_prev_1', 'id_estacion_destino_prev_2', 'id_estacion_destino_prev_3',
        'barrio_destino_prev_1', 'barrio_destino_prev_2', 'barrio_destino_prev_3',
        'cantidad_estaciones_cercanas_destino_prev_1', 'cantidad_estaciones_cercanas_destino_prev_2',
        'cantidad_estaciones_cercanas_destino_prev_3',
        'mes_destino_prev_1', 'mes_destino_prev_2', 'mes_destino_prev_3',
        'dia_destino_prev_1', 'dia_destino_prev_2', 'dia_destino_prev_3',
        'hora_destino_prev_1', 'hora_destino_prev_2', 'hora_destino_prev_3',
        'minuto_destino_prev_1', 'minuto_destino_prev_2', 'minuto_destino_prev_3',
        'segundo_destino_prev_1', 'segundo_destino_prev_2', 'segundo_destino_prev_3',
        'estacion_referencia', 'N_arribos_intervalo', 'N_salidas_intervalo',
        'N_ARRIBOS_prev_1', 'N_SALIDAS_prev_1', 'N_ARRIBOS_prev_2', 'N_SALIDAS_prev_2',
        'N_ARRIBOS_prev_3', 'N_SALIDAS_prev_3', 'N_ARRIBOS_prev_4', 'N_SALIDAS_prev_4',
        'N_ARRIBOS_prev_5', 'N_SALIDAS_prev_5', 'N_ARRIBOS_prev_6', 'N_SALIDAS_prev_6'
    ]
    
    # Leer archivo generado
    output_file = '../data/processed/trips_final_features.csv'
    try:
        df = pd.read_csv(output_file, nrows=1)
        actual_columns = list(df.columns)
        
        print("=== VERIFICACIÓN DE COLUMNAS ===")
        print(f"Columnas esperadas: {len(expected_columns)}")
        print(f"Columnas generadas: {len(actual_columns)}")
        
        # Verificar coincidencias
        missing = set(expected_columns) - set(actual_columns)
        extra = set(actual_columns) - set(expected_columns)
        
        if not missing and not extra:
            print("✅ PERFECTO: Todas las columnas coinciden exactamente")
        else:
            if missing:
                print(f"❌ Columnas faltantes: {missing}")
            if extra:
                print(f"⚠️  Columnas extra: {extra}")
        
        # Verificar variable objetivo específicamente
        if 'N_arribos_intervalo' in actual_columns:
            print("✅ Variable objetivo N_arribos_intervalo presente")
        else:
            print("❌ Variable objetivo N_arribos_intervalo FALTANTE")
            
        # Mostrar muestra de datos
        print("\n=== MUESTRA DE DATOS ===")
        df_sample = pd.read_csv(output_file, nrows=3)
        print("Shape de muestra:", df_sample.shape)
        print("\nPrimeras 3 filas de N_arribos_intervalo:")
        if 'N_arribos_intervalo' in df_sample.columns:
            print(df_sample['N_arribos_intervalo'].values)
        
        print("\nPrimeras 3 filas de features prev:")
        prev_cols = [col for col in df_sample.columns if '_prev_' in col][:5]
        for col in prev_cols:
            print(f"{col}: {df_sample[col].values}")
            
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {output_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    verify_columns() 