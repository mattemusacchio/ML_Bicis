#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.feature_engineering import BikeFeatureEngineering

def main():
    """Ejecutar pipeline completo de feature engineering"""
    
    # Configuración
    input_file = '../data/processed/trips_enriched.csv'
    output_file = '../data/processed/trips_final_features.csv'
    
    # Verificar que existe el archivo de entrada
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el archivo {input_file}")
        return
    
    # Crear pipeline
    fe = BikeFeatureEngineering(
        time_window_minutes=30,
        n_clusters=16
    )
    
    # Ejecutar transformación
    try:
        df_processed = fe.transform(input_file, output_file)
        print(f"\n✅ Pipeline completado exitosamente")
        print(f"   Archivo generado: {output_file}")
        print(f"   Forma final: {df_processed.shape}")
        
        # Mostrar algunas columnas de ejemplo
        print(f"\nPrimeras 5 filas de N_arribos_intervalo:")
        print(df_processed['N_arribos_intervalo'].head())
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 