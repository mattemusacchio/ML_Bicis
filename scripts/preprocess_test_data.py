#!/usr/bin/env python3
"""
Script para preprocesar el dataset de test usando el pipeline del modelo M0
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Importar funciones del modelo M0
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modelo_final import ModeloM0Predictor

def main():
    print("🔄 Preprocesando dataset de test...")
    
    # Cargar datos de test
    test_path = os.path.join('data', 'processed', 'trips_2024_test.csv')
    if not os.path.exists(test_path):
        print(f"❌ No se encontró el archivo {test_path}")
        sys.exit(1)
    
    # Cargar datos
    print(f"📖 Cargando datos de test...")
    df = pd.read_csv(test_path)
    print(f"✅ Datos cargados: {df.shape}")
    
    # Inicializar predictor
    predictor = ModeloM0Predictor()
    
    # Aplicar pipeline de preprocesamiento
    print(f"🔄 Aplicando pipeline de preprocesamiento...")
    df_processed = predictor.preprocesar_datos_raw(df)
    print(f"✅ Preprocesamiento completado: {df_processed.shape}")
    
    # Guardar datos preprocesados
    output_path = os.path.join('data', 'processed', 'trips_2024_test_processed.csv')
    df_processed.to_csv(output_path, index=False)
    print(f"💾 Datos preprocesados guardados en: {output_path}")

if __name__ == "__main__":
    main() 