"""
Pipeline Completo - Enfoque Simplificado
Ejecuta todo el proceso desde feature engineering hasta entrenamiento de modelos.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """
    Ejecuta un comando y maneja errores.
    
    Args:
        command: Comando a ejecutar
        description: Descripción del paso
    """
    print(f"\n🔄 {description}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Cambiar al directorio de scripts
        os.chdir('scripts')
        
        # Ejecutar comando
        result = subprocess.run(command, shell=True, capture_output=False, text=True)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} completado en {execution_time:.2f} segundos")
            return True
        else:
            print(f"❌ Error en {description}")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {e}")
        return False
    finally:
        # Volver al directorio principal
        os.chdir('..')

def check_requirements():
    """Verifica que existen los archivos necesarios"""
    print("🔍 VERIFICANDO REQUISITOS")
    print("="*50)
    
    required_files = [
        '../data/processed/trips_enriched.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"❌ No encontrado: {file_path}")
        else:
            print(f"✅ Encontrado: {file_path}")
    
    if missing_files:
        print(f"\n❌ FALTAN ARCHIVOS REQUERIDOS:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print(f"\nPor favor, ejecuta primero el notebook original para generar trips_enriched.csv")
        return False
    
    print(f"\n✅ Todos los archivos requeridos están disponibles")
    return True

def create_directories():
    """Crea los directorios necesarios"""
    directories = ['data', 'models']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Creado directorio: {directory}")
        else:
            print(f"✅ Directorio existe: {directory}")

def main():
    """
    Ejecuta el pipeline completo del enfoque simplificado.
    """
    start_time = time.time()
    
    print("🚀 PIPELINE COMPLETO - ENFOQUE SIMPLIFICADO")
    print("="*70)
    print(f"🕐 Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Directorio: {os.getcwd()}")
    
    # 1. Verificar requisitos
    if not check_requirements():
        print("\n❌ PIPELINE ABORTADO - Faltan archivos requeridos")
        return False
    
    # 2. Crear directorios
    print(f"\n📁 CREANDO DIRECTORIOS NECESARIOS")
    create_directories()
    
    # 3. Feature Engineering
    success = run_command(
        "python feature_engineering.py",
        "FEATURE ENGINEERING - Creando datasets para ambos enfoques"
    )
    if not success:
        print("\n❌ PIPELINE ABORTADO - Error en feature engineering")
        return False
    
    # 4. Entrenamiento modelo global
    success = run_command(
        "python train_global_features.py",
        "ENTRENAMIENTO MODELO GLOBAL - Usando todas las estaciones"
    )
    if not success:
        print("\n⚠️  Error en modelo global, continuando con modelo cercanas...")
    
    # 5. Entrenamiento modelo estaciones cercanas
    success = run_command(
        "python train_nearby_features.py",
        "ENTRENAMIENTO MODELO CERCANAS - Usando estaciones geográficamente cercanas"
    )
    if not success:
        print("\n❌ PIPELINE ABORTADO - Error en modelo cercanas")
        return False
    
    # 6. Resumen final
    total_time = time.time() - start_time
    
    print("\n🎉 PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
    print(f"🕐 Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n📁 ARCHIVOS GENERADOS:")
    generated_files = [
        "data/dataset_global.csv",
        "data/dataset_nearby.csv", 
        "data/train_global.csv",
        "data/val_global.csv",
        "data/train_nearby.csv",
        "data/val_nearby.csv",
        "models/xgboost_global_features.pkl",
        "models/xgboost_nearby_features.pkl"
    ]
    
    for file_path in generated_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   ✅ {file_path} ({file_size:.1f} MB)")
        else:
            print(f"   ❌ {file_path} (no generado)")
    
    print(f"\n🔗 PRÓXIMOS PASOS:")
    print("   1. Abrir: notebooks/analisis_estacion_individual.ipynb")
    print("   2. Revisar métricas y comparar modelos")
    print("   3. Analizar feature importance")
    print("   4. Planificar escalamiento a múltiples estaciones")
    
    print(f"\n📊 PARA VER RESULTADOS RÁPIDAMENTE:")
    print("   cd scripts")
    print("   python -c \"from utils import load_model; print('R² Global:', load_model('../models/xgboost_global_features.pkl')['metrics']['val_r2']); print('R² Cercanas:', load_model('../models/xgboost_nearby_features.pkl')['metrics']['val_r2'])\"")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎯 ¡ENFOQUE SIMPLIFICADO IMPLEMENTADO EXITOSAMENTE!")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline falló - revisar errores arriba")
        sys.exit(1) 