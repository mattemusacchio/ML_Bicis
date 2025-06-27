import pandas as pd
import json
import numpy as np

def extract_stations_from_trips():
    """Extraer todas las estaciones únicas de los datos de trips"""
    print("🔍 Extrayendo estaciones del dataset...")
    
    # Cargar datos (solo las columnas necesarias para optimizar memoria)
    columns_needed = [
        'id_estacion_origen', 'nombre_estacion_origen', 
        'direccion_estacion_origen', 'long_estacion_origen', 'lat_estacion_origen',
        'id_estacion_destino', 'nombre_estacion_destino', 
        'direccion_estacion_destino', 'long_estacion_destino', 'lat_estacion_destino'
    ]
    
    print("📊 Cargando datos de trips 2024...")
    df = pd.read_csv('data/raw/trips_2024.csv', usecols=columns_needed)
    
    # Extraer estaciones origen
    print("🚀 Procesando estaciones origen...")
    estaciones_origen = df[['id_estacion_origen', 'nombre_estacion_origen', 
                           'direccion_estacion_origen', 'long_estacion_origen', 
                           'lat_estacion_origen']].copy()
    estaciones_origen.columns = ['id', 'name', 'address', 'lng', 'lat']
    
    # Extraer estaciones destino
    print("🎯 Procesando estaciones destino...")
    estaciones_destino = df[['id_estacion_destino', 'nombre_estacion_destino', 
                            'direccion_estacion_destino', 'long_estacion_destino', 
                            'lat_estacion_destino']].copy()
    estaciones_destino.columns = ['id', 'name', 'address', 'lng', 'lat']
    
    # Combinar y obtener estaciones únicas
    print("🔗 Combinando estaciones...")
    all_stations = pd.concat([estaciones_origen, estaciones_destino], ignore_index=True)
    
    # Eliminar duplicados basado en el ID
    unique_stations = all_stations.drop_duplicates(subset=['id']).reset_index(drop=True)
    
    # Limpiar datos
    print("🧹 Limpiando datos...")
    # Eliminar filas con valores nulos en coordenadas
    unique_stations = unique_stations.dropna(subset=['lat', 'lng'])
    
    # Convertir a tipos correctos
    unique_stations['id'] = unique_stations['id'].astype(str)
    unique_stations['lat'] = pd.to_numeric(unique_stations['lat'], errors='coerce')
    unique_stations['lng'] = pd.to_numeric(unique_stations['lng'], errors='coerce')
    
    # Eliminar coordenadas inválidas
    unique_stations = unique_stations[
        (unique_stations['lat'].between(-35, -34)) &  # Buenos Aires lat range
        (unique_stations['lng'].between(-59, -58))     # Buenos Aires lng range
    ]
    
    # Calcular estadísticas de uso (frecuencia como proxy para popularidad)
    print("📈 Calculando estadísticas de uso...")
    origen_counts = df['id_estacion_origen'].value_counts()
    destino_counts = df['id_estacion_destino'].value_counts()
    total_counts = origen_counts.add(destino_counts, fill_value=0)
    
    # Agregar estadísticas a las estaciones
    unique_stations['total_trips'] = unique_stations['id'].astype(float).map(total_counts).fillna(0)
    unique_stations['popularity_score'] = (unique_stations['total_trips'] / unique_stations['total_trips'].max() * 100).round(1)
    
    # Simular disponibilidad actual (para la demo)
    np.random.seed(42)  # Para reproducibilidad
    unique_stations['available_bikes'] = np.random.randint(0, 30, len(unique_stations))
    unique_stations['total_docks'] = unique_stations['available_bikes'] + np.random.randint(0, 15, len(unique_stations))
    
    # Determinar status basado en disponibilidad
    def get_status(row):
        if row['available_bikes'] == 0:
            return 'empty'
        elif row['available_bikes'] >= row['total_docks'] * 0.8:
            return 'full'
        else:
            return 'active'
    
    unique_stations['status'] = unique_stations.apply(get_status, axis=1)
    
    print(f"✅ Procesadas {len(unique_stations)} estaciones únicas")
    print(f"📍 Rango de coordenadas:")
    print(f"   Latitud: {unique_stations['lat'].min():.4f} a {unique_stations['lat'].max():.4f}")
    print(f"   Longitud: {unique_stations['lng'].min():.4f} a {unique_stations['lng'].max():.4f}")
    
    return unique_stations

def save_stations_data(stations_df):
    """Guardar datos de estaciones en formato JSON para la página web"""
    print("💾 Guardando datos de estaciones...")
    
    # Convertir a lista de diccionarios
    stations_list = []
    for _, row in stations_df.iterrows():
        station = {
            'id': str(row['id']),
            'name': str(row['name']) if pd.notna(row['name']) else f"Estación {row['id']}",
            'address': str(row['address']) if pd.notna(row['address']) else "Dirección no disponible",
            'lat': float(row['lat']),
            'lng': float(row['lng']),
            'available_bikes': int(row['available_bikes']),
            'total_docks': int(row['total_docks']),
            'status': str(row['status']),
            'total_trips': int(row['total_trips']),
            'popularity_score': float(row['popularity_score'])
        }
        stations_list.append(station)
    
    # Crear estructura para la API
    api_data = {
        'stations': stations_list,
        'metadata': {
            'total_stations': len(stations_list),
            'last_updated': pd.Timestamp.now().isoformat(),
            'data_source': 'trips_2024.csv',
            'coordinate_system': 'WGS84'
        }
    }
    
    # Guardar en el directorio de la página
    output_path = 'pagina/public/stations_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(api_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Datos guardados en: {output_path}")
    
    # También crear un archivo más pequeño para desarrollo
    sample_stations = stations_list[:50]  # Solo primeras 50 para pruebas rápidas
    sample_data = {
        'stations': sample_stations,
        'metadata': {
            'total_stations': len(sample_stations),
            'note': 'Sample data for development',
            'last_updated': pd.Timestamp.now().isoformat()
        }
    }
    
    sample_path = 'pagina/public/stations_sample.json'
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Datos de muestra guardados en: {sample_path}")
    
    return api_data

def create_stations_api_endpoint():
    """Crear endpoint API para servir datos de estaciones"""
    print("🌐 Creando endpoint API...")
    
    api_code = '''import { NextResponse } from 'next/server'
import path from 'path'
import { promises as fs } from 'fs'

export async function GET() {
  try {
    // Leer datos de estaciones desde el archivo JSON
    const jsonDirectory = path.join(process.cwd(), 'public')
    const fileContents = await fs.readFile(jsonDirectory + '/stations_data.json', 'utf8')
    const data = JSON.parse(fileContents)
    
    return NextResponse.json({
      success: true,
      data: data.stations,
      metadata: data.metadata
    })
  } catch (error) {
    console.error('Error loading stations data:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to load stations data',
        data: []
      }, 
      { status: 500 }
    )
  }
}

// Opcional: endpoint para obtener datos en tiempo real (simulado)
export async function POST() {
  try {
    // Simular actualización de datos en tiempo real
    const jsonDirectory = path.join(process.cwd(), 'public')
    const fileContents = await fs.readFile(jsonDirectory + '/stations_data.json', 'utf8')
    const data = JSON.parse(fileContents)
    
    // Simular cambios en disponibilidad
    const updatedStations = data.stations.map(station => ({
      ...station,
      available_bikes: Math.max(0, station.available_bikes + Math.floor(Math.random() * 6 - 3)),
      last_updated: new Date().toISOString()
    }))
    
    return NextResponse.json({
      success: true,
      data: updatedStations,
      metadata: {
        ...data.metadata,
        last_updated: new Date().toISOString(),
        note: 'Real-time simulation'
      }
    })
  } catch (error) {
    return NextResponse.json(
      { 
        success: false, 
        error: 'Failed to update stations data'
      }, 
      { status: 500 }
    )
  }
}'''

    # Guardar endpoint API
    api_path = 'pagina/app/api/stations/route.ts'
    with open(api_path, 'w', encoding='utf-8') as f:
        f.write(api_code)
    
    print(f"✅ Endpoint API creado en: {api_path}")

def main():
    """Función principal"""
    print("=== EXTRACCIÓN DE DATOS DE ESTACIONES REALES ===")
    
    try:
        # Extraer estaciones
        stations_df = extract_stations_from_trips()
        
        # Guardar datos
        api_data = save_stations_data(stations_df)
        
        # Crear endpoint API
        create_stations_api_endpoint()
        
        print("\n=== RESUMEN ===")
        print(f"✅ {len(stations_df)} estaciones extraídas exitosamente")
        print(f"📍 Datos guardados en archivos JSON")
        print(f"🌐 Endpoint API creado")
        print(f"🎯 La página web ahora puede usar datos reales de estaciones")
        
        # Mostrar algunas estadísticas
        print(f"\n📊 ESTADÍSTICAS:")
        print(f"   Estación más popular: {stations_df.loc[stations_df['total_trips'].idxmax(), 'name']}")
        print(f"   Total de trips analizados: {stations_df['total_trips'].sum():,.0f}")
        print(f"   Promedio de trips por estación: {stations_df['total_trips'].mean():.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main() 